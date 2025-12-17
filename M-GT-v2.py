import numpy as np
import time
import pickle
import random
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm

# 使用 TransformerConv
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import (
    dropout_edge,
    negative_sampling,
    remove_self_loops,
    add_self_loops,
    to_scipy_sparse_matrix,
    degree,
)

from sklearn import metrics
import mlflow

# ==========================================
# 1. 参数配置
# ==========================================
CONFIG = {
    "EPOCH": 2000,
    "LR": 0.0005,
    "WEIGHT_DECAY": 1e-4,
    "DROPOUT_EDGE": 0.1,
    "DROPOUT_FEAT": 0.2,
    "DROPOUT_ATT": 0.3,
    "HIDDEN_1": 64,
    "HIDDEN_2": 32,
    "HEADS": 4,
    "BETA": True,
    "PE_DIM": 8,  # [新增] 位置编码的维度 (特征向量个数)
    "FOCAL_GAMMA": 2.0,  # [新增] Focal Loss 的聚焦参数
    "FOCAL_ALPHA": 0.6,  # [新增] Focal Loss 的平衡参数 (替代 pos_weight)
    "SEED": 42,
    "TARGET_RUN": 0,
    "TARGET_FOLD": 0,
    "SCHEDULER_PATIENCE": 15,
    "NOTE": "GT + LapPE + Focal Loss + LayerNorm",
    "EXP_NAME": 251215,
}


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to: {seed}")


device = torch.device("cpu")  # 如有 GPU 改为 "cuda"

# ==========================================
# 2. 数据加载与处理
# ==========================================
print("Loading data...")
# 假设文件存在
try:
    data = torch.load("./CPDB_data_v2.pt", weights_only=False)
except:
    print("Error: ./CPDB_data_v2.pt not found.")
    exit()

y_all = np.logical_or(data.y, data.y_te)
Y = torch.tensor(y_all).type(torch.float32).to(device)

if not isinstance(data.mask, torch.Tensor):
    mask_tensor = torch.tensor(data.mask)
    mask_te_tensor = torch.tensor(data.mask_te)
else:
    mask_tensor = data.mask
    mask_te_tensor = data.mask_te

# 特征处理
data.x = data.x[:, :48]
try:
    datas = torch.load("./data/str_fearures.pkl", map_location="cpu")
    data.x = torch.cat((data.x, datas), 1)
except:
    pass

data.x = data.x.float()


# --- [新增] 计算 Laplacian Positional Encoding (LapPE) ---
def compute_pe(edge_index, num_nodes, k=8):
    print(f"Computing Laplacian PE (k={k})...")
    # 转换为 scipy 稀疏矩阵
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    # 计算归一化拉普拉斯矩阵: L = I - D^(-1/2) A D^(-1/2)
    # 这里的 adj 已经是 A
    deg = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    L = sp.eye(num_nodes) - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # 计算最小的 k 个特征值和特征向量 (排除最小的那个，通常是0)
    # 'SM' = Smallest Magnitude
    try:
        eig_vals, eig_vecs = eigsh(L, k=k + 1, which="SM")
        # 取第 1 到 k+1 个 (跳过第0个常数向量)
        pe = torch.from_numpy(eig_vecs[:, 1 : k + 1]).float()
    except:
        print(
            "Warning: Eigendecomposition failed (graph might be too small or disconnected). Using zeros."
        )
        pe = torch.zeros((num_nodes, k))

    return pe


# 计算 PE 并拼接到 x
pe = compute_pe(data.edge_index, data.x.size(0), k=CONFIG["PE_DIM"])
data.x = torch.cat((data.x, pe.to(data.x.device)), dim=1)
print(f"New feature dim: {data.x.size(1)}")

data = data.to(device)

with open("./data/k_sets.pkl", "rb") as handle:
    k_sets = pickle.load(handle)

# 处理自环
pb, _ = remove_self_loops(data.edge_index)
pb, _ = add_self_loops(pb)
E = data.edge_index
num_nodes = data.x.size(0)
num_edges = pb.size(1)


# ==========================================
# 3. [新增] Focal Loss 定义
# ==========================================
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, targets: labels
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # pt = p if y=1 else 1-p

        # 动态调整 alpha
        # 如果 target=1, alpha_t = alpha
        # 如果 target=0, alpha_t = 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ==========================================
# 4. 模型定义 (GT + LayerNorm + Residual)
# ==========================================
class GraphTransformerNet(torch.nn.Module):
    def __init__(self):
        super(GraphTransformerNet, self).__init__()

        in_channels = data.x.size(1)
        hidden_1 = CONFIG["HIDDEN_1"]
        hidden_2 = CONFIG["HIDDEN_2"]
        heads = CONFIG["HEADS"]

        # --- Layer 1 ---
        self.conv1 = TransformerConv(
            in_channels,
            hidden_1,
            heads=heads,
            dropout=CONFIG["DROPOUT_ATT"],
            beta=CONFIG["BETA"],
            concat=True,
        )
        # [修改] 使用 LayerNorm (Transformer 标配)
        self.ln1 = LayerNorm(hidden_1 * heads)
        # 用于残差连接的线性映射 (如果维度不匹配)
        self.lin_skip1 = Linear(in_channels, hidden_1 * heads)

        # --- Layer 2 ---
        self.conv2 = TransformerConv(
            hidden_1 * heads,
            hidden_2,
            heads=heads,
            dropout=CONFIG["DROPOUT_ATT"],
            beta=CONFIG["BETA"],
            concat=True,
        )
        self.ln2 = LayerNorm(hidden_2 * heads)
        self.lin_skip2 = Linear(hidden_1 * heads, hidden_2 * heads)

        # --- Layer 3 (Output) ---
        self.conv3 = TransformerConv(
            hidden_2 * heads,
            1,
            heads=1,
            dropout=CONFIG["DROPOUT_ATT"],
            beta=CONFIG["BETA"],
            concat=False,
        )

        # 可学习参数 (用于多任务 Loss 权重)
        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self):
        edge_index, _ = dropout_edge(
            data.edge_index,
            p=CONFIG["DROPOUT_EDGE"],
            force_undirected=True,
            training=self.training,
        )

        x_in = F.dropout(data.x, p=CONFIG["DROPOUT_FEAT"], training=self.training)

        # --- Block 1 ---
        x = self.conv1(x_in, edge_index)
        x = self.ln1(x)
        x = F.elu(x)
        x = F.dropout(x, p=CONFIG["DROPOUT_FEAT"], training=self.training)
        # [新增] 残差连接
        x = x + self.lin_skip1(x_in)

        # --- Block 2 ---
        x_prev = x
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.elu(x)
        # [新增] 残差连接
        x = x + self.lin_skip2(x_prev)

        z = x  # 用于重构 Loss 的 embedding

        # --- Reconstruction Loss ---
        pos_loss = -torch.log(
            torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15
        ).mean()

        neg_edge_index = negative_sampling(pb, num_nodes, num_edges)
        neg_loss = -torch.log(
            1
            - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))
            + 1e-15
        ).mean()

        r_loss = pos_loss + neg_loss

        # --- Block 3 ---
        x_out = F.dropout(x, p=CONFIG["DROPOUT_FEAT"], training=self.training)
        x_out = self.conv3(x_out, edge_index)

        return x_out, r_loss, self.c1, self.c2


# ==========================================
# 5. 训练与测试
# ==========================================
# 初始化 Focal Loss
criterion_cls = FocalLoss(alpha=CONFIG["FOCAL_ALPHA"], gamma=CONFIG["FOCAL_GAMMA"])


def train(mask, model, optimizer):
    model.train()
    optimizer.zero_grad()
    pred, rl, c1, c2 = model()

    # [修改] 使用 Focal Loss
    main_loss = criterion_cls(pred[mask], Y[mask])

    # 联合 Loss
    loss = main_loss / (c1 * c1) + rl / (c2 * c2) + 2 * torch.log(c2 * c1)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(mask, model):
    model.eval()
    x, _, _, _ = model()
    pred = torch.sigmoid(x[mask]).cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()

    if np.isnan(pred).any():
        pred = np.nan_to_num(pred)

    precision, recall, _ = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    roc_auc = metrics.roc_auc_score(Yn, pred)
    return roc_auc, area


# ==========================================
# 主流程
# ==========================================
mlflow.set_experiment(CONFIG["EXP_NAME"])

with mlflow.start_run(run_name=CONFIG["NOTE"]):
    mlflow.log_params(CONFIG)

    target_run = CONFIG["TARGET_RUN"]
    target_fold = CONFIG["TARGET_FOLD"]

    _, _, tr_mask, te_mask = k_sets[target_run][target_fold]
    tr_mask = torch.as_tensor(tr_mask).to(device)
    te_mask = torch.as_tensor(te_mask).to(device)

    model = GraphTransformerNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=CONFIG["SCHEDULER_PATIENCE"]
    )

    time_start = time.time()
    loop = tqdm(range(1, CONFIG["EPOCH"] + 1), desc="Training GT+", dynamic_ncols=True)

    best_auc = 0
    best_epoch = 0

    for epoch in loop:
        # train 函数不再需要 pos_weight，因为 Focal Loss 处理了
        loss_val = train(tr_mask, model, optimizer)
        mlflow.log_metric("train_loss", loss_val, step=epoch)

        if epoch % 10 == 0:
            train_auc, train_aupr = test(tr_mask, model)
            val_auc, val_aupr = test(te_mask, model)

            mlflow.log_metrics(
                {"train_auc": train_auc, "val_auc": val_auc, "val_aupr": val_aupr},
                step=epoch,
            )

            scheduler.step(val_auc)

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                # torch.save(model.state_dict(), "best_model.pth")

            loop.set_postfix(
                loss=f"{loss_val:.4f}",
                tr_auc=f"{train_auc:.4f}",
                val_auc=f"{val_auc:.4f}",
                best=f"{best_auc:.4f}",
            )
            model.train()

    final_auc, final_aupr = test(te_mask, model)
    total_duration = time.time() - time_start

    mlflow.log_metrics(
        {
            "final_test_auc": final_auc,
            "final_test_aupr": final_aupr,
            "best_val_auc": best_auc,
            "duration_seconds": total_duration,
        }
    )

    print("\n" + "=" * 30)
    print(f"Final Test AUC : {final_auc:.5f}")
    print(f"Best Test AUC  : {best_auc:.5f} (Epoch {best_epoch})")
    print("=" * 30)
