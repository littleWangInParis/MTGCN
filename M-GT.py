import numpy as np
import time
import pickle
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Linear

# --- [修改] 导入 TransformerConv ---
from torch_geometric.nn import TransformerConv, BatchNorm
from torch_geometric.utils import (
    dropout_edge,
    negative_sampling,
    remove_self_loops,
    add_self_loops,
)

from sklearn import metrics
import mlflow

# ==========================================
# 1. 参数配置 (针对 Graph Transformer 调整)
# ==========================================
CONFIG = {
    "EPOCH": 2000,
    "LR": 0.0005,  # [建议] Transformer 通常需要比 GAT 更小的学习率，或者配合 Warmup
    "WEIGHT_DECAY": 1e-4,
    "DROPOUT_EDGE": 0.1,
    "DROPOUT_FEAT": 0.2,
    "DROPOUT_ATT": 0.3,  # Transformer 内部的 Dropout
    "HIDDEN_1": 64,
    "HIDDEN_2": 32,
    "HEADS": 4,  # 多头注意力数量
    "BETA": True,  # [新增] 是否开启门控机制，Graph Transformer 的关键特性
    "LINEAR_HIDDEN": 100,
    "SEED": 42,
    "TARGET_RUN": 0,
    "TARGET_FOLD": 0,
    "SCHEDULER_PATIENCE": 10,
    "NOTE": "Graph Transformer with Gating and Aux Loss",
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


device = torch.device("cpu")  # 如果有GPU请改为 "cuda"

print("Loading data...")
# 假设文件路径保持不变
try:
    data = torch.load("./CPDB_data_v2.pt", weights_only=False)
except FileNotFoundError:
    # 为了防止报错，如果没有文件，这里仅作占位，实际运行时请确保文件存在
    print("Warning: Data file not found. Please ensure './CPDB_data_v2.pt' exists.")
    exit()

y_all = np.logical_or(data.y, data.y_te)
Y = torch.tensor(y_all).type(torch.float32).to(device)

if not isinstance(data.mask, torch.Tensor):
    mask_tensor = torch.tensor(data.mask)
    mask_te_tensor = torch.tensor(data.mask_te)
else:
    mask_tensor = data.mask
    mask_te_tensor = data.mask_te

mask_all = (mask_tensor | mask_te_tensor).to(device)

data.x = data.x[:, :48]

try:
    datas = torch.load("./data/str_fearures.pkl", map_location="cpu")
    data.x = torch.cat((data.x, datas), 1)
except FileNotFoundError:
    print("Warning: Feature file not found.")
    exit()

data.x = data.x.float()
data = data.to(device)

try:
    with open("./data/k_sets.pkl", "rb") as handle:
        k_sets = pickle.load(handle)
except FileNotFoundError:
    print("Warning: k_sets file not found.")
    exit()

# 处理自环
pb, _ = remove_self_loops(data.edge_index)
pb, _ = add_self_loops(pb)
E = data.edge_index

num_nodes = data.x.size(0)
num_edges = pb.size(1)


# ==========================================
# 4. 模型定义 (Graph Transformer)
# ==========================================
class GraphTransformerNet(torch.nn.Module):
    def __init__(self):
        super(GraphTransformerNet, self).__init__()

        in_channels = data.x.size(1)

        # --- Layer 1: TransformerConv ---
        # TransformerConv 参数: in, out, heads, dropout, beta(门控)
        self.conv1 = TransformerConv(
            in_channels,
            CONFIG["HIDDEN_1"],
            heads=CONFIG["HEADS"],
            dropout=CONFIG["DROPOUT_ATT"],
            beta=CONFIG["BETA"],  # 开启门控
            concat=True,
        )

        # Batch Norm 1
        self.bn1 = BatchNorm(CONFIG["HIDDEN_1"] * CONFIG["HEADS"])

        # --- Layer 2: TransformerConv ---
        self.conv2 = TransformerConv(
            CONFIG["HIDDEN_1"] * CONFIG["HEADS"],
            CONFIG["HIDDEN_2"],
            heads=CONFIG["HEADS"],
            dropout=CONFIG["DROPOUT_ATT"],
            beta=CONFIG["BETA"],
            concat=True,
        )

        # Batch Norm 2
        self.bn2 = BatchNorm(CONFIG["HIDDEN_2"] * CONFIG["HEADS"])

        # --- Layer 3 (Output): TransformerConv ---
        # 最后一层通常 heads=1 或者 concat=False 并求平均
        # 这里为了保持输出维度为 1，我们设 heads=1
        self.conv3 = TransformerConv(
            CONFIG["HIDDEN_2"] * CONFIG["HEADS"],
            1,
            heads=1,
            dropout=CONFIG["DROPOUT_ATT"],
            beta=CONFIG["BETA"],
            concat=False,
        )

        # 辅助线性层 (用于重构 Loss)
        self.gat_out_dim = CONFIG["HIDDEN_2"] * CONFIG["HEADS"]
        # 注意：这里不需要改动太多，只要维度对齐即可

        # 可学习参数 (保持不变)
        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self):
        # Edge Dropout
        edge_index, _ = dropout_edge(
            data.edge_index,
            p=CONFIG["DROPOUT_EDGE"],
            force_undirected=True,
            training=self.training,
        )

        x0 = F.dropout(data.x, p=CONFIG["DROPOUT_FEAT"], training=self.training)

        # Layer 1
        x = self.conv1(x0, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=CONFIG["DROPOUT_FEAT"], training=self.training)

        # Layer 2
        x1 = self.conv2(x, edge_index)
        x1 = self.bn2(x1)
        x1 = F.elu(x1)

        x_skip = z = x1

        # --- Loss Calculation (Auxiliary) ---
        # 这里的逻辑保持完全一致
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

        # Layer 3
        x_out = F.dropout(x_skip, p=CONFIG["DROPOUT_FEAT"], training=self.training)
        x_out = self.conv3(x_out, edge_index)

        return x_out, r_loss, self.c1, self.c2


# ==========================================
# 5. 训练与测试函数 (保持不变)
# ==========================================
def train(mask, model, optimizer, pos_weight):
    model.train()
    optimizer.zero_grad()
    pred, rl, c1, c2 = model()

    main_loss = F.binary_cross_entropy_with_logits(
        pred[mask], Y[mask], pos_weight=pos_weight
    )

    loss = main_loss / (c1 * c1) + rl / (c2 * c2) + 2 * torch.log(c2 * c1)
    loss.backward()

    # 梯度裁剪
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
        print("Warning: NaN detected in predictions")
        pred = np.nan_to_num(pred)

    precision, recall, _ = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    roc_auc = metrics.roc_auc_score(Yn, pred)

    return roc_auc, area


mlflow.set_experiment(CONFIG["EXP_NAME"])

with mlflow.start_run(run_name=CONFIG["NOTE"]):
    mlflow.log_params(CONFIG)

    target_run = CONFIG["TARGET_RUN"]
    target_fold = CONFIG["TARGET_FOLD"]

    _, _, tr_mask, te_mask = k_sets[target_run][target_fold]
    tr_mask = torch.as_tensor(tr_mask).to(device)
    te_mask = torch.as_tensor(te_mask).to(device)

    train_labels = Y[tr_mask]
    num_pos = train_labels.sum()
    num_neg = (train_labels == 0).sum()
    weight_value = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([weight_value]).to(device)
    print(f"Pos Weight: {weight_value:.4f}")

    # [修改] 实例化 Graph Transformer
    model = GraphTransformerNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=CONFIG["SCHEDULER_PATIENCE"],
    )

    time_start = time.time()
    loop = tqdm(range(1, CONFIG["EPOCH"] + 1), desc="Training GT", dynamic_ncols=True)

    best_auc = 0
    best_epoch = 0

    for epoch in loop:
        loss_val = train(tr_mask, model, optimizer, pos_weight)
        mlflow.log_metric("train_loss", loss_val, step=epoch)

        if epoch % 50 == 0:
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
                # torch.save(model.state_dict(), "best_gt_model.pth")

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
