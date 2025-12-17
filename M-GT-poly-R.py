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

from sklearn import metrics  # 包含 accuracy_score, recall_score 等
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
    "PE_DIM": 12,
    "FOCAL_GAMMA": 2.0,
    "FOCAL_ALPHA": 0.7,
    "SEED": 42,
    "TARGET_RUN": 0,
    "TARGET_FOLD": 0,
    "SCHEDULER_PATIENCE": 5,
    "PATIENCE": 15,
    "NOTE": "GT + LapPE + Poly Loss + LayerNorm",
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


set_seed(CONFIG["SEED"])
device = torch.device("cpu")  # 如有 GPU 改为 "cuda"

# ==========================================
# 2. 数据加载与处理
# ==========================================
print("Loading data...")
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


# --- Laplacian Positional Encoding (LapPE) ---
def compute_pe(edge_index, num_nodes, k=8):
    print(f"Computing Laplacian PE (k={k})...")
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    deg = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(num_nodes) - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    try:
        _, eig_vecs = eigsh(L, k=k + 1, which="SM")
        pe = torch.from_numpy(eig_vecs[:, 1 : k + 1]).float()
    except:
        print("Warning: Eigendecomposition failed. Using zeros.")
        pe = torch.zeros((num_nodes, k))
    return pe


pe = compute_pe(data.edge_index, data.x.size(0), k=CONFIG["PE_DIM"])
data.x = torch.cat((data.x, pe.to(data.x.device)), dim=1)
print(f"New feature dim: {data.x.size(1)}")

data = data.to(device)

with open("./data/k_sets.pkl", "rb") as handle:
    k_sets = pickle.load(handle)

pb, _ = remove_self_loops(data.edge_index)
pb, _ = add_self_loops(pb)
E = data.edge_index
num_nodes = data.x.size(0)
num_edges = pb.size(1)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            print(
                f"Validation score improved ({self.best_score:.6f} --> {score:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)


# ==========================================
# 3. Poly Loss
# ==========================================
class PolyLoss(torch.nn.Module):
    def __init__(self, epsilon=1.0, alpha=0.25, gamma=2.0, reduction="mean"):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, targets: 0 or 1
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)

        # Focal Loss term
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_term = alpha_t * (1 - pt) ** self.gamma * bce_loss

        # Poly1 term: epsilon * (1-pt)^gamma
        # 这一项能让模型在处理易分样本时更加灵活
        poly_term = self.epsilon * (1 - pt) ** (self.gamma + 1)

        loss = focal_term + poly_term

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ==========================================
# 4. 模型定义
# ==========================================
class GraphTransformerNet(torch.nn.Module):
    def __init__(self):
        super(GraphTransformerNet, self).__init__()
        in_channels = data.x.size(1)
        hidden_1 = CONFIG["HIDDEN_1"]
        hidden_2 = CONFIG["HIDDEN_2"]
        heads = CONFIG["HEADS"]

        self.conv1 = TransformerConv(
            in_channels,
            hidden_1,
            heads=heads,
            dropout=CONFIG["DROPOUT_ATT"],
            beta=CONFIG["BETA"],
            concat=True,
        )
        self.ln1 = LayerNorm(hidden_1 * heads)
        self.lin_skip1 = Linear(in_channels, hidden_1 * heads)

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

        self.conv3 = TransformerConv(
            hidden_2 * heads,
            1,
            heads=1,
            dropout=CONFIG["DROPOUT_ATT"],
            beta=CONFIG["BETA"],
            concat=False,
        )

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

        x = self.conv1(x_in, edge_index)
        x = self.ln1(x)
        x = F.elu(x)
        x = F.dropout(x, p=CONFIG["DROPOUT_FEAT"], training=self.training)
        x = x + self.lin_skip1(x_in)

        x_prev = x
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.elu(x)
        x = x + self.lin_skip2(x_prev)

        z = x
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

        x_out = F.dropout(x, p=CONFIG["DROPOUT_FEAT"], training=self.training)
        x_out = self.conv3(x_out, edge_index)

        return x_out, r_loss, self.c1, self.c2


# 定义 KL 散度计算函数
def compute_kl_loss(p, q, pad_mask=None):
    # 1. 将 logits 转换为概率
    p_prob = torch.sigmoid(p)
    q_prob = torch.sigmoid(q)

    # 2. 加上微小值防止 log(0)
    eps = 1e-8

    # 3. 计算 KL(P || Q)
    # 公式: p * log(p/q) + (1-p) * log((1-p)/(1-q))
    kl_p_q = p_prob * torch.log(p_prob / (q_prob + eps) + eps) + (
        1 - p_prob
    ) * torch.log((1 - p_prob) / (1 - q_prob + eps) + eps)

    # 4. 计算 KL(Q || P)
    kl_q_p = q_prob * torch.log(q_prob / (p_prob + eps) + eps) + (
        1 - q_prob
    ) * torch.log((1 - q_prob) / (1 - p_prob + eps) + eps)

    # 5. R-Drop Loss = (KL(P||Q) + KL(Q||P)) / 2
    # 使用 mean() 对 batch 内所有样本求平均
    loss = (kl_p_q + kl_q_p).mean() / 2

    return loss


# 实例化 PolyLoss (替代 Focal Loss)
criterion_cls = PolyLoss(
    epsilon=1.0, alpha=CONFIG["FOCAL_ALPHA"], gamma=CONFIG["FOCAL_GAMMA"]
)


def train(mask, model, optimizer):
    model.train()
    optimizer.zero_grad()

    # --- R-Drop: 两次前向传播 ---
    # 第一次前向
    pred1, rl1, c1, c2 = model()
    # 第二次前向 (Dropout mask 会不同)
    pred2, rl2, _, _ = model()

    # --- 1. 主任务分类损失 (PolyLoss) ---
    loss_cls1 = criterion_cls(pred1[mask], Y[mask])
    loss_cls2 = criterion_cls(pred2[mask], Y[mask])
    main_loss = 0.5 * (loss_cls1 + loss_cls2)

    # --- 2. 辅助任务重构损失 ---
    rl_loss = 0.5 * (rl1 + rl2)

    # --- 3. R-Drop 一致性损失 (修正后的双向 KL) ---
    # 计算 mask 区域的一致性（也可以尝试对全图计算，利用无标签数据）
    kl_loss = compute_kl_loss(pred1[mask], pred2[mask])

    # --- 4. 损失融合 ---
    # R-Drop 权重系数，通常取 1.0 到 5.0 之间
    rdrop_alpha = 1.0

    # 结合 Kendall 不确定性加权 + R-Drop
    loss = (
        main_loss / (c1 * c1) + rl_loss / (c2 * c2) + 2 * torch.log(c2 * c1)
    ) + rdrop_alpha * kl_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(mask, model, threshold=0.5):
    """
    计算多种指标。
    threshold: 将概率转换为类别的阈值，默认为 0.5
    """
    model.eval()
    x, _, _, _ = model()
    # 得到预测概率
    pred_prob = torch.sigmoid(x[mask]).cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()

    if np.isnan(pred_prob).any():
        pred_prob = np.nan_to_num(pred_prob)

    # --- 1. 基于概率的指标 (AUC, AUPR) ---
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(Yn, pred_prob)
    aupr = metrics.auc(recall_curve, precision_curve)
    roc_auc = metrics.roc_auc_score(Yn, pred_prob)

    # --- 2. 基于类别的指标 (Accuracy, Recall, etc.) ---
    # 将概率转换为 0/1 标签
    pred_label = (pred_prob > threshold).astype(int)

    # Accuracy
    acc = metrics.accuracy_score(Yn, pred_label)
    # Recall (Sensitivity)
    recall = metrics.recall_score(Yn, pred_label)
    # Precision
    precision = metrics.precision_score(Yn, pred_label)
    # F1-Score
    f1 = metrics.f1_score(Yn, pred_label)
    # MCC (Matthews Correlation Coefficient) - 适合不平衡数据
    mcc = metrics.matthews_corrcoef(Yn, pred_label)

    # Specificity (特异性) = TN / (TN + FP)
    # 混淆矩阵: ravel() 返回 [tn, fp, fn, tp]
    tn, fp, fn, tp = metrics.confusion_matrix(Yn, pred_label).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 返回一个字典，方便后续调用
    results = {
        "auc": roc_auc,
        "aupr": aupr,
        "acc": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
        "specificity": specificity,
    }

    return results


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
    early_stopping = EarlyStopping(
        patience=CONFIG["PATIENCE"], verbose=True, path="best_model.pt"
    )

    time_start = time.time()
    loop = tqdm(range(1, CONFIG["EPOCH"] + 1), desc="Training", dynamic_ncols=True)

    best_auc = 0
    best_epoch = 0

    for epoch in loop:
        loss_val = train(tr_mask, model, optimizer)
        mlflow.log_metric("train_loss", loss_val, step=epoch)

        if epoch % 10 == 0:
            # 计算训练集指标
            train_res = test(tr_mask, model)
            # 计算验证/测试集指标
            val_res = test(te_mask, model)

            # 记录 AUC 供 Scheduler 和 EarlyStopping 使用
            val_auc = val_res["auc"]

            # --- 记录所有指标到 MLflow ---
            # 为了区分训练集和验证集，给 Key 加上前缀
            log_metrics = {}
            for k, v in train_res.items():
                log_metrics[f"train_{k}"] = v
            for k, v in val_res.items():
                log_metrics[f"val_{k}"] = v

            mlflow.log_metrics(log_metrics, step=epoch)

            scheduler.step(val_auc)

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch

            # 更新进度条显示 (显示最重要的几个)
            loop.set_postfix(
                loss=f"{loss_val:.4f}",
                v_auc=f"{val_auc:.4f}",
                v_f1=f"{val_res['f1']:.4f}",  # 新增显示 F1
                v_rec=f"{val_res['recall']:.4f}",  # 新增显示 Recall
            )

            early_stopping(val_auc, model)
            if early_stopping.early_stop:
                print("\nEarly stopping triggered!")
                break

            model.train()

    print("Loading best model from early stopping...")
    model.load_state_dict(torch.load("best_model.pt"))

    # 最终测试
    final_res = test(te_mask, model)
    total_duration = time.time() - time_start

    # 记录最终结果
    final_metrics = {f"final_test_{k}": v for k, v in final_res.items()}
    final_metrics["best_val_auc"] = best_auc
    final_metrics["duration_seconds"] = total_duration
    mlflow.log_metrics(final_metrics)

    print("\n" + "=" * 40)
    print(f"Final Test Results (Epoch {best_epoch}):")
    print(f"AUC        : {final_res['auc']:.5f}")
    print(f"AUPR       : {final_res['aupr']:.5f}")
    print(f"Accuracy   : {final_res['acc']:.5f}")
    print(f"Recall     : {final_res['recall']:.5f}")
    print(f"Precision  : {final_res['precision']:.5f}")
    print(f"F1-Score   : {final_res['f1']:.5f}")
    print(f"Specificity: {final_res['specificity']:.5f}")
    print(f"MCC        : {final_res['mcc']:.5f}")
    print("=" * 40)
