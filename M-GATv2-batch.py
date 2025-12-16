import sys
import numpy as np
import pandas as pd
import time
import pickle
import random
import os
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

# --- 导入 GATv2Conv ---
from torch_geometric.nn import GATv2Conv, BatchNorm
from torch_geometric.data import Data
from torch_geometric.utils import (
    dropout_edge,
    negative_sampling,
    remove_self_loops,
    add_self_loops,
)

from sklearn import metrics
import mlflow
import mlflow.pytorch

# ==========================================
# 1. 参数配置 (优化版)
# ==========================================
CONFIG = {
    "EPOCH": 2000,  # GAT 收敛可能慢一点，或者早停
    "LR": 0.001,  # [修改] GAT 建议使用较小的学习率 (0.001 或 0.0005)
    "WEIGHT_DECAY": 5e-4,  # [新增] 强正则化防止过拟合
    "DROPOUT_EDGE": 0.2,  # [微调] 边 Dropout 不宜过大，否则破坏结构
    "DROPOUT_FEAT": 0.5,  # 特征 Dropout
    "DROPOUT_ATT": 0.5,  # [关键] 注意力系数的 Dropout
    "HIDDEN_1": 64,  # 每个头的维度
    "HIDDEN_2": 32,
    "HEADS": 4,  # 多头数量
    "LINEAR_HIDDEN": 100,
    "SEED": 42,
    "TARGET_RUN": 0,
    "TARGET_FOLD": 0,
    "NOTE": "GATv2 Optimization with BatchNorm and Scheduler",
}


# ==========================================
# 2. 工具函数
# ==========================================
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to: {seed}")


def log_single_result(config, auc, aupr, duration):
    filename = "tuning_single_run_log.csv"
    record = config.copy()
    record["Test_AUC"] = auc
    record["Test_AUPR"] = aupr
    record["Duration_Sec"] = round(duration, 2)
    record["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame([record])
    if not os.path.exists(filename):
        df_new.to_csv(filename, index=False)
    else:
        df_new.to_csv(filename, mode="a", header=False, index=False)
    print(f"\n[Result Saved] Results appended to {filename}")


set_seed(CONFIG["SEED"])

device = torch.device("cpu")

data = torch.load("./CPDB_data_v2.pt", weights_only=False)

y_all = np.logical_or(data.y, data.y_te)
Y = torch.tensor(y_all).type(torch.float32).to(device)

if not isinstance(data.mask, torch.Tensor):
    mask_tensor = torch.tensor(data.mask)
    mask_te_tensor = torch.tensor(data.mask_te)
else:
    mask_tensor = data.mask
    mask_te_tensor = data.mask_te

mask_all = (mask_tensor | mask_te_tensor).to(device)

# 特征处理
data.x = data.x[:, :48]

# 尝试加载外部特征
if os.path.exists("./data/str_fearures.pkl"):
    datas = torch.load("./data/str_fearures.pkl", map_location="cpu")
    data.x = torch.cat((data.x, datas), 1)
else:
    print("Warning: './data/str_fearures.pkl' not found, skipping feature concat.")

data.x = data.x.float()
data = data.to(device)

# 加载 K-Fold
if os.path.exists("./data/k_sets.pkl"):
    with open("./data/k_sets.pkl", "rb") as handle:
        k_sets = pickle.load(handle)
else:
    sys.exit(1)

# 处理自环
pb, _ = remove_self_loops(data.edge_index)
pb, _ = add_self_loops(pb)
E = data.edge_index

num_nodes = data.x.size(0)
num_edges = pb.size(1)


# ==========================================
# 4. 模型定义 (GATv2 + BatchNorm)
# ==========================================
class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()

        in_channels = data.x.size(1)

        # --- Layer 1 ---
        self.conv1 = GATv2Conv(
            in_channels,
            CONFIG["HIDDEN_1"],
            heads=CONFIG["HEADS"],
            dropout=CONFIG["DROPOUT_ATT"],
            concat=True,
        )
        # [新增] Batch Norm 1
        # 输入维度是 HIDDEN_1 * HEADS
        self.bn1 = BatchNorm(CONFIG["HIDDEN_1"] * CONFIG["HEADS"])

        # --- Layer 2 ---
        self.conv2 = GATv2Conv(
            CONFIG["HIDDEN_1"] * CONFIG["HEADS"],
            CONFIG["HIDDEN_2"],
            heads=CONFIG["HEADS"],
            dropout=CONFIG["DROPOUT_ATT"],
            concat=True,
        )
        # [新增] Batch Norm 2
        self.bn2 = BatchNorm(CONFIG["HIDDEN_2"] * CONFIG["HEADS"])

        # --- Layer 3 (Output) ---
        self.conv3 = GATv2Conv(
            CONFIG["HIDDEN_2"] * CONFIG["HEADS"],
            1,
            heads=1,
            dropout=CONFIG["DROPOUT_ATT"],
            concat=False,
        )

        # 辅助线性层
        self.gat_out_dim = CONFIG["HIDDEN_2"] * CONFIG["HEADS"]
        self.lin1 = Linear(in_channels, self.gat_out_dim)
        self.lin2 = Linear(in_channels, self.gat_out_dim)

        # 可学习参数
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
        x = self.bn1(x)  # [新增] BN
        x = F.elu(x)
        x = F.dropout(x, p=CONFIG["DROPOUT_FEAT"], training=self.training)

        # Layer 2
        x1 = self.conv2(x, edge_index)
        x1 = self.bn2(x1)  # [新增] BN
        x1 = F.elu(x1)

        # Residual Connections
        # 注意：lin1(x0) 只是简单的线性映射，没有图结构信息。
        # 如果 GAT 效果不好，可能是因为 x1 (图特征) 被 lin1(x0) (原始特征) 干扰了，或者反之。
        # 这里保留你的逻辑，但建议如果效果还不好，可以尝试去掉 lin1/lin2，使用纯 GAT 结构。
        x_skip = x1 + F.elu(self.lin1(x0))
        z = x1 + F.elu(self.lin2(x0))

        # --- Loss Calculation (Auxiliary) ---
        # 这里的重构 Loss 计算量较大，如果显存不够可以适当优化
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
# 5. 训练与测试函数
# ==========================================
def train(mask, model, optimizer, pos_weight):
    model.train()
    optimizer.zero_grad()
    pred, rl, c1, c2 = model()

    # 主分类 Loss
    main_loss = F.binary_cross_entropy_with_logits(
        pred[mask], Y[mask], pos_weight=pos_weight
    )

    # 联合 Loss
    loss = main_loss / (c1 * c1) + rl / (c2 * c2) + 2 * torch.log(c2 * c1)
    loss.backward()

    # [新增] 梯度裁剪：防止 GAT 梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(mask, model):
    model.eval()
    x, _, _, _ = model()
    pred = torch.sigmoid(x[mask]).cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()

    # 防止 NaN
    if np.isnan(pred).any():
        print("Warning: NaN detected in predictions")
        pred = np.nan_to_num(pred)

    precision, recall, _ = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    try:
        roc_auc = metrics.roc_auc_score(Yn, pred)
    except ValueError:
        roc_auc = 0.0

    return roc_auc, area


# ==========================================
# 6. 主程序
# ==========================================
mlflow.set_experiment("CPDB_GAT_Optimized")

with mlflow.start_run():
    mlflow.log_params(CONFIG)

    target_run = CONFIG["TARGET_RUN"]
    target_fold = CONFIG["TARGET_FOLD"]

    _, _, tr_mask, te_mask = k_sets[target_run][target_fold]
    tr_mask = torch.as_tensor(tr_mask).to(device)
    te_mask = torch.as_tensor(te_mask).to(device)

    # 计算 pos_weight
    train_labels = Y[tr_mask]
    num_pos = train_labels.sum()
    num_neg = (train_labels == 0).sum()
    weight_value = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([weight_value]).to(device)
    print(f"Pos Weight: {weight_value:.4f}")

    model = GATNet().to(device)

    # [修改] 加入 Weight Decay
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"]
    )

    # [新增] 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=100
    )

    time_start = time.time()
    loop = tqdm(range(1, CONFIG["EPOCH"] + 1), desc="Training GAT", dynamic_ncols=True)

    best_auc = 0
    best_epoch = 0

    for epoch in loop:
        loss_val = train(tr_mask, model, optimizer, pos_weight)
        mlflow.log_metric("train_loss", loss_val, step=epoch)

        # 每 50 个 epoch 验证一次
        if epoch % 50 == 0:
            val_auc, val_aupr = test(te_mask, model)
            mlflow.log_metrics({"val_auc": val_auc, "val_aupr": val_aupr}, step=epoch)

            # 更新 Scheduler (根据 AUC 调整学习率)
            scheduler.step(val_auc)

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                # 保存最佳模型
                # torch.save(model.state_dict(), "best_gat_model.pth")

            loop.set_postfix(
                loss=f"{loss_val:.4f}", val_auc=f"{val_auc:.4f}", best=f"{best_auc:.4f}"
            )
            model.train()

    final_auc, final_aupr = test(te_mask, model)
    total_duration = time.time() - time_start

    print("\n" + "=" * 30)
    print(f"Final Test AUC : {final_auc:.5f}")
    print(f"Best Test AUC  : {best_auc:.5f} (Epoch {best_epoch})")
    print("=" * 30)

    mlflow.log_metrics(
        {
            "final_test_auc": final_auc,
            "final_test_aupr": final_aupr,
            "best_val_auc": best_auc,
            "duration_seconds": total_duration,
        }
    )

    log_single_result(CONFIG, final_auc, final_aupr, total_duration)
