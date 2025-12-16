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

import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import (
    dropout_edge,
    negative_sampling,
    remove_self_loops,
    add_self_loops,
)

from sklearn import metrics

import mlflow
import mlflow.pytorch

CONFIG = {
    "EPOCH": 2500,
    "LR": 0.001,
    "DROPOUT_EDGE": 0.5,
    "DROPOUT_FEAT": 0.5,
    "HIDDEN_1": 300,
    "HIDDEN_2": 100,
    "CHEB_K": 2,
    "LINEAR_HIDDEN": 100,
    "SEED": 42,
    "TARGET_RUN": 0,  # 指定使用 k_sets 中的第几组数据 (0-9)
    "TARGET_FOLD": 0,  # 指定使用第几折 (0-4)
    "NOTE": "Single Run Test with BCE for real",
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


def log_single_result(config, auc, aupr, duration):
    """保留原本的CSV记录功能作为备份"""
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

if not torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("Loading data...")

if not os.path.exists("./CPDB_data_v2.pt"):
    print("Error: Data file './CPDB_data_v2.pt' not found.")
    sys.exit(1)

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

data.x = data.x[:, :48]

if os.path.exists("./data/str_fearures.pkl"):
    datas = torch.load("./data/str_fearures.pkl", map_location="cpu")
    data.x = torch.cat((data.x, datas), 1)
else:
    print("Warning: './data/str_fearures.pkl' not found, skipping feature concat.")

data.x = data.x.float()
data = data.to(device)

if os.path.exists("./data/k_sets.pkl"):
    with open("./data/k_sets.pkl", "rb") as handle:
        k_sets = pickle.load(handle)
else:
    print("Error: './data/k_sets.pkl' not found.")
    sys.exit(1)

pb, _ = remove_self_loops(data.edge_index)
pb, _ = add_self_loops(pb)
E = data.edge_index


# ==========================================
# 4. 模型定义
# ==========================================
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ChebConv(
            64, CONFIG["HIDDEN_1"], K=CONFIG["CHEB_K"], normalization="sym"
        )
        self.conv2 = ChebConv(
            CONFIG["HIDDEN_1"],
            CONFIG["HIDDEN_2"],
            K=CONFIG["CHEB_K"],
            normalization="sym",
        )
        self.conv3 = ChebConv(
            CONFIG["HIDDEN_2"], 1, K=CONFIG["CHEB_K"], normalization="sym"
        )

        self.lin1 = Linear(64, CONFIG["LINEAR_HIDDEN"])
        self.lin2 = Linear(64, CONFIG["LINEAR_HIDDEN"])

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self):
        edge_index, _ = dropout_edge(
            data.edge_index,
            p=CONFIG["DROPOUT_EDGE"],
            force_undirected=True,
            training=self.training,
        )

        x0 = F.dropout(data.x, p=CONFIG["DROPOUT_FEAT"], training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, p=CONFIG["DROPOUT_FEAT"], training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        pos_loss = -torch.log(
            torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15
        ).mean()

        # 注意：这里的 13627 和 504378 是硬编码的节点数/边数，如果换数据可能报错
        neg_edge_index = negative_sampling(pb, 13627, 504378)

        neg_loss = -torch.log(
            1
            - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))
            + 1e-15
        ).mean()

        r_loss = pos_loss + neg_loss

        x = F.dropout(x, p=CONFIG["DROPOUT_FEAT"], training=self.training)
        x = self.conv3(x, edge_index)

        return x, r_loss, self.c1, self.c2


# ==========================================
# 5. 训练与测试函数
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
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(mask, model):
    model.eval()
    x, _, _, _ = model()
    pred = torch.sigmoid(x[mask]).cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()
    precision, recall, _ = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(Yn, pred), area


# ==========================================
# 6. 主程序 (集成 MLflow)
# ==========================================

# --- [MLflow Add] 设置实验名称 ---
mlflow.set_experiment("CPDB_GNN_Experiment")

# --- [MLflow Add] 开启 Run ---
with mlflow.start_run():
    # --- [MLflow Add] 记录所有超参数 ---
    mlflow.log_params(CONFIG)

    # 获取指定的训练集和测试集
    target_run = CONFIG["TARGET_RUN"]
    target_fold = CONFIG["TARGET_FOLD"]
    print(f"Selecting Data: Run {target_run}, Fold {target_fold}")

    _, _, tr_mask, te_mask = k_sets[target_run][target_fold]
    tr_mask = torch.as_tensor(tr_mask).to(device)
    te_mask = torch.as_tensor(te_mask).to(device)

    # ==========================================
    # [新增] 计算 pos_weight
    # ==========================================
    train_labels = Y[tr_mask]
    num_pos = train_labels.sum()
    num_neg = (train_labels == 0).sum()

    # 避免除以0
    if num_pos == 0:
        weight_value = 1.0
    else:
        weight_value = num_neg / num_pos

    # 转为 Tensor 并放入 GPU/CPU
    pos_weight = torch.tensor([weight_value]).to(device)

    print(f"Class Imbalance Calculation: Pos={num_pos}, Neg={num_neg}")
    print(f"Using pos_weight: {weight_value:.4f}")

    # 记录到 MLflow 方便后续分析
    mlflow.log_param("pos_weight", weight_value.item())
    # ==========================================

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])

    time_start = time.time()

    # 进度条
    loop = tqdm(range(1, CONFIG["EPOCH"] + 1), desc="Training", dynamic_ncols=True)

    best_auc = 0
    best_epoch = 0

    for epoch in loop:
        loss_val = train(tr_mask, model, optimizer, pos_weight)

        # --- [MLflow Add] 记录训练 Loss ---
        mlflow.log_metric("train_loss", loss_val, step=epoch)

        # 每 100 个 Epoch 验证一次效果
        if epoch % 100 == 0:
            val_auc, val_aupr = test(te_mask, model)

            # --- [MLflow Add] 记录验证集指标 ---
            mlflow.log_metrics({"val_auc": val_auc, "val_aupr": val_aupr}, step=epoch)

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch

            loop.set_postfix(loss=f"{loss_val:.4f}", val_auc=f"{val_auc:.4f}")

            # 恢复训练模式
            model.train()

    # 最终测试
    final_auc, final_aupr = test(te_mask, model)
    total_duration = time.time() - time_start

    print("\n" + "=" * 30)
    print("SINGLE RUN RESULT")
    print("=" * 30)
    print(f"Final Test AUC : {final_auc:.5f}")
    print(f"Final Test AUPR: {final_aupr:.5f}")
    print(f"Best AUC seen  : {best_auc:.5f} (at epoch {best_epoch})")
    print(f"Time Elapsed   : {total_duration:.2f}s")
    print("=" * 30)

    # --- [MLflow Add] 记录最终结果 ---
    mlflow.log_metrics(
        {
            "final_test_auc": final_auc,
            "final_test_aupr": final_aupr,
            "best_val_auc": best_auc,
            "duration_seconds": total_duration,
        }
    )

    # 这会将模型序列化并存储在 MLflow Artifacts 中
    mlflow.pytorch.log_model(model, "model")
    print("[MLflow] Model saved to artifacts.")

    # 同时也保留原本的 CSV 记录
    log_single_result(CONFIG, final_auc, final_aupr, total_duration)
