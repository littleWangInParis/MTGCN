import sys
import numpy as np
import pandas as pd
import time
import pickle
from tqdm import tqdm  # <--- 新增：导入 tqdm

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

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon (MPS) acceleration")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

EPOCH = 2500

data = torch.load("./CPDB_data_v2.pt", weights_only=False)

# 处理标签 Y
y_all = np.logical_or(data.y, data.y_te)

Y = torch.tensor(y_all).type(torch.float32).to(device)

# 处理 Mask
if not isinstance(data.mask, torch.Tensor):
    mask_tensor = torch.tensor(data.mask)
    mask_te_tensor = torch.tensor(data.mask_te)
else:
    mask_tensor = data.mask
    mask_te_tensor = data.mask_te

mask_all = (mask_tensor | mask_te_tensor).to(device)

# 截取特征
data.x = data.x[:, :48]

# 加载额外特征数据
datas = torch.load("./data/str_fearures.pkl", map_location="cpu")

# 拼接特征
data.x = torch.cat((data.x, datas), 1)

# 移动 Data 对象到设备
data.x = data.x.float()
data = data.to(device)

# 加载 k-fold 数据集
with open("./data/k_sets.pkl", "rb") as handle:
    k_sets = pickle.load(handle)

# 处理边索引
pb, _ = remove_self_loops(data.edge_index)
pb, _ = add_self_loops(pb)
E = data.edge_index


def train(mask):
    model.train()
    optimizer.zero_grad()

    pred, rl, c1, c2 = model()

    # 计算 Loss
    loss = (
        F.binary_cross_entropy_with_logits(pred[mask], Y[mask]) / (c1 * c1)
        + rl / (c2 * c2)
        + 2 * torch.log(c2 * c1)
    )
    loss.backward()
    optimizer.step()

    # 以此返回 loss 供 tqdm 显示（可选）
    return loss.item()


@torch.no_grad()
def test(mask):
    model.eval()
    x, _, _, _ = model()

    pred = torch.sigmoid(x[mask]).cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()

    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)

    return metrics.roc_auc_score(Yn, pred), area


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ChebConv(64, 300, K=2, normalization="sym")
        self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
        self.conv3 = ChebConv(100, 1, K=2, normalization="sym")

        self.lin1 = Linear(64, 100)
        self.lin2 = Linear(64, 100)

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self):
        edge_index, _ = dropout_edge(
            data.edge_index,
            p=0.5,
            force_undirected=True,
            training=self.training,
        )

        x0 = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        pos_loss = -torch.log(
            torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15
        ).mean()

        neg_edge_index = negative_sampling(pb, 13627, 504378)

        neg_loss = -torch.log(
            1
            - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))
            + 1e-15
        ).mean()

        r_loss = pos_loss + neg_loss

        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x, r_loss, self.c1, self.c2


time_start = time.time()
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))

# ---------------------------------------------------------
# 修改部分：交叉验证循环增加 tqdm
# ---------------------------------------------------------
for i in range(10):
    # print(f"Run: {i}") # 可以注释掉，因为 tqdm 会显示信息
    for cv_run in range(5):
        _, _, tr_mask, te_mask = k_sets[i][cv_run]

        tr_mask = torch.as_tensor(tr_mask).to(device)
        te_mask = torch.as_tensor(te_mask).to(device)

        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 使用 tqdm 包装 range
        # desc: 进度条左侧的描述文字
        # leave=False: 完成后清除进度条，避免 50 次循环打印 50 行进度条
        # dynamic_ncols=True: 自动调整宽度
        loop = tqdm(
            range(1, EPOCH),
            desc=f"Run {i + 1}/10 | Fold {cv_run + 1}/5",
            leave=False,
            dynamic_ncols=True,
        )

        for epoch in loop:
            loss_val = train(tr_mask)
            # 可选：在进度条右侧显示当前的 Loss
            if epoch % 100 == 0:
                loop.set_postfix(loss=f"{loss_val:.4f}")

        AUC[i][cv_run], AUPR[i][cv_run] = test(te_mask)

    # 每个 Run 结束后打印一次总时间，作为阶段性反馈
    print(f"Run {i + 1} finished. Time elapsed: {time.time() - time_start:.2f}s")


print("Mean AUC:", AUC.mean())
print("Var AUC:", AUC.var())
print("Mean AUPR:", AUPR.mean())
print("Var AUPR:", AUPR.var())

# ---------------------------------------------------------
# 修改部分：Final Training 增加 tqdm
# ---------------------------------------------------------
print("Starting final training...")
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 这里 leave=True (默认)，保留最终训练的进度条
final_loop = tqdm(range(1, EPOCH), desc="Final Training", dynamic_ncols=True)

for epoch in final_loop:
    loss_val = train(mask_all)
    if epoch % 100 == 0:
        final_loop.set_postfix(loss=f"{loss_val:.4f}")


x, _, _, _ = model()
pred = torch.sigmoid(x[~mask_all]).cpu().detach().numpy()
torch.save(pred, "pred.pkl")
print("Done. Predictions saved to pred.pkl")
