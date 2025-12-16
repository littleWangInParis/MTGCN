import torch
from torch_geometric.data import Data

# 1. 加载旧版本的 Data 对象
legacy_data = torch.load("./data/CPDB_data.pkl", map_location="cpu", weights_only=False)

# 2. 创建一个新的、兼容当前 PyG 版本的空 Data 对象
data = Data()

# 3. 将旧对象的所有属性转移到新对象中
for key, item in legacy_data.__dict__.items():
    if item is not None:
        data[key] = item

torch.save(data, "./data_v2.pt")

d = torch.load("./data_v2.pt", weights_only=False)
