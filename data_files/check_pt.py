# import torch
# import os
#
# # 指定PT文件所在目录
# pt_dir = "out_gen"
# if not os.path.exists(pt_dir):
#     os.makedirs(pt_dir)
#
# # 获取目录中所有的PT文件
# pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
#
# # 遍历并加载每个PT文件
# for pt_file in pt_files:
#     # 构建完整文件路径
#     file_path = os.path.join(pt_dir, pt_file)
#
#     # 加载模型
#     model = torch.load(file_path, weights_only=False)
#
#     # 提取文件名（不含扩展名）作为模型名称
#     model_name = os.path.splitext(pt_file)[0]
#
#     # 打印模型结构
#     print(f"{model_name}: {model}")
#     print("-" * 50)  # 分隔线，方便阅读

"""
sageformer:
aes_secworks_orig:Data(x=[74990, 3], edge_index=[2, 112681], edge_attr=[112681, 1])
apex1_orig:Data(x=[1915, 3], edge_index=[2, 3445], edge_attr=[3445, 1])
i2c_orig:Data(x=[2195, 3], edge_index=[2, 3187], edge_attr=[3187, 1])
k2_orig:Data(x=[2635, 3 ], edge_index=[2, 4877], edge_attr=[4877, 1])
max_orig:Data(x=[6279, 3], edge_index=[2, 8632], edge_attr=[8632, 1])
tv80_orig:Data(x=[19877, 3], edge_index=[2, 30569], edge_attr=[30569, 1])

original:
max_orig:Data(x=[6279, 2], edge_index=[2, 8632], edge_attr=[8632, 1], node_depth=[6279])
aes_secworks_orig:Data(x=[74990, 2], edge_index=[2, 112681], edge_attr=[112681, 1], node_depth=[74990])
tv:Data(x=[19877, 2], edge_index=[2, 30569], edge_attr=[30569, 1], node_depth=[19877])



now:
aes_secworks_orig:Data(x=(74990, 2), edge_index=(2, 112681), edge_attr=(112681, 1), node_depth=(74990,))
apex1_orig:Data(x=(1912, 2), edge_index=(2, 3445), edge_attr=(3445, 1), node_depth=(1912,))
i2c_orig:Data(x=(2195, 2), edge_index=(2, 3187), edge_attr=(3187, 1), node_depth=(2195,))
k2_orig:Data(x=(2632, 2), edge_index=(2, 4877), edge_attr=(4877, 1), node_depth=(2632,))
max_orig:Data(x=(6031, 2), edge_index=(2, 10534), edge_attr=(10534, 1), node_depth=(6031,))
tv80_orig:Data(x=(19877, 2), edge_index=(2, 30569), edge_attr=(30569, 1), node_depth=(19877,))


"""

import torch


# 1. 加载存储电路特征的.pt文件（替换为你的文件路径）
pt_file_path = "out_gen/sample_0_0_000001.pt"  # 例如用户之前的"aes_secworks_orig.pt"
data = torch.load(pt_file_path, weights_only=False)  # 加载Data对象
print(f"{data}")
# 2. 提取节点特征x（维度[N,2]）
x = data.x  # x.shape = [N, 2]，N为当前电路的节点数
print(f"电路节点特征x的维度：{x.shape}")
print(f"x的部分个节点特征（示例）：\n{x[8000:8010]}")  # 查看前5个节点的特征

# 3. 查看单个节点的特征（以第10个节点为例，索引从0开始）
target_node_idx = 10  # 选择要查看的节点索引
node_feature = x[target_node_idx]

# 4. 解析单个节点的特征含义（对照论文3.1节的定义）
node_type = int(node_feature[0].item())
inverted_pred_count = int(node_feature[1].item())

# 映射节点类型的文字含义
node_type_map = {0: "输入节点（PI）", 1: "输出节点（PO）", 2: "中间节点（AND门）"}
print(f"\n第{target_node_idx}个节点的特征解析：")
print(f"- 节点类型（x[:,0]）：{node_type} → {node_type_map[node_type]}")
print(f"- 反向前驱数（x[:,1]）：{inverted_pred_count} → {inverted_pred_count}个输入反相")

# 5. 统计全电路的节点类型分布（验证特征合理性）
node_type_counts = {0: 0, 1: 0, 2: 0}
for node_type_val in x[:, 0]:
    node_type_counts[int(node_type_val.item())] += 1
print(f"\n全电路节点类型分布（符合AIG结构逻辑）：")
print(f"- 输入节点（PI）：{node_type_counts[0]}个")
print(f"- 输出节点（PO）：{node_type_counts[1]}个")
print(f"- 中间AND门节点：{node_type_counts[2]}个")