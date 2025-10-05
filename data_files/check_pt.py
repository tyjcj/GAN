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
import os
import csv
import argparse
from tqdm import tqdm


def calculate_parameters(pt_file_path):
    """从单个.pt文件中计算电路参数"""
    try:
        # 加载PyG Data对象
        data = torch.load(pt_file_path, weights_only=False)
        max_depth = data.node_depth.max().item()
        depth_zero_ratio = (data.node_depth == 0).sum().item() / data.x.shape[0]

        print(f"最大深度: {max_depth}")
        print(f"深度为0的节点比例: {depth_zero_ratio:.2%}")

        if max_depth == 0:
            print("❌ 警告：所有节点深度为0，数据可能有问题！")
        elif depth_zero_ratio > 0.1:  # 超过10%节点深度为0
            print("⚠️  深度为0的节点过多，可能存在问题")
        else:
            print("✅ 深度分布正常")
        # 计算各项参数
        # Primary Inputs (PI)：节点类型为0的节点数量
        pi_count = int((data.x[:, 0] == 0).sum().item())

        # Primary Outputs (PO)：节点类型为1的节点数量
        po_count = int((data.x[:, 0] == 1).sum().item())

        # N
        N_nodes = int((data.x[:, 0] == 2).sum().item())

        # Edges (E)：总边数（edge_index的列数）
        total_edges = data.edge_index.shape[1] if data.edge_index.numel() > 0 else 0

        #Inverted edges (I)：反相边数量（edge_attr为1的边）
        inverted_edges = int((data.edge_attr == 1).sum().item()) if data.edge_attr.numel() > 0 else 0

        # Netlist Depth (D)：网表深度（最大节点深度）
        netlist_depth = int(data.node_depth.max().item()) if data.node_depth.numel() > 0 else 0

        return {
            'filename': os.path.basename(pt_file_path),
            'PI': pi_count,
            'PO': po_count,
            'N': N_nodes,
            'I':inverted_edges,
            'E': total_edges,
            'D': netlist_depth,
            'status': 'success'
        }
    except Exception as e:
        return {
            'filename': os.path.basename(pt_file_path),
            'PI': None,
            'PO': None,
            'N': None,
            'E': None,
            'I': None,
            'D': None,
            'status': f'error: {str(e)}'
        }


def process_pt_directory(pt_dir, output_csv):
    """处理目录下所有.pt文件并将结果保存到CSV"""
    # 获取所有.pt文件
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
    for pt_file in pt_files:
        # 构建完整文件路径
        file_path = os.path.join(pt_dir, pt_file)

        # 加载模型
        model = torch.load(file_path, weights_only=False)

        # 提取文件名（不含扩展名）作为模型名称
        model_name = os.path.splitext(pt_file)[0]

        # 打印模型结构
        print(f"{model_name}: {model}")
        print("-" * 50)  # 分隔线，方便阅读
    if not pt_files:
        print(f"在目录 {pt_dir} 中未找到任何.pt文件")
        return

    # 创建输出CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'PI', 'PO', 'N', 'E', 'I', 'D', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # 处理每个PT文件
        print(f"发现 {len(pt_files)} 个PT文件，开始处理...")
        for pt_file in tqdm(pt_files, desc="处理进度"):
            file_path = os.path.join(pt_dir, pt_file)
            params = calculate_parameters(file_path)
            writer.writerow(params)

    print(f"处理完成，结果已保存到 {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算PT文件中的电路参数并保存到CSV')
    parser.add_argument('--pt-dir', required=True, help='PT文件所在目录')
    parser.add_argument('--output-csv', default='pt_parameters.csv', help='输出CSV文件路径')
    args = parser.parse_args()

    process_pt_directory(args.pt_dir, args.output_csv)
"""
python data_files/check_pt.py --pt data_files/datasets/ISCAS85/graph --output data_files/datasets/ISCAS85/pt_parameters.csv
"""