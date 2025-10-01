import os
import argparse
import torch
import networkx as nx
from torch_geometric.data import Data
from collections import deque


def compute_node_depth(G):
    """拓扑 BFS 计算节点深度"""
    in_degree = {n: G.in_degree(n) for n in G.nodes()}
    depth = {n: 0 for n in G.nodes()}

    q = deque([n for n in G.nodes() if in_degree[n] == 0])
    while q:
        u = q.popleft()
        for v in G.successors(u):
            depth[v] = max(depth[v], depth[u] + 1)
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    return [depth[n] for n in G.nodes()]


def pygDataFromNetworkx(G):
    """从 networkx Graph 转 PyG Data，只保留 x, edge_index, edge_attr, node_depth"""
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G

    # edge_index
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    # x = [node_type, num_inverted_predecessors]
    node_type = []
    num_inv_preds = []
    for _, feat in G.nodes(data=True):
        node_type.append(int(feat.get("node_type", 2)))  # 默认 Internal=2
        num_inv_preds.append(int(feat.get("num_inverted_predecessors", 0)))
    x = torch.tensor(list(zip(node_type, num_inv_preds)), dtype=torch.long)

    # edge_attr
    edge_types = []
    for _, _, feat in G.edges(data=True):
        edge_types.append(int(feat.get("edge_type", 0)))  # 默认 BUFF=0
    edge_attr = torch.tensor(edge_types, dtype=torch.long).view(-1, 1)

    # node_depth
    node_depth = torch.tensor(compute_node_depth(G), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_depth=node_depth
    )


def main():
    parser = argparse.ArgumentParser(description="Convert GraphML to PyG .pt")
    parser.add_argument("--graphml", required=True, help="Input GraphML file")
    parser.add_argument("--out", required=True, help="Output .pt file")
    args = parser.parse_args()

    # 读取 graphml
    G = nx.read_graphml(args.graphml)
    data = pygDataFromNetworkx(G)

    # 保存 pt
    torch.save(data, args.out)
    print(f"[OK] Saved {args.out} with {data}")


if __name__ == "__main__":
    main()
"""
python generate_pt.py \
  --graphml openabcd/graphml/ac97_ctrl_orig.graphml \
  --out openabcd/ac97_ctrl_orig.pt

"""