# utilities/pt2graphml.py
import argparse
import os
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data


def pt_to_graphml(pt_path: str, out_path: str, verbose: bool = True) -> None:
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    data = Data(**raw) if isinstance(raw, dict) and "x" in raw else raw

    x = data.x.detach().cpu().numpy()
    edge_index = data.edge_index.detach().cpu().numpy()
    if getattr(data, "edge_attr", None) is not None:
        edge_attr = data.edge_attr.view(-1).detach().cpu().numpy()
    else:
        edge_attr = np.zeros(edge_index.shape[1], dtype=np.int64)

    node_depth = getattr(data, "node_depth", None)
    if node_depth is not None:
        node_depth = node_depth.detach().cpu().numpy()
    else:
        node_depth = np.arange(len(x), dtype=np.int64)

    # 用 DiGraph 即可（AIG 不需要平行边）；出现重复会在 graphml2bench 的 sanitize 再去重
    G = nx.DiGraph()
    for i in range(len(x)):
        G.add_node(
            str(i),
            node_type=int(x[i, 0]),
            inverted_preds=int(x[i, 1]),
            depth=int(node_depth[i]),
        )

    for i in range(edge_index.shape[1]):
        src = str(int(edge_index[0, i]))
        dst = str(int(edge_index[1, i]))
        inv = int(edge_attr[i])
        if src == dst:
            continue
        G.add_edge(src, dst, edge_type=inv)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nx.write_graphml(G, out_path)
    if verbose:
        print(f"[INFO] Saved GraphML: {out_path} (nodes={len(x)}, edges={edge_index.shape[1]})")


def main():
    p = argparse.ArgumentParser(description="Convert PyG .pt (AIG) to GraphML")
    p.add_argument("-i", "--input", required=True, help=".pt file path")
    p.add_argument("-o", "--output", required=True, help=".graphml output path")
    args = p.parse_args()
    pt_to_graphml(args.input, args.output)


if __name__ == "__main__":
    main()

"""
python utilities/pt2graphml.py -i results/ISCAS85/AIGfake/sample_0.pt \
-o results/ISCAS85/graphml_fake/sample_0.graphml
"""