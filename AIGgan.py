# AIGgan.py  —— 只做采样与落盘；约束调用 constraints.py
import os
import argparse
import torch
import networkx as nx
from torch_geometric.data import Data
from utilities.data_process import load_pt_dataset
from models.generator import AIGGenerator
from models.constraints import enforce_aig_constraints  # 只调用

PI, PO, AND = 0, 1, 2

def to_networkx(d: Data) -> nx.DiGraph:
    """用于同步属性的小工具（非约束）。"""
    G = nx.DiGraph()
    N = d.x.size(0)
    for i in range(N):
        G.add_node(i, node_type=int(d.x[i, 0].item()))
    if getattr(d, "edge_index", None) is not None and d.edge_index.numel() > 0:
        E = d.edge_index.size(1)
        inv = d.edge_attr.view(-1).tolist() if getattr(d, "edge_attr", None) is not None else [0]*E
        for k in range(E):
            u = int(d.edge_index[0, k].item())
            v = int(d.edge_index[1, k].item())
            if 0 <= u < N and 0 <= v < N:
                G.add_edge(u, v, edge_type=int(inv[k]))
    return G

def resync_attrs(d: Data) -> Data:
    """
    同步两个“非结构约束”的属性：
    - x[:,1] = inverted_preds（每个节点反相前驱条数）
    - node_depth = 拓扑深度（若存在环则置0；正常 enforce 后应为 DAG）
    """
    N = d.x.size(0)
    inv_cnt = torch.zeros(N, dtype=torch.long, device=d.x.device)
    if getattr(d, "edge_index", None) is not None and d.edge_index.numel() > 0:
        for k in range(d.edge_index.size(1)):
            u = int(d.edge_index[0, k].item())
            v = int(d.edge_index[1, k].item())
            inv = int(d.edge_attr[k, 0].item()) if getattr(d, "edge_attr", None) is not None else 0
            if inv == 1:
                inv_cnt[v] += 1
    # 确保 x 形状为 [N,2]
    if d.x.size(1) < 2:
        pad = torch.zeros((N, 2 - d.x.size(1)), dtype=d.x.dtype, device=d.x.device)
        d.x = torch.cat([d.x, pad], dim=1)
    d.x[:, 1] = inv_cnt

    # depth
    try:
        G = to_networkx(d)
        order = list(nx.topological_sort(G))
        depth = {n: 0 for n in order}
        for n in order:
            for p in G.predecessors(n):
                depth[n] = max(depth[n], depth[p] + 1)
        node_depth = torch.tensor([depth[i] for i in range(N)], dtype=torch.long, device=d.x.device)
    except nx.NetworkXUnfeasible:
        node_depth = torch.zeros(N, dtype=torch.long, device=d.x.device)
    d.node_depth = node_depth
    return d

def summarize(d: Data) -> str:
    n = d.x.size(0)
    e = d.edge_index.size(1) if getattr(d, "edge_index", None) is not None else 0
    t = d.x[:,0].long()
    pi = int((t==PI).sum().item()); po = int((t==PO).sum().item()); aand = int((t==AND).sum().item())
    inv = int((d.edge_attr.view(-1)==1).sum().item()) if getattr(d, "edge_attr", None) is not None else 0
    return f"N={n}, E={e}, PI={pi}, PO={po}, AND={aand}, inv_edges={inv}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="真实AIG(.pt)数据集目录，用作模板")
    ap.add_argument("--gen-ckpt", required=True, help="训练好的生成器权重 .pt")
    ap.add_argument("--out", required=True, help="输出的样本路径，例如 results/ISCAS85/AIGfake/sample_0.pt")
    ap.add_argument("--g-hidden", type=int, default=128)
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--candidate-k", type=int, default=512)
    ap.add_argument("--template-idx", type=int, default=0, help="使用哪个模板索引")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1) 载入模板
    dataset = load_pt_dataset(args.dataset_dir, max_files=None)
    if len(dataset) == 0:
        raise RuntimeError(f"No .pt files in {args.dataset_dir}")
    tpl = dataset[min(args.template_idx, len(dataset)-1)]

    # 2) 载入生成器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = AIGGenerator(node_in_dim=2, hidden_dim=args.g_hidden, z_dim=args.z_dim, candidate_k=args.candidate_k).to(device)
    state = torch.load(args.gen_ckpt, map_location=device)
    G.load_state_dict(state)
    G.eval()

    # 3) 采样
    z = torch.randn(args.z_dim, device=device)
    tpl = tpl.to(device)
    with torch.no_grad():
        fake_raw = G(tpl, z)  # Data(x=[N,2], edge_index=[2,E], edge_attr=[E,1], node_depth=[N]?)

    print("[INFO] RAW:", summarize(fake_raw))

    # 4) 调用 constraints.py 的修复（规则都在 constraints.py）
    fake_fix = enforce_aig_constraints(fake_raw)
    fake_fix = resync_attrs(fake_fix)  # 仅同步属性（非约束）
    print("[INFO] FIX:", summarize(fake_fix))

    # 5) 保存 .pt
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(fake_fix.cpu(), args.out)
    print(f"[OK] saved {args.out}")

if __name__ == "__main__":
    main()
"""
python AIGgan.py \
  --dataset-dir data_files/datasets/ISCAS85/graph \
  --gen-ckpt   results/ISCAS85/train_results/generator_epoch9.pt \
  --out        results/ISCAS85/AIGfake/sample_0.pt \
  --candidate-k 64

"""