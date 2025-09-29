# models/constraints.py
import torch
from torch_geometric.data import Data
from collections import deque
import random

def topological_sort_kahn(num_nodes: int, edge_index: torch.Tensor):
    indeg = [0] * num_nodes
    g = [[] for _ in range(num_nodes)]
    if edge_index is None or edge_index.numel() == 0:
        return True, list(range(num_nodes))
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        indeg[v] += 1
        g[u].append(v)
    q = deque([i for i, d in enumerate(indeg) if d == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    is_dag = (len(topo) == num_nodes)
    return is_dag, topo

def remove_self_and_dups(edge_index: torch.Tensor, edge_attr: torch.Tensor):
    if edge_index is None or edge_index.numel() == 0:
        return edge_index.new_empty((2,0)), edge_attr.new_empty((0,1))
    pairs = {}
    new_src = []
    new_dst = []
    new_attr = []
    for i, (u, v) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        if u == v:
            continue
        key = (u, v)
        inv = int(edge_attr[i].item()) if edge_attr is not None else 0
        if key in pairs:
            # merge with OR
            pairs[key] = pairs[key] | inv
            # already added first occurrence; update attr if needed
            # find index and update new_attr
            # (we keep first occurrence)
            continue
        pairs[key] = inv
        new_src.append(u)
        new_dst.append(v)
        new_attr.append(inv)
    if len(new_src) == 0:
        return edge_index.new_empty((2,0)), edge_attr.new_empty((0,1))
    ei = torch.tensor([new_src, new_dst], dtype=torch.long, device=edge_index.device)
    ea = torch.tensor(new_attr, dtype=torch.long, device=edge_index.device).unsqueeze(1)
    return ei, ea

def compute_node_indegrees(num_nodes: int, edge_index: torch.Tensor):
    indeg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    if edge_index is None or edge_index.numel() == 0:
        return indeg
    for v in edge_index[1].tolist():
        indeg[v] += 1
    return indeg

def compute_node_depths_from_PIs(num_nodes: int, edge_index: torch.Tensor, node_types: torch.Tensor):
    g = [[] for _ in range(num_nodes)]
    if edge_index is not None and edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            g[u].append(v)
    depths = torch.full((num_nodes,), -1, dtype=torch.long, device=node_types.device)
    q = deque()
    for i in range(num_nodes):
        if int(node_types[i].item()) == 0:
            depths[i] = 0
            q.append(i)
    while q:
        u = q.popleft()
        for v in g[u]:
            if depths[v] == -1 or depths[v] > depths[u] + 1:
                depths[v] = depths[u] + 1
                q.append(v)
    return depths

def enforce_aig_constraints(data: Data, allow_fix_cycles: bool = True) -> Data:
    """
    Hard-enforce AIG constraints, but be conservative to avoid overcorrection.
    Ensures:
      - no self-loops
      - duplicate edges removed (keep OR'ed inversion)
      - PI indegree = 0 (drop incoming)
      - PO indegree = 1 (keep first predecessor)
      - AND indegree = 2 (if less, add candidate; if more, keep first two)
      - recalc node_depth and x[:,1] as inverted counts
    """
    num_nodes = data.x.size(0)
    device = data.x.device

    if data.edge_index is None or data.edge_index.numel() == 0:
        data.edge_index = torch.zeros((2,0), dtype=torch.long, device=device)
        data.edge_attr = torch.zeros((0,1), dtype=torch.long, device=device)
        data.node_depth = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        data.x[:,1] = 0
        return data

    ei, ea = remove_self_and_dups(data.edge_index, data.edge_attr)
    data.edge_index = ei
    data.edge_attr = ea

    node_types = data.x[:,0].long()
    preds = [[] for _ in range(num_nodes)]
    if data.edge_index.numel() > 0:
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()
        for u,v in zip(src,dst):
            preds[v].append(u)

    new_src = []
    new_dst = []
    new_attr = []

    for v in range(num_nodes):
        t = int(node_types[v].item())
        cur_preds = preds[v]
        if t == 0:
            # PI: indegree must be 0, drop all incoming
            continue
        elif t == 1:
            # PO: keep only 1 predecessor; if none, connect to a random PI or AND of lower depth
            if len(cur_preds) >= 1:
                u = cur_preds[0]
                # find attr
                for i in range(ei.size(1)):
                    if int(ei[0,i].item())==u and int(ei[1,i].item())==v:
                        new_src.append(u); new_dst.append(v); new_attr.append(int(ea[i].item()))
                        break
            else:
                # choose a candidate
                candidates = [i for i in range(num_nodes) if int(node_types[i].item()) in (0,2) and i!=v]
                if candidates:
                    u = random.choice(candidates)
                    new_src.append(u); new_dst.append(v); new_attr.append(0)
        elif t == 2:
            # AND: must have 2 predecessors
            if len(cur_preds) >= 2:
                kept = cur_preds[:2]
                for u in kept:
                    for i in range(ei.size(1)):
                        if int(ei[0,i].item())==u and int(ei[1,i].item())==v:
                            new_src.append(u); new_dst.append(v); new_attr.append(int(ea[i].item()))
                            break
            else:
                kept = list(cur_preds)
                candidates = [i for i in range(num_nodes) if int(node_types[i].item()) in (0,2) and i!=v]
                random.shuffle(candidates)
                for u in candidates:
                    if u not in kept:
                        kept.append(u)
                    if len(kept)==2:
                        break
                while len(kept)<2:
                    kept.append(0 if len(kept)==0 else kept[-1])
                for u in kept[:2]:
                    # default inv flag 0
                    new_src.append(u); new_dst.append(v); new_attr.append(0)

    if len(new_src)==0:
        data.edge_index = torch.zeros((2,0), dtype=torch.long, device=device)
        data.edge_attr = torch.zeros((0,1), dtype=torch.long, device=device)
    else:
        ei = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        ea = torch.tensor(new_attr, dtype=torch.long, device=device).unsqueeze(1)
        data.edge_index = ei
        data.edge_attr = ea

    # Fix cycles: try topological sort; if not dag, remove one incoming edge for nodes not in topo
    is_dag, topo = topological_sort_kahn(num_nodes, data.edge_index)
    attempts = 0
    max_attempts = max(3, num_nodes // 1000)
    while not is_dag and attempts < max_attempts and allow_fix_cycles:
        present = set(topo)
        all_nodes = set(range(num_nodes))
        in_cycle = list(all_nodes - present)
        if not in_cycle:
            break
        for v in in_cycle:
            # remove one incoming edge if exists
            rem_idx = None
            for i in range(data.edge_index.size(1)):
                if int(data.edge_index[1,i].item()) == v:
                    rem_idx = i
                    break
            if rem_idx is not None:
                mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=device)
                mask[rem_idx] = False
                data.edge_index = data.edge_index[:, mask]
                if data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[mask]
        is_dag, topo = topological_sort_kahn(num_nodes, data.edge_index)
        attempts += 1

    # Recompute node_depth from PIs (shortest path)
    nd = compute_node_depths_from_PIs(num_nodes, data.edge_index, node_types)
    data.node_depth = nd

    # Recompute inverted predecessor counts into x[:,1]
    inv_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if data.edge_index is not None and data.edge_index.numel() > 0:
        dst = data.edge_index[1].tolist()
        invs = data.edge_attr.view(-1).long().tolist()
        for d, inv in zip(dst, invs):
            inv_counts[d] += int(inv)
    inv_counts = torch.clamp(inv_counts, max=2)
    data.x = data.x.clone()
    data.x[:,1] = inv_counts.to(data.x.dtype)

    return data
