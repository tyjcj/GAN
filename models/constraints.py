# -----------------------------
# enforce constraints & node_depth
# -----------------------------

import random
import torch
from torch_geometric.data import Data


def remove_self_and_dups(edge_index: torch.Tensor, edge_attr: torch.Tensor):
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    seen = {}
    keep_src = []
    keep_dst = []
    keep_attr = []
    for i, (u,v) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        if u == v:
            continue
        key = (u,v)
        inv = int(edge_attr[i].item()) if edge_attr is not None and edge_attr.numel()>0 else 0
        if key in seen:
            # OR combine inversion (if any edge invert)
            prev_idx = seen[key]
            if keep_attr[prev_idx] == 0 and inv == 1:
                keep_attr[prev_idx] = 1
            continue
        seen[key] = len(keep_src)
        keep_src.append(u); keep_dst.append(v); keep_attr.append(inv)
    if len(keep_src) == 0:
        return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,1), dtype=torch.long)
    ei = torch.tensor([keep_src, keep_dst], dtype=torch.long)
    ea = torch.tensor(keep_attr, dtype=torch.long).unsqueeze(1)
    return ei, ea

def topo_sort_kahn(num_nodes: int, edge_index: torch.Tensor):
    indeg = [0]*num_nodes
    g = [[] for _ in range(num_nodes)]
    if edge_index is None or edge_index.numel()==0:
        return True, list(range(num_nodes))
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u,v in zip(src,dst):
        indeg[v] += 1
        g[u].append(v)
    from collections import deque
    q = deque([i for i,d in enumerate(indeg) if d==0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v]==0:
                q.append(v)
    return (len(topo) == num_nodes), topo

def compute_node_depths(num_nodes: int, edge_index: torch.Tensor, node_types: torch.Tensor):
    # BFS-like shortest paths from PIs (node_types==0)
    g = [[] for _ in range(num_nodes)]
    if edge_index is not None and edge_index.numel()>0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u,v in zip(src,dst):
            g[u].append(v)
    from collections import deque
    depths = [-1]*num_nodes
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
    return torch.tensor(depths, dtype=torch.long)

def enforce_aig_constraints(data: Data, allow_fix_cycles: bool = True) -> Data:
    """
    Hard enforce AIG constraints and compute node_depth and x[:,1] inverted counts.
    - Remove self loops, merge duplicate edges (OR inversion)
    - For PI (type 0): indegree must be 0 -> drop incoming
    - For PO (type 1): indegree must be 1 -> keep first pred or attach a candidate
    - For AND (type 2): indegree must be 2 -> keep first two or add candidates
    - Try to fix cycles using Kahn by removing one incoming edge for nodes in cycle (up to attempts)
    - Recompute node_depth and x[:,1]
    """
    device = data.x.device
    num_nodes = data.x.size(0)
    node_types = data.x[:,0].long().cpu()
    # ensure edge_index & edge_attr exist
    if not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.numel()==0:
        data.edge_index = torch.zeros((2,0), dtype=torch.long, device=device)
        data.edge_attr = torch.zeros((0,1), dtype=torch.long, device=device)
        data.node_depth = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        data.x[:,1] = 0
        return data

    ei, ea = remove_self_and_dups(data.edge_index.cpu(), data.edge_attr.cpu())
    ei = ei.to(device); ea = ea.to(device)
    # collect preds
    preds = [[] for _ in range(num_nodes)]
    for u,v,inv in zip(ei[0].tolist(), ei[1].tolist(), ea.view(-1).tolist()):
        preds[v].append((u, int(inv)))

    new_src = []
    new_dst = []
    new_attr = []

    for v in range(num_nodes):
        t = int(node_types[v].item())
        cur_preds = preds[v]
        if t == 0:
            # PI: drop all incoming
            continue
        elif t == 1:
            # PO: keep only one
            if len(cur_preds) >= 1:
                u,inv = cur_preds[0]
                new_src.append(u); new_dst.append(v); new_attr.append(inv)
            else:
                # choose any existing node as driver (prefer PI)
                candidates = [i for i in range(num_nodes) if int(node_types[i].item()) in (0,2) and i != v]
                if candidates:
                    u = random.choice(candidates)
                    new_src.append(u); new_dst.append(v); new_attr.append(0)
        elif t == 2:
            # AND: ensure 2 predecessors
            if len(cur_preds) >= 2:
                kept = cur_preds[:2]
                for u,inv in kept:
                    new_src.append(u); new_dst.append(v); new_attr.append(inv)
            else:
                kept = [u for u,inv in cur_preds]
                candidates = [i for i in range(num_nodes) if int(node_types[i].item()) in (0,2) and i != v]
                random.shuffle(candidates)
                for u in candidates:
                    if u not in kept:
                        kept.append(u)
                    if len(kept) == 2:
                        break
                # pad if still <2
                while len(kept) < 2:
                    kept.append(0)
                for u in kept[:2]:
                    new_src.append(u); new_dst.append(v); new_attr.append(0)

    if len(new_src) == 0:
        data.edge_index = torch.zeros((2,0), dtype=torch.long, device=device)
        data.edge_attr = torch.zeros((0,1), dtype=torch.long, device=device)
    else:
        data.edge_index = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        data.edge_attr = torch.tensor(new_attr, dtype=torch.long, device=device).unsqueeze(1)

    # fix cycles: try Kahn, remove one incoming edge per node in cycle
    is_dag, topo = topo_sort_kahn(num_nodes, data.edge_index.cpu())
    attempts = 0
    max_attempts = max(3, num_nodes // 2000)
    while (not is_dag) and attempts < max_attempts and allow_fix_cycles:
        present = set(topo)
        all_nodes = set(range(num_nodes))
        in_cycle = list(all_nodes - present)
        if not in_cycle:
            break
        # for each node in cycle remove first incoming edge
        mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=device)
        for idx in range(data.edge_index.size(1)):
            u = int(data.edge_index[0, idx].item()); v = int(data.edge_index[1, idx].item())
            if v in in_cycle:
                mask[idx] = False
                break
        data.edge_index = data.edge_index[:, mask]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]
        is_dag, topo = topo_sort_kahn(num_nodes, data.edge_index.cpu())
        attempts += 1

    # recompute node_depth
    nd = compute_node_depths(num_nodes, data.edge_index.cpu(), node_types)
    data.node_depth = nd.to(device)

    # recompute inverted predecessor counts -> x[:,1]
    inv_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if data.edge_index is not None and data.edge_index.numel()>0:
        dsts = data.edge_index[1].tolist()
        invs = data.edge_attr.view(-1).long().tolist()
        for d, inv in zip(dsts, invs):
            inv_counts[d] += int(inv)
    # clamp to 2 as spec says 0/1/2 possible
    inv_counts = torch.clamp(inv_counts, max=2)
    data.x = data.x.clone()
    # ensure x has second column
    if data.x.size(1) < 2:
        newx = torch.zeros((num_nodes,2), dtype=data.x.dtype, device=device)
        newx[:,0] = data.x[:,0]
        data.x = newx
    data.x[:,1] = inv_counts.to(dtype=data.x.dtype)

    return data