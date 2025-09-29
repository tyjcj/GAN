# GAN/constraints.py
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Optional
import random
from collections import deque, defaultdict


def topological_sort_kahn(num_nodes: int, edge_index: torch.Tensor) -> Tuple[bool, List[int]]:
    """
    Return (is_dag, topo_order). If not DAG, returned order is partial.
    edge_index: [2, E]
    """
    indeg = [0] * num_nodes
    g = [[] for _ in range(num_nodes)]
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


def remove_duplicate_edges(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    pairs = {}
    new_src = []
    new_dst = []
    new_attr = []
    for i, (u, v) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        key = (u, v)
        if key in pairs:
            # if duplicate, keep first; combine invert flags as OR
            if edge_attr is not None:
                pairs[key] = pairs[key] | bool(edge_attr[i].item())
            continue
        pairs[key] = bool(edge_attr[i].item()) if edge_attr is not None else False
        new_src.append(u)
        new_dst.append(v)
        new_attr.append(int(edge_attr[i].item()) if edge_attr is not None else 0)
    ei = torch.tensor([new_src, new_dst], dtype=torch.long)
    ea = torch.tensor(new_attr, dtype=torch.long).unsqueeze(1) if edge_attr is not None else None
    return ei, ea


def compute_node_indegrees(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    indeg = torch.zeros(num_nodes, dtype=torch.long)
    if edge_index.numel() == 0:
        return indeg
    for v in edge_index[1].tolist():
        indeg[v] += 1
    return indeg


def compute_predecessors_list(num_nodes: int, edge_index: torch.Tensor) -> List[List[int]]:
    preds = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return preds
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        preds[v].append(u)
    return preds


def compute_node_depths_from_PIs(num_nodes: int, edge_index: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
    """
    Compute shortest path depth from PIs (type==0) using BFS.
    If node unreachable, set depth to -1.
    """
    g = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        g[u].append(v)
    depths = torch.full((num_nodes,), -1, dtype=torch.long)
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


def enforce_aig_constraints(data: Data, max_prev_candidates: int = 4096, allow_fix_cycles: bool = True) -> Data:
    """
    Hard-enforce AIG constraints (mutates Data).
    - PI nodes (type==0): indegree == 0
    - AND nodes (type==2): indegree == 2
    - PO nodes (type==1): indegree == 1
    - Ensure edges are from lower depth to higher depth (if node_depth given)
    - Remove duplicates and self loops
    - Fix cycles by removing incoming edges of nodes in cycles (iterative)
    - Compute node_depth after fixes and set data.node_depth
    - Compute inverted count per node and write to data.x[:,1]
    """
    num_nodes = data.x.size(0)
    device = data.x.device

    if data.edge_index is None or data.edge_index.numel() == 0:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        data.edge_attr = torch.zeros((0, 1), dtype=torch.long, device=device)
        # fix node_depth
        if not hasattr(data, 'node_depth') or data.node_depth is None:
            data.node_depth = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        return data

    # 1) remove self loops
    mask = data.edge_index[0] != data.edge_index[1]
    data.edge_index = data.edge_index[:, mask]
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[mask]

    # 2) remove duplicates
    data.edge_index, data.edge_attr = remove_duplicate_edges(data.edge_index, data.edge_attr)

    # 3) ensure edges go from lower depth to higher depth if node_depth provided
    if hasattr(data, 'node_depth') and data.node_depth is not None:
        nd = data.node_depth
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()
        keep_mask = []
        for u, v in zip(src, dst):
            du = int(nd[u].item()) if nd[u].item() >= 0 else None
            dv = int(nd[v].item()) if nd[v].item() >= 0 else None
            if du is not None and dv is not None:
                if du < dv:
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            else:
                # if depth unknown, keep for now
                keep_mask.append(True)
        if len(keep_mask) > 0:
            mask = torch.tensor(keep_mask, dtype=torch.bool, device=data.edge_index.device)
            data.edge_index = data.edge_index[:, mask]
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[mask]

    # 4) enforce indegree constraints
    indeg = compute_node_indegrees(num_nodes, data.edge_index)
    preds = compute_predecessors_list(num_nodes, data.edge_index)

    node_types = data.x[:, 0].long()  # expected 0 PI,1 PO,2 AND

    new_src = []
    new_dst = []
    new_attr = []

    # keep edges that are needed and adjust others
    # We'll rebuild edges: for each node v, determine its required predecessors
    for v in range(num_nodes):
        t = int(node_types[v].item())
        cur_preds = preds[v]
        if t == 0:
            # PI: indegree must be 0 -> drop any incoming edges
            continue
        elif t == 1:
            # PO: indegree must be 1
            if len(cur_preds) >= 1:
                u = cur_preds[0]
                # Keep first (could be randomized)
                # find corresponding attr
                for i in range(data.edge_index.size(1)):
                    if int(data.edge_index[0, i].item()) == u and int(data.edge_index[1, i].item()) == v:
                        new_src.append(u); new_dst.append(v); new_attr.append(int(data.edge_attr[i].item()) if data.edge_attr is not None else 0)
                        break
            else:
                # no predecessor -> connect to a random valid source (choose any PI or AND with depth lower)
                candidates = [i for i in range(num_nodes) if i != v and int(node_types[i].item()) in (0, 2)]
                if candidates:
                    u = random.choice(candidates)
                    new_src.append(u); new_dst.append(v); new_attr.append(0)
        elif t == 2:
            # AND: indegree must be 2
            if len(cur_preds) >= 2:
                # keep first two
                kept = cur_preds[:2]
                for u in kept:
                    # find attr
                    for i in range(data.edge_index.size(1)):
                        if int(data.edge_index[0, i].item()) == u and int(data.edge_index[1, i].item()) == v:
                            new_src.append(u); new_dst.append(v); new_attr.append(int(data.edge_attr[i].item()) if data.edge_attr is not None else 0)
                            break
            else:
                # fill missing preds
                kept = list(cur_preds)
                candidates = [i for i in range(num_nodes) if i != v and int(node_types[i].item()) in (0, 2)]
                random.shuffle(candidates)
                for u in candidates:
                    if u not in kept:
                        kept.append(u)
                    if len(kept) == 2:
                        break
                # if still less than 2 (rare), allow duplicates but will avoid contradictory later
                while len(kept) < 2:
                    kept.append(kept[-1] if kept else 0)
                for u in kept[:2]:
                    new_src.append(u); new_dst.append(v); new_attr.append(0)

    if len(new_src) == 0:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.edge_attr = torch.zeros((0, 1), dtype=torch.long)
    else:
        ei = torch.tensor([new_src, new_dst], dtype=torch.long)
        ea = torch.tensor(new_attr, dtype=torch.long).unsqueeze(1)
        data.edge_index = ei
        data.edge_attr = ea

    # 5) ensure no contradictory inputs: for each AND node, if inputs from same node but one inverted -> that is A & ~A -> becomes constant 0, avoid this by flipping one inversion to 0
    preds = compute_predecessors_list(num_nodes, data.edge_index)
    for v in range(num_nodes):
        if int(node_types[v].item()) == 2:
            p = preds[v]
            if len(p) == 2 and p[0] == p[1]:
                # inspect attrs
                idxs = []
                for i in range(data.edge_index.size(1)):
                    if int(data.edge_index[1, i].item()) == v:
                        idxs.append(i)
                if len(idxs) >= 2:
                    a0 = int(data.edge_attr[idxs[0]].item()); a1 = int(data.edge_attr[idxs[1]].item())
                    # if contradictory (0 and 1), then set both to 0 (prefer non-inverted)
                    if a0 != a1:
                        data.edge_attr[idxs[0]] = 0
                        data.edge_attr[idxs[1]] = 0

    # 6) remove cycles iteratively if any
    is_dag, topo = topological_sort_kahn(num_nodes, data.edge_index)
    attempts = 0
    max_attempts = num_nodes * 2
    while not is_dag and attempts < max_attempts and allow_fix_cycles:
        # find nodes not in topo (in cycle) -> remove one incoming edge for each such node
        present = set(topo)
        all_nodes = set(range(num_nodes))
        in_cycle = list(all_nodes - present)
        if not in_cycle:
            break
        for v in in_cycle:
            # remove one incoming edge (prefer from an incoming whose source has depth >= v depth if depths exist)
            rem_idx = None
            for i in range(data.edge_index.size(1)):
                if int(data.edge_index[1, i].item()) == v:
                    rem_idx = i
                    break
            if rem_idx is not None:
                mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
                mask[rem_idx] = False
                data.edge_index = data.edge_index[:, mask]
                if data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[mask]
        is_dag, topo = topological_sort_kahn(num_nodes, data.edge_index)
        attempts += 1

    # 7) recompute node_depths
    node_types = data.x[:, 0].long()
    nd = compute_node_depths_from_PIs(num_nodes, data.edge_index, node_types)
    data.node_depth = nd

    # 8) recompute inverted_count and store to x[:,1]
    inv_counts = torch.zeros(num_nodes, dtype=torch.long)
    if data.edge_attr is not None and data.edge_index is not None and data.edge_index.numel() > 0:
        dst = data.edge_index[1].tolist()
        invs = data.edge_attr.view(-1).long().tolist()
        for d, inv in zip(dst, invs):
            inv_counts[d] += int(inv)
    data.x = data.x.clone()
    # clamp inv_counts to {0,1,2}
    inv_counts = torch.clamp(inv_counts, max=2)
    data.x[:, 1] = inv_counts.to(data.x.dtype)

    return data
