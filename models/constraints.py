import random
from typing import Dict, List, Tuple, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data


PI, PO, AND = 0, 1, 2


# ----------------------------
# Utilities (kept deterministic)
# ----------------------------
def remove_self_and_dups(edge_index: Tensor, edge_attr: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Remove self-loops and duplicate directed edges (u->v). If edge_attr is a 0/1 inversion bit,
    duplicates are merged by OR (any inversion wins).
    """
    if edge_index is None or edge_index.numel() == 0:
        return edge_index, edge_attr

    seen = {}
    keep_src = []
    keep_dst = []
    keep_attr = []

    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()

    for i, (u, v) in enumerate(zip(src_list, dst_list)):
        if u == v: 
            continue  # drop self-loop
        inv = int(edge_attr[i].item()) if edge_attr is not None else 0
        key = (u, v)
        if key in seen:
            # Merge duplicates by OR (inversion)
            j = seen[key]
            if keep_attr[j] == 0 and inv == 1:
                keep_attr[j] = 1
            continue
        seen[key] = len(keep_src)
        keep_src.append(u)
        keep_dst.append(v)
        keep_attr.append(inv)

    if len(keep_src) == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.long)

    return torch.tensor([keep_src, keep_dst], dtype=torch.long), torch.tensor(keep_attr, dtype=torch.long).unsqueeze(1)


def remove_isolated_nodes(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Remove isolated nodes (those with no incoming or outgoing edges) by finding nodes
    that are not connected to any other node.
    """
    connected_nodes = set(edge_index[0].tolist()) | set(edge_index[1].tolist())
    valid_nodes = list(connected_nodes)
    mask = torch.isin(torch.arange(num_nodes), torch.tensor(valid_nodes))
    return mask


def topo_sort_kahn(num_nodes: int, edge_index: Tensor) -> Tuple[bool, List[int]]:
    """Kahn's algorithm for topological sorting (DAG check)."""
    indeg = [0] * num_nodes
    g = [[] for _ in range(num_nodes)]
    if edge_index is None or edge_index.numel() == 0:
        return True, list(range(num_nodes))
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        indeg[v] += 1
        g[u].append(v)
    from collections import deque
    q = deque([i for i, d in enumerate(indeg) if d == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return (len(topo) == num_nodes), topo


def compute_node_depths(num_nodes: int, edge_index: Tensor, node_types: Tensor) -> Tensor:
    """
    Compute node depths via BFS from all PIs (depth(PI)=0).
    AND/PO depth is min distance from any PI along directed edges; unreachable -> -1.
    """
    g = [[] for _ in range(num_nodes)]
    if edge_index is not None and edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            g[u].append(v)
    from collections import deque
    depths = [-1] * num_nodes
    q = deque()
    for i in range(num_nodes):
        if int(node_types[i].item()) == PI:
            depths[i] = 0
            q.append(i)
    while q:
        u = q.popleft()
        for v in g[u]:
            if depths[v] == -1 or depths[v] > depths[u] + 1:
                depths[v] = depths[u] + 1
                q.append(v)
    return torch.tensor(depths, dtype=torch.long)


# ----------------------------
# Type order & masks (for conditional generation)
# ----------------------------
def type_order_index(node_types: Tensor) -> Tensor:
    """
    Create a stable order index that enforces PI -> AND -> PO layering.
    All PIs come first, then ANDs, then POs (preserving original index order inside each group).
    Returns: pos[i] = rank position of node i.
    """
    num_nodes = node_types.size(0)
    pi_idx = [i for i in range(num_nodes) if int(node_types[i].item()) == PI]
    and_idx = [i for i in range(num_nodes) if int(node_types[i].item()) == AND]
    po_idx = [i for i in range(num_nodes) if int(node_types[i].item()) == PO]
    order = pi_idx + and_idx + po_idx
    pos = torch.empty(num_nodes, dtype=torch.long)
    for r, i in enumerate(order):
        pos[i] = r
    return pos


def aig_allowed_edge_mask(node_types: Tensor, enforce_dag: bool = True) -> Tensor:
    """
    Dense (N x N) boolean mask M[u,v] for whether directed edge u->v is allowed by AIG structure:
      - No self-loops
      - Src type in {PI, AND}
      - Dst type in {AND, PO}
      - If enforce_dag: layer order PI->AND->PO and strictly forward (pos[u] < pos[v])
      - No edges into PI
      - No edges out of PO
    """
    n = node_types.size(0)
    t = node_types.long()
    pos = type_order_index(t) if enforce_dag else torch.arange(n, dtype=torch.long)

    src_ok = (t == PI) | (t == AND)
    dst_ok = (t == AND) | (t == PO)

    M = torch.zeros((n, n), dtype=torch.bool, device=t.device)
    for u in range(n):
        if not bool(src_ok[u]):
            continue
        for v in range(n):
            if u == v:
                continue
            if not bool(dst_ok[v]):
                continue
            if enforce_dag and not (int(pos[u]) < int(pos[v])):  # Check if DAG is enforced
                continue
            M[u, v] = True
    return M


# ----------------------------
# Generator-side helper masks (for DAG sampling)
# ----------------------------
def sampling_mask_topo(node_types: Tensor) -> Tensor:
    """
    Dense (N x N) boolean mask for generator sampling under strict topological order:
      - Only allow {PI,AND} -> {AND,PO}
      - No self-loops
      - Strict PI->AND->PO order and forward-only edges using type_order_index
    """
    return aig_allowed_edge_mask(node_types, enforce_dag=True)


def sampling_mask_layered(node_types: Tensor, layer_index: Tensor) -> Tensor:
    """
    Dense (N x N) mask allowing edges only from layer L to L+1 under AIG type rules.
    layer_index[i] = layer id for node i (monotone increasing toward outputs).
    """
    n = node_types.size(0)
    t = node_types.long()
    src_ok = (t == PI) | (t == AND)
    dst_ok = (t == AND) | (t == PO)
    M = torch.zeros((n, n), dtype=torch.bool, device=node_types.device)
    for u in range(n):
        if not bool(src_ok[u]):
            continue
        for v in range(n):
            if u == v:
                continue
            if not bool(dst_ok[v]):
                continue
            if int(layer_index[u].item()) + 1 == int(layer_index[v].item()):
                M[u, v] = True
    return M


def apply_edge_mask_to_logits(edge_logits: Tensor, mask: Tensor, invalid_fill: float = -1e9) -> Tensor:
    """
    Apply boolean mask over dense edge logits A (N x N): invalid positions -> large negative.
    """
    assert edge_logits.shape == mask.shape, "edge_logits and mask must have same shape"
    out = edge_logits.clone()
    out = torch.where(mask, out, torch.full_like(out, invalid_fill))
    eye = torch.eye(out.size(0), dtype=torch.bool, device=out.device)
    out = torch.where(eye, torch.full_like(out, invalid_fill), out)
    return out


def mask_edges_by_types(edge_index: Tensor, node_types: Tensor, enforce_dag: bool = True) -> Tensor:
    """
    Given a sparse edge_index, return boolean mask (E,) that marks whether each edge respects the allowed mask.
    Does not change the graph, only computes which are valid wrt AIG structural rules (excluding fan-in counts).
    """
    M = aig_allowed_edge_mask(node_types, enforce_dag=enforce_dag)
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=node_types.device)
    src = edge_index[0]
    dst = edge_index[1]
    return M[src, dst]


# ----------------------------
# Penalties for training (no repair)
# ----------------------------
def _indegree_tensor(num_nodes: int, edge_index: Optional[Tensor]) -> Tensor:
    """
    Calculate the indegree (number of incoming edges) for each node.
    """
    indeg = torch.zeros(num_nodes, dtype=torch.long)
    if edge_index is not None and edge_index.numel() > 0:
        for v in edge_index[1].tolist():
            indeg[v] += 1
    return indeg


def _outdegree_tensor(num_nodes: int, edge_index: Optional[Tensor]) -> Tensor:
    """
    Calculate the outdegree (number of outgoing edges) for each node.
    """
    outdeg = torch.zeros(num_nodes, dtype=torch.long)
    if edge_index is not None and edge_index.numel() > 0:
        for u in edge_index[0].tolist():
            outdeg[u] += 1
    return outdeg


def aig_constraint_penalties(data: Data, weights: Optional[Dict[str, float]] = None) -> Dict[str, Tensor]:
    """
    Compute differentiable(ish) penalties for violating AIG constraints.
    We DO NOT modify the graph. Use this in GAN training.
    """
    device = data.x.device
    num_nodes = data.x.size(0)
    t = data.x[:, 0].long()  # on device

    # edges
    ei = data.edge_index if getattr(data, 'edge_index', None) is not None else torch.zeros((2,0), dtype=torch.long, device=device)
    if ei.numel() == 0:
        # trivial terms
        zero = torch.tensor(0.0, device=device)
        terms = {k: zero for k in ["pi_in","po_out","backedge","selfloop","dup_fanin","indeg_mse","fanin_type","reach_violation"]}
        terms["total"] = zero
        return terms

    src = ei[0]
    dst = ei[1]

    pos = type_order_index(t).to(device)

    # self-loops
    selfloop = (src == dst).sum().to(torch.float32)

    # edges into PI / out of PO
    pi_in = (t[dst] == PI).sum().to(torch.float32)
    po_out = (t[src] == PO).sum().to(torch.float32)

    # back-edges wrt PI->AND->PO layering
    backedge = (~(pos[src] < pos[dst])).sum().to(torch.float32)

    # fanin-type violations
    dst_t = t[dst]
    src_t = t[src]
    fanin_type = (((dst_t == AND) | (dst_t == PO)) & (~((src_t == PI) | (src_t == AND)))).sum().to(torch.float32)

    # duplicate fanin per node: (#in_edges - #unique_pred)
    # use scatter to count in-edges; unique per (dst, src) can be approximated by unique pairs
    indeg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    indeg = indeg.scatter_add(0, dst, torch.ones_like(dst))
    # unique pairs count
    pairs = src.to(torch.int64) * num_nodes + dst.to(torch.int64)
    unique_pairs = torch.unique(pairs).numel()
    dup_fanin = (dst.numel() - unique_pairs)
    dup_fanin = torch.tensor(float(dup_fanin), device=device)

    # indegree target penalty
    target = torch.zeros(num_nodes, dtype=torch.long, device=device)
    target = torch.where(t == PI, torch.zeros_like(target), target)
    target = torch.where(t == PO, torch.ones_like(target), target)
    target = torch.where(t == AND, torch.full_like(target, 2), target)
    indeg_mse = (indeg.to(torch.float32) - target.to(torch.float32)).abs().sum()

    # reachability coverage (approx): boolean BFS via adjacency lists on device is non-trivial; use CPU-light fallback with minimal transfer
    # transfer compact edges to CPU ints for simple BFS, but only indices, not features
    src_cpu = src.detach().cpu().tolist()
    dst_cpu = dst.detach().cpu().tolist()
    from collections import deque, defaultdict as _dd
    g_fwd = _dd(list)
    g_rev = _dd(list)
    for u, v in zip(src_cpu, dst_cpu):
        g_fwd[u].append(v)
        g_rev[v].append(u)
    pis = [i for i in range(num_nodes) if int(t[i].item()) == PI]
    pos_list = [i for i in range(num_nodes) if int(t[i].item()) == PO]
    seen_fwd = set()
    dq = deque(pis)
    while dq:
        u = dq.popleft()
        if u in seen_fwd:
            continue
        seen_fwd.add(u)
        for v in g_fwd.get(u, []):
            dq.append(v)
    seen_rev = set()
    dq = deque(pos_list)
    while dq:
        v = dq.popleft()
        if v in seen_rev:
            continue
        seen_rev.add(v)
        for p in g_rev.get(v, []):
            dq.append(p)
    on_path = len(seen_fwd & seen_rev)
    reach_violation = torch.tensor(float(num_nodes - on_path), device=device)

    w = {
        "pi_in": 1.0,
        "po_out": 1.0,
        "backedge": 2.0,
        "selfloop": 3.0,
        "dup_fanin": 1.0,
        "indeg_mse": 1.0,
        "fanin_type": 1.0,
        "reach_violation": 0.5,
    }
    if weights:
        w.update(weights)

    terms = {
        "pi_in": pi_in,
        "po_out": po_out,
        "backedge": backedge,
        "selfloop": selfloop,
        "dup_fanin": dup_fanin,
        "indeg_mse": indeg_mse,
        "fanin_type": fanin_type,
        "reach_violation": reach_violation,
    }
    total = sum(w[k] * terms[k] for k in terms.keys())
    terms["total"] = total
    return terms


# ----------------------------
# Enforce AIG Constraints with self-loop, duplicate, isolated node removal
# ----------------------------
from collections import defaultdict, deque


def validate_strict_aig(data: Data, check_all_on_path: bool = True) -> List[str]:
    """
    Strictly validate AIG rules; return list of violation strings (empty -> valid).
    No modifications are performed.
    Rules:
      - Types must be in {PI, PO, AND}
      - No self-loops
      - No edges into PI; No edges out of PO
      - Src type in {PI,AND} for edges into AND/PO
      - PO indegree == 1
      - AND indegree == 2 and two distinct fanins
      - DAG: no cycles (Kahn topological sort)
      - (optional) Every node must be on some PI->PO path
      - No duplicate edges (u->v) multiples
    """
    errs: List[str] = []
    x = data.x
    if x is None or x.size(1) == 0:
        return ["data.x missing or empty"]

    node_types = x[:, 0].long().cpu()  # Node types: PI = 0, PO = 1, AND = 2
    num_nodes = node_types.size(0)

    # 1. Type sanity check (only PI, PO, AND allowed)
    bad_types = [i for i in range(num_nodes) if int(node_types[i]) not in (PI, PO, AND)]
    if bad_types:
        errs.append(f"Unknown node_type indices: {bad_types[:10]} ...")

    # 2. Edge validation (check self-loops, duplicate edges, and directionality)
    ei = data.edge_index
    ea = data.edge_attr
    if ei is None:
        ei = torch.zeros((2, 0), dtype=torch.long)
    if ea is None:
        ea = torch.zeros((0, 1), dtype=torch.long)

    src = ei[0].tolist()
    dst = ei[1].tolist()

    # 2.1 Self-loops & duplicate edges
    if any(u == v for u, v in zip(src, dst)):
        loops = [(u, v) for u, v in zip(src, dst) if u == v]
        errs.append(f"Self-loops not allowed: examples {loops[:5]} ...")

    seen_pairs = set()
    dups = []
    for u, v in zip(src, dst):
        if (u, v) in seen_pairs:
            dups.append((u, v))
        seen_pairs.add((u, v))
    if dups:
        errs.append(f"Duplicate edges found: examples {dups[:5]} ...")

    # 2.2 No edges into PI & No edges out of PO
    bad_to_pi = [(u, v) for u, v in zip(src, dst) if int(node_types[v]) == PI]
    bad_from_po = [(u, v) for u, v in zip(src, dst) if int(node_types[u]) == PO]
    if bad_to_pi:
        errs.append(f"Edges into PI not allowed: examples {bad_to_pi[:5]} ...")
    if bad_from_po:
        errs.append(f"Edges out of PO not allowed: examples {bad_from_po[:5]} ...")

    # 2.3 Src type validation: edges into AND/PO must have a valid src type (PI or AND)
    bad_src_type = [(u, v) for u, v in zip(src, dst) if
                    int(node_types[v]) in (AND, PO) and int(node_types[u]) not in (PI, AND)]
    if bad_src_type:
        errs.append(f"Edges with invalid src type: examples {bad_src_type[:5]} ...")

    # 3. Indegree constraints
    indeg = _indegree_tensor(num_nodes, ei)

    # 3.1 PO nodes must have indegree = 1
    po_nodes = [i for i in range(num_nodes) if int(node_types[i]) == PO]
    for v in po_nodes:
        if indeg[v] != 1:
            errs.append(f"[PO] node {v} indegree={int(indeg[v])} (must be 1)")

    # 3.2 AND nodes must have indegree = 2 and distinct fanins
    and_nodes = [i for i in range(num_nodes) if int(node_types[i]) == AND]
    preds = defaultdict(list)
    for u, v in zip(src, dst):
        if v in and_nodes:
            preds[v].append(u)
    for v in and_nodes:
        lst = preds[v]
        if len(lst) != 2:
            errs.append(f"[AND] node {v} indegree={len(lst)} (must be 2)")
        elif len(set(lst)) != 2:
            errs.append(f"[AND] node {v} duplicated fanin {lst}")

    # 4. DAG check (no cycles)
    ok, _ = topo_sort_kahn(num_nodes, ei)
    if not ok:
        errs.append("Graph contains a cycle (fails DAG).")

    # 5. Reachability coverage: all nodes must be on a PI->PO path (optional)
    if check_all_on_path:
        from collections import deque
        # Forward from PIs
        g_fwd = [[] for _ in range(num_nodes)]
        for u, v in zip(src, dst):
            g_fwd[u].append(v)
        seen_fwd = set()
        dq = deque([i for i in range(num_nodes) if int(node_types[i]) == PI])
        while dq:
            u = dq.popleft()
            if u in seen_fwd:
                continue
            seen_fwd.add(u)
            for v in g_fwd[u]:
                dq.append(v)

        # Reverse from POs
        g_rev = [[] for _ in range(num_nodes)]
        for u, v in zip(src, dst):
            g_rev[v].append(u)
        seen_rev = set()
        dq = deque([i for i in range(num_nodes) if int(node_types[i]) == PO])
        while dq:
            v = dq.popleft()
            if v in seen_rev:
                continue
            seen_rev.add(v)
            for p in g_rev[v]:
                dq.append(p)

        on_path = seen_fwd & seen_rev
        if len(on_path) != num_nodes:
            bad = [i for i in range(num_nodes) if i not in on_path]
            errs.append(f"Dangling/uncovered nodes (not on PI->PO path): {bad[:10]} ... (count={len(bad)})")

    return errs


def enforce_aig_constraints(data: Data, strict: bool = False, check_all_on_path: bool = True) -> Data:
    """
    Enforce AIG constraints and compute node_depth and x[:,1] inverted counts.
    """
    device = data.x.device
    num_nodes = data.x.size(0)
    node_types = data.x[:, 0].long()

    # Ensure edge_index & edge_attr exist
    if not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.numel() == 0:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        data.edge_attr = torch.zeros((0, 1), dtype=torch.long, device=device)
        data.node_depth = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        data.x[:, 1] = 0
        return data

    # 清理与快速前向过滤（GPU端）
    ei, ea = remove_self_and_dups(data.edge_index, data.edge_attr)
    if ei is None:
        ei = torch.zeros((2, 0), dtype=torch.long, device=device)
    if ea is None:
        ea = torch.zeros((0, 1), dtype=torch.long, device=device)
    ei = ei.to(device)
    ea = ea.to(device)
    # 越界过滤
    if ei.numel() > 0:
        src_valid = (ei[0] >= 0) & (ei[0] < num_nodes)
        dst_valid = (ei[1] >= 0) & (ei[1] < num_nodes)
        valid_mask = src_valid & dst_valid
        ei = ei[:, valid_mask]
        if ea.numel() > 0:
            ea = ea[valid_mask]
    # 按拓扑顺序（类型顺序）移除后向边，避免 Kahn 循环修复的高开销
    pos = type_order_index(node_types).to(device)
    if ei.numel() > 0:
        forward_mask = pos[ei[0]] < pos[ei[1]]
        ei = ei[:, forward_mask]
        if ea.numel() > 0:
            ea = ea[forward_mask]

    # Collect predecessors
    preds = [[] for _ in range(num_nodes)]
    for u, v, inv in zip(ei[0].tolist(), ei[1].tolist(), ea.view(-1).tolist()):
        preds[v].append((u, int(inv)))

    new_src = []
    new_dst = []
    new_attr = []

    # Enforce constraints (no random padding; prefer legal reconnection or skip)
    for v in range(num_nodes):
        t = int(node_types[v].item())
        cur_preds = preds[v]
        if t == 0:
            continue  # PI: drop all incoming
        elif t == 1:
            # PO: keep only one from legal predecessors; prefer AND, then deeper
            legal = [(u, inv) for (u, inv) in cur_preds if int(node_types[u].item()) in (0, 2) and u != v]
            if len(legal) >= 1:
                # choose best by type priority AND>PI, then by node_depth if exists
                nd = None
                if hasattr(data, 'node_depth') and data.node_depth is not None and data.node_depth.numel() == num_nodes:
                    nd = data.node_depth
                def po_key(ui):
                    u, inv = ui
                    is_and = 1 if int(node_types[u].item()) == AND else 0
                    depth = int(nd[u].item()) if nd is not None else 0
                    return (-is_and, -depth, u)
                u, inv = sorted(legal, key=po_key)[0]
                new_src.append(u)
                new_dst.append(v)
                new_attr.append(inv)
            # else: skip connecting this PO (remain without fanin)
        elif t == 2:
            # AND: ensure 2 predecessors from legal distinct sources; prefer PI/AND with smaller order and deeper depth
            legal = [(u, inv) for (u, inv) in cur_preds if int(node_types[u].item()) in (0, 2) and u != v]
            # add more candidates from graph if insufficient, but only legal and distinct
            if len(legal) < 2:
                present = set(u for u, _ in legal)
                extra = []
                for u in range(num_nodes):
                    if u == v or u in present:
                        continue
                    tt = int(node_types[u].item())
                    if tt in (0, 2):
                        extra.append((u, 0))
                legal.extend(extra)
            if len(legal) >= 2:
                nd = None
                if hasattr(data, 'node_depth') and data.node_depth is not None and data.node_depth.numel() == num_nodes:
                    nd = data.node_depth
                # sort by: prefer PI/AND order (PI first to ensure DAG), then deeper depth, then id
                pos = type_order_index(node_types)
                def and_key(ui):
                    u, inv = ui
                    order = int(pos[u].item())
                    depth = int(nd[u].item()) if nd is not None else 0
                    return (order, -depth, u)
                uniq = []
                seen_u = set()
                for u, inv in sorted(legal, key=and_key):
                    if u in seen_u:
                        continue
                    uniq.append((u, inv))
                    seen_u.add(u)
                    if len(uniq) == 2:
                        break
                if len(uniq) == 2:
                    for u, inv in uniq:
                        new_src.append(u)
                        new_dst.append(v)
                        new_attr.append(inv)
                # else: skip this AND (do not force padding)

    # Handle edge case when no edges exist after processing
    if len(new_src) == 0:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        data.edge_attr = torch.zeros((0, 1), dtype=torch.long, device=device)
    else:
        data.edge_index = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        data.edge_attr = torch.tensor(new_attr, dtype=torch.long, device=device).unsqueeze(1)

    # 已通过 forward_mask 剪除后向边，通常不再需要 Kahn；如 strict 则保守再跑一次 Kahn 校验
    if strict:
        is_dag, _ = topo_sort_kahn(num_nodes, ei.cpu())
        if not is_dag and ei.numel() > 0:
            # 简单移除少量入边（选择 pos 不递增的边）
            keep = pos[ei[0]] < pos[ei[1]]
            ei = ei[:, keep]
            if ea.numel() > 0:
                ea = ea[keep]

    # Recompute node_depth
    nd = compute_node_depths(num_nodes, ei.cpu(), node_types)
    data.node_depth = nd.to(device)

    # Recompute inverted predecessor counts -> x[:, 1]
    inv_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if ei is not None and ei.numel() > 0:
        dsts = ei[1].clamp(min=0, max=num_nodes-1).tolist()
        invs = ea.view(-1).long().tolist() if ea is not None and ea.numel() > 0 else [0]*len(dsts)
        for d, inv in zip(dsts, invs):
            inv_counts[d] += int(inv)

    # 尺寸对齐（健壮分支，杜绝负维度）
    target_n = int(data.x.size(0))
    curr_n = int(inv_counts.size(0))
    if curr_n < target_n:
        pad_len = target_n - curr_n
        inv_counts = torch.cat([inv_counts, torch.zeros(pad_len, dtype=torch.long, device=device)], dim=0)
    elif curr_n > target_n:
        inv_counts = inv_counts[:target_n]

    inv_counts = torch.clamp(inv_counts, max=2)
    data.x = data.x.clone()
    if data.x.size(1) < 2:
        newx = torch.zeros((num_nodes, 2), dtype=data.x.dtype, device=device)
        newx[:, 0] = data.x[:, 0]
        data.x = newx
    data.x[:, 1] = inv_counts.to(dtype=data.x.dtype)

    return data


# ----------------------------
# Fast validity check (no Python BFS): type/direction/degree only
# ----------------------------
def fast_validity_checks(data: Data) -> bool:
    """Quick checks: no self-loop; only forward type order; degrees PI=0-in, PO=1-in, AND=2-in."""
    x = data.x
    if x is None or x.size(1) == 0:
        return False
    t = x[:, 0].long()
    n = x.size(0)
    ei = data.edge_index
    if ei is None or ei.numel() == 0:
        return False
    if (ei[0] == ei[1]).any():
        return False
    pos = type_order_index(t)
    if (~(pos[ei[0]] < pos[ei[1]])).any():
        return False
    indeg = torch.zeros(n, dtype=torch.long, device=t.device).scatter_add(0, ei[1], torch.ones_like(ei[1]))
    if (indeg[t == PO] != 1).any():
        return False
    if (indeg[t == AND] != 2).any():
        return False
    if (indeg[t == PI] != 0).any():
        return False
    return True

