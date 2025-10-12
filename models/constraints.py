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
    node_types = data.x[:, 0].long().cpu()

    # Edge tensors (on CPU for simple Python ops; move back at the end)
    ei = data.edge_index
    ea = data.edge_attr
    if ei is None:
        ei = torch.zeros((2, 0), dtype=torch.long)
    if ea is None:
        ea = torch.zeros((0, 1), dtype=torch.long)

    src = ei[0].tolist()
    dst = ei[1].tolist()

    # Basic counts
    t = node_types
    pos = type_order_index(t)  # PI->AND->PO layering

    pi_mask = (t == PI)
    po_mask = (t == PO)
    and_mask = (t == AND)

    # self-loop penalty
    selfloop = sum(1 for u, v in zip(src, dst) if u == v)

    # edges into PI & out of PO
    pi_in = sum(1 for u, v in zip(src, dst) if int(t[v]) == PI)
    po_out = sum(1 for u, v in zip(src, dst) if int(t[u]) == PO)

    # back-edges wrt PI->AND->PO layering
    backedge = sum(1 for u, v in zip(src, dst) if not (int(pos[u]) < int(pos[v])))

    # fanin-type violations (dst in {AND,PO} but src not in {PI,AND})
    fanin_type = 0
    for u, v in zip(src, dst):
        if int(t[v]) in (AND, PO):
            if int(t[u]) not in (PI, AND):
                fanin_type += 1

    # duplicate fanin penalty: (#in_edges - #unique_pred) accumulated over nodes
    from collections import defaultdict, deque
    preds = defaultdict(list)
    for u, v in zip(src, dst):
        preds[v].append(u)
    dup_fanin = 0
    for v, lst in preds.items():
        dup_fanin += max(0, len(lst) - len(set(lst)))

    # indegree target penalty
    indeg = _indegree_tensor(num_nodes, ei)
    target = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_nodes):
        tt = int(t[i])
        if tt == PI:
            target[i] = 0
        elif tt == PO:
            target[i] = 1
        elif tt == AND:
            target[i] = 2
        else:
            target[i] = 0
    indeg_mse = (indeg.to(torch.float32) - target.to(torch.float32)).abs().sum()

    # reachability coverage: nodes on PI->PO paths
    g_fwd = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        g_fwd[u].append(v)
    seen_fwd = set()
    dq = deque([i for i in range(num_nodes) if int(t[i]) == PI])
    while dq:
        u = dq.popleft()
        if u in seen_fwd:
            continue
        seen_fwd.add(u)
        for v in g_fwd[u]:
            dq.append(v)
    # reverse from POs
    g_rev = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        g_rev[v].append(u)
    seen_rev = set()
    dq = deque([i for i in range(num_nodes) if int(t[i]) == PO])
    while dq:
        v = dq.popleft()
        if v in seen_rev:
            continue
        seen_rev.add(v)
        for p in g_rev[v]:
            dq.append(p)
    on_path = seen_fwd & seen_rev
    reach_violation = num_nodes - len(on_path)  # nodes unused by any PI->PO cone

    # assemble losses
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
        "pi_in": torch.tensor(float(pi_in), device=device),
        "po_out": torch.tensor(float(po_out), device=device),
        "backedge": torch.tensor(float(backedge), device=device),
        "selfloop": torch.tensor(float(selfloop), device=device),
        "dup_fanin": torch.tensor(float(dup_fanin), device=device),
        "indeg_mse": indeg_mse.to(device),
        "fanin_type": torch.tensor(float(fanin_type), device=device),
        "reach_violation": torch.tensor(float(reach_violation), device=device),
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

    ei, ea = remove_self_and_dups(data.edge_index.cpu(), data.edge_attr.cpu())
    ei = ei.to(device)
    ea = ea.to(device)

    # Collect predecessors
    preds = [[] for _ in range(num_nodes)]
    for u, v, inv in zip(ei[0].tolist(), ei[1].tolist(), ea.view(-1).tolist()):
        preds[v].append((u, int(inv)))

    new_src = []
    new_dst = []
    new_attr = []

    # Enforce constraints
    for v in range(num_nodes):
        t = int(node_types[v].item())
        cur_preds = preds[v]
        if t == 0:
            continue  # PI: drop all incoming
        elif t == 1:
            # PO: keep only one
            if len(cur_preds) >= 1:
                u, inv = cur_preds[0]
                new_src.append(u)
                new_dst.append(v)
                new_attr.append(inv)
            else:
                candidates = [i for i in range(num_nodes) if int(node_types[i].item()) in (0, 2) and i != v]
                if candidates:
                    u = random.choice(candidates)
                    new_src.append(u)
                    new_dst.append(v)
                    new_attr.append(0)
        elif t == 2:
            # AND: ensure 2 predecessors
            if len(cur_preds) >= 2:
                kept = cur_preds[:2]
                for u, inv in kept:
                    new_src.append(u)
                    new_dst.append(v)
                    new_attr.append(inv)
            else:
                kept = [u for u, inv in cur_preds]
                candidates = [i for i in range(num_nodes) if int(node_types[i].item()) in (0, 2) and i != v]
                random.shuffle(candidates)
                for u in candidates:
                    if u not in kept:
                        kept.append(u)
                    if len(kept) == 2:
                        break
                # Pad if still <2
                while len(kept) < 2:
                    kept.append(0)  # Pad with 0 if less than 2
                for u in kept[:2]:
                    new_src.append(u)
                    new_dst.append(v)
                    new_attr.append(0)

    # Handle edge case when no edges exist after processing
    if len(new_src) == 0:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        data.edge_attr = torch.zeros((0, 1), dtype=torch.long, device=device)
    else:
        data.edge_index = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        data.edge_attr = torch.tensor(new_attr, dtype=torch.long, device=device).unsqueeze(1)

    # Fix cycles: Try Kahn, remove one incoming edge per node in cycle
    is_dag, topo = topo_sort_kahn(num_nodes, data.edge_index.cpu())
    attempts = 0
    max_attempts = max(3, num_nodes // 2000)
    while (not is_dag) and attempts < max_attempts:
        present = set(topo)
        all_nodes = set(range(num_nodes))
        in_cycle = list(all_nodes - present)
        if not in_cycle:
            break
        # For each node in cycle remove one incoming edge
        mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=device)
        for idx in range(data.edge_index.size(1)):
            u = int(data.edge_index[0, idx].item())
            v = int(data.edge_index[1, idx].item())
            if v in in_cycle:
                mask[idx] = False
                break
        data.edge_index = data.edge_index[:, mask]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]
        is_dag, topo = topo_sort_kahn(num_nodes, data.edge_index.cpu())
        attempts += 1

    # Recompute node_depth
    nd = compute_node_depths(num_nodes, data.edge_index.cpu(), node_types)
    data.node_depth = nd.to(device)

    # Recompute inverted predecessor counts -> x[:, 1]
    inv_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if data.edge_index is not None and data.edge_index.numel() > 0:
        dsts = data.edge_index[1].tolist()
        invs = data.edge_attr.view(-1).long().tolist()
        for d, inv in zip(dsts, invs):
            inv_counts[d] += int(inv)

    # Ensure inv_counts matches the size of x[:, 1]
    if inv_counts.size(0) != data.x.size(0):
        inv_counts = torch.cat([inv_counts, torch.zeros(data.x.size(0) - inv_counts.size(0), dtype=torch.long, device=device)])

    inv_counts = torch.clamp(inv_counts, max=2)
    data.x = data.x.clone()
    if data.x.size(1) < 2:
        newx = torch.zeros((num_nodes, 2), dtype=data.x.dtype, device=device)
        newx[:, 0] = data.x[:, 0]
        data.x = newx
    data.x[:, 1] = inv_counts.to(dtype=data.x.dtype)

    return data

