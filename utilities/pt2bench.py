import torch
import os
from torch_geometric.data import Data
from collections import defaultdict

PI, PO, AND = 0, 1, 2


def _sanitize_and_orient_edges(edge_index, edge_attr, node_types, node_depth=None, verbose=True):
    import numpy as np
    num_nodes = node_types.shape[0]
    priority = {PI: 0, AND: 1, PO: 2}

    if edge_index is None:
        return np.zeros((2, 0), dtype=int), np.zeros((0,), dtype=int)

    e_src = edge_index[0].tolist()
    e_dst = edge_index[1].tolist()
    if edge_attr is None:
        e_inv = [0] * len(e_src)
    else:
        # Robust flatten for numpy/list/torch
        try:
            ea_arr = np.asarray(edge_attr).reshape(-1)
            e_inv = ea_arr.tolist()
        except Exception:
            # Fallback to list
            e_inv = list(edge_attr)

    total = len(e_src)
    removed_self = removed_oob = removed_pp = flips = 0
    cleaned = []

    for s, d, inv in zip(e_src, e_dst, e_inv):
        if s is None or d is None:
            continue
        if s < 0 or d < 0 or s >= num_nodes or d >= num_nodes:
            removed_oob += 1
            continue
        if s == d:
            removed_self += 1
            continue
        ts, td = int(node_types[s]), int(node_types[d])
        # drop PI->PI or PO->PO
        if (ts == PI and td == PI) or (ts == PO and td == PO):
            removed_pp += 1
            continue
        # prefer into non-PI
        if td == PI and ts != PI:
            s, d, ts, td = d, s, td, ts
            flips += 1
        # avoid from PO
        if ts == PO and td != PO:
            s, d, ts, td = d, s, td, ts
            flips += 1
        # prefer increasing depth
        if node_depth is not None:
            ds = float(node_depth[s])
            dd = float(node_depth[d])
            if ds > dd:
                s, d, ts, td = d, s, td, ts
                flips += 1
        # enforce type priority
        if priority.get(ts, 9) > priority.get(td, -1):
            s, d, ts, td = d, s, td, ts
            flips += 1
        # final guard
        if (ts == PI and td == PI) or (ts == PO and td == PO):
            removed_pp += 1
            continue
        cleaned.append((int(s), int(d), int(inv)))

    # dedup
    seen = set()
    deduped = []
    for s, d, inv in cleaned:
        key = (s, d, inv)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((s, d, inv))

    # enforce strictly increasing depth if provided (adaptive)
    depth_filtered = []
    dropped_non_increasing = 0
    if node_depth is not None and len(deduped) > 0:
        for s, d, inv in deduped:
            ds = float(node_depth[s])
            dd = float(node_depth[d])
            if ds < dd:
                depth_filtered.append((s, d, inv))
            else:
                dropped_non_increasing += 1
        # If depth filtering removes too many edges, disable it (fallback to type-only + cycle breaking)
        if dropped_non_increasing > 0.3 * len(deduped):
            depth_filtered = deduped
            depth_filter_disabled = True
        else:
            depth_filter_disabled = False
    else:
        depth_filtered = deduped
        depth_filter_disabled = False

    # break residual cycles via Kahn: iteratively remove one incoming edge from nodes in cycles
    def break_cycles_trimming(edges, n):
        if not edges:
            return [], 0
        from collections import defaultdict, deque
        edges = list(edges)
        removed_cycle_edges = 0
        while True:
            indeg = [0] * n
            g = [[] for _ in range(n)]
            for s, d, _ in edges:
                if 0 <= s < n and 0 <= d < n:
                    g[s].append(d)
                    indeg[d] += 1
            q = deque([i for i in range(n) if indeg[i] == 0])
            seen = 0
            order = []
            while q:
                u = q.popleft()
                order.append(u)
                seen += 1
                for v in g[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
            if seen == n:
                # DAG
                return edges, removed_cycle_edges
            # find a node still with indegree>0 to cut an incoming edge
            victim = None
            for i in range(n):
                if indeg[i] > 0:
                    victim = i
                    break
            if victim is None:
                return edges, removed_cycle_edges
            # choose an incoming edge to victim to remove; prefer one with non-increasing depth
            incoming_idx = -1
            for idx, (s, d, inv) in enumerate(edges):
                if d == victim:
                    if node_depth is not None and not (float(node_depth[s]) < float(node_depth[d])):
                        incoming_idx = idx
                        break
                    if incoming_idx == -1:
                        incoming_idx = idx
            if incoming_idx == -1:
                return edges, removed_cycle_edges
            edges.pop(incoming_idx)
            removed_cycle_edges += 1

    broken_edges, removed_cycle_edges = break_cycles_trimming(depth_filtered, num_nodes)

    if verbose:
        kept = len(deduped)
        kept2 = len(depth_filtered)
        kept3 = len(broken_edges)
        print(f"[CLEAN] total={total} kept={kept} removed_self={removed_self} removed_oob={removed_oob} removed_PP={removed_pp} flips={flips}")
        if node_depth is not None:
            print(f"[CLEAN] dropped_non_increasing={dropped_non_increasing}")
            if depth_filter_disabled:
                print(f"[CLEAN] depth_filter_disabled=True (fallback to type-only)")
        if removed_cycle_edges > 0:
            print(f"[CLEAN] removed_cycle_edges={removed_cycle_edges} (DAG enforced)")

    if not broken_edges:
        return np.zeros((2, 0), dtype=int), np.zeros((0,), dtype=int)

    new_src, new_dst, new_inv = zip(*broken_edges)
    return (torch.tensor([new_src, new_dst], dtype=torch.long).numpy(),
            torch.tensor(new_inv, dtype=torch.long).numpy())


essential_types = {PI, AND, PO}

def _reachable_from_inputs(edge_index, node_types):
    num_nodes = node_types.shape[0]
    succ = defaultdict(list)
    if edge_index is not None and edge_index.shape[1] > 0:
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            succ[int(s)].append(int(d))
    frontier = [i for i in range(num_nodes) if int(node_types[i]) == PI]
    seen = set(frontier)
    while frontier:
        nxt = []
        for u in frontier:
            for v in succ.get(u, []):
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return seen


def _reverse_reachable_to_pos(edge_index, node_types):
    num_nodes = node_types.shape[0]
    preds = defaultdict(list)
    if edge_index is not None and edge_index.shape[1] > 0:
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            preds[int(d)].append(int(s))
    frontier = [i for i in range(num_nodes) if int(node_types[i]) == PO]
    seen = set(frontier)
    while frontier:
        nxt = []
        for v in frontier:
            for p in preds.get(v, []):
                if p not in seen:
                    seen.add(p)
                    nxt.append(p)
        frontier = nxt
    return seen


def pt_to_bench(pt_path, out_path, verbose=True):
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    data = Data(**raw) if isinstance(raw, dict) and "x" in raw else raw

    x = data.x.numpy()
    node_types = x[:, 0].astype(int)
    num_nodes = x.shape[0]

    node_depth = getattr(data, 'node_depth', None)
    if node_depth is not None:
        try:
            node_depth = node_depth.numpy()
        except Exception:
            pass

    edge_index = data.edge_index.numpy() if getattr(data, 'edge_index', None) is not None else torch.zeros((2,0), dtype=torch.long).numpy()
    edge_attr = data.edge_attr.view(-1).numpy() if getattr(data, 'edge_attr', None) is not None else [0]* (edge_index.shape[1] if edge_index is not None else 0)

    # sanitize & orient
    edge_index, edge_attr = _sanitize_and_orient_edges(edge_index, edge_attr, node_types, node_depth=node_depth, verbose=verbose)

    # preds map
    edge_map = {i: [] for i in range(num_nodes)}
    for e in range(edge_index.shape[1] if edge_index is not None else 0):
        s = int(edge_index[0, e])
        d = int(edge_index[1, e])
        inv = int(edge_attr[e])
        edge_map.setdefault(d, []).append((s, inv))

    # reachability: forward from PI and backward from PO, on-path = intersection
    reachable = _reachable_from_inputs(edge_index, node_types)
    rev_reach = _reverse_reachable_to_pos(edge_index, node_types)
    on_path = reachable & rev_reach
    # If there is no clean PI->PO path (common for noisy samples), fall back to a
    # maximally connected region so that we can still emit a structurally valid
    # circuit instead of returning an empty netlist.
    if on_path:
        active_region = set(on_path)
    elif reachable:
        active_region = set(reachable)
    elif rev_reach:
        active_region = set(rev_reach)
    else:
        active_region = set(i for i in range(num_nodes) if node_types[i] in (PI, AND, PO))
    num_reach_and = sum(1 for i in range(num_nodes) if node_types[i] == AND and i in on_path)
    num_reach_po = sum(1 for i in range(num_nodes) if node_types[i] == PO and i in on_path)
    if verbose:
        print(f"[REACH] reachable_nodes={len(reachable)} on_path={len(on_path)} AND_on_path={num_reach_and} PO_on_path={num_reach_po}")

    bench_name = os.path.basename(pt_path)
    lines = [f'# Benchmark "{bench_name}" written by AIG-GAN']

    # inputs
    node_names = {}
    defined_signals = set()
    input_counter = 1
    for i in range(num_nodes):
        if node_types[i] == PI:
            name = f"N{input_counter}"
            node_names[i] = name
            lines.append(f"INPUT({name})")
            defined_signals.add(name)
            input_counter += 1
        else:
            node_names[i] = f"n{i}"

    # AND/internal
    not_defs = set()
    body_lines = []
    and_defined = set()
    for i in range(num_nodes):
        if node_types[i] != AND or i not in active_region:
            continue
        preds = [(s, inv) for (s, inv) in edge_map.get(i, []) if s in active_region and node_types[s] in (PI, AND)]
        if node_depth is not None:
            preds = sorted(preds, key=lambda t: (float(node_depth[t[0]]), t[0]))
        else:
            preds = sorted(preds, key=lambda t: t[0])
        if len(preds) == 0:
            continue
        in_exprs = []
        for src, inv in preds:
            base = node_names[src]
            if inv:
                inv_name = f"{base}_not"
                if inv_name not in not_defs:
                    body_lines.append(f"{inv_name} = NOT({base})")
                    not_defs.add(inv_name)
                    defined_signals.add(inv_name)
                base = inv_name
            in_exprs.append(base)
        # Ensure at least two fanins for structural AIG.
        if len(in_exprs) == 1:
            # Duplicate the single fanin to synthesise a degenerate AND (acts as buffer).
            in_exprs.append(in_exprs[0])
        # Build only AND logic (AIG 2-input). If >2 inputs, fold to tree.
        if len(in_exprs) == 2:
            body_lines.append(f"{node_names[i]} = AND({in_exprs[0]}, {in_exprs[1]})")
        else:
            acc = in_exprs[0]
            for k, term in enumerate(in_exprs[1:-1], start=1):
                tmp = f"{node_names[i]}_and{k}"
                body_lines.append(f"{tmp} = AND({acc}, {term})")
                defined_signals.add(tmp)
                acc = tmp
            body_lines.append(f"{node_names[i]} = AND({acc}, {in_exprs[-1]})")
        defined_signals.add(node_names[i])
        and_defined.add(i)

    # outputs
    output_counter = 1
    added_outputs = set()
    outputs_emitted = 0
    def resolve_output_signal(cand_s, cand_inv):
        """Return a defined signal name to drive output or None."""
        base = node_names[cand_s]
        if node_types[cand_s] == AND and cand_s not in and_defined:
            return None
        if node_types[cand_s] == PI and base not in defined_signals:
            defined_signals.add(base)
        signal = base
        if cand_inv:
            inv_name = f"{base}_not_out{output_counter}"
            body_lines.append(f"{inv_name} = NOT({base})")
            defined_signals.add(inv_name)
            signal = inv_name
        return signal

    # Candidate POs: prefer those on a valid path; if none exist, accept any PO
    # that has at least one predecessor to keep the output list non-empty.
    po_candidates = []
    for i in range(num_nodes):
        if node_types[i] != PO:
            continue
        if i in on_path or (not on_path and (i in active_region or edge_map.get(i))):
            po_candidates.append(i)

    for i in po_candidates:
        preds = [(s, inv) for (s, inv) in edge_map.get(i, []) if s in on_path]
        if len(preds) == 0:
            # Try relaxed predecessors if strict on_path yielded nothing.
            preds = [(s, inv) for (s, inv) in edge_map.get(i, []) if s in active_region]
        if len(preds) == 0:
            preds = edge_map.get(i, [])
        if len(preds) == 0:
            continue
        # Prefer AND-driven predecessor, then by depth
        def sort_key(tup):
            s, inv = tup
            is_and = 1 if node_types[s] == AND else 0
            d = float(node_depth[s]) if node_depth is not None else 0.0
            return (-is_and, d, s)
        preds_sorted = sorted(preds, key=sort_key)
        base = None
        for cand_s, cand_inv in preds_sorted:
            # Prefer AND nodes with definitions, fall back to defined PIs.
            if node_types[cand_s] == AND and cand_s not in and_defined:
                continue
            candidate = resolve_output_signal(cand_s, cand_inv)
            if candidate is not None:
                base = candidate
                break
        if base is None:
            # As a last resort, try any predecessor regardless of type.
            for cand_s, cand_inv in preds_sorted:
                candidate = resolve_output_signal(cand_s, cand_inv)
                if candidate is not None:
                    base = candidate
                    break
        if base is None:
            continue
        driver_signal = base
        # 如果 base 已在输出集合中，追加去重后缀
        if base in added_outputs:
            base = f"{base}_dup{output_counter}"
        lines.append(f"OUTPUT({base})")
        added_outputs.add(base)
        output_counter += 1
        outputs_emitted += 1

    # Fallback: if no outputs emitted but there are ANDs on-path, synthesize outputs from deepest ANDs
    if outputs_emitted == 0:
        and_nodes_on_path = [i for i in range(num_nodes) if node_types[i] == AND and i in active_region and i in and_defined]
        if len(and_nodes_on_path) == 0:
            # as last resort, pick any AND that we defined
            and_nodes_on_path = [i for i in range(num_nodes) if node_types[i] == AND and i in and_defined]
        if len(and_nodes_on_path) > 0:
            # pick up to 2 deepest ANDs
            if node_depth is not None:
                and_nodes_on_path.sort(key=lambda i: (float(node_depth[i]), i), reverse=True)
            else:
                and_nodes_on_path.sort(reverse=True)
            for pick in and_nodes_on_path[:2]:
                name = node_names[pick]
                if name not in added_outputs:
                    lines.append(f"OUTPUT({name})")
                    added_outputs.add(name)
                    output_counter += 1
        elif reachable:
            # If we still have nothing, expose a couple of primary inputs so the
            # bench remains syntactically valid even for trivial graphs.
            pi_fallback = [i for i in range(num_nodes) if node_types[i] == PI]
            for pick in pi_fallback[: max(1, min(2, len(pi_fallback)) )]:
                name = node_names[pick]
                if name not in added_outputs:
                    lines.append(f"OUTPUT({name})")
                    added_outputs.add(name)
                    output_counter += 1

    lines.extend(body_lines)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    if verbose:
        and_count = sum(1 for b in body_lines if " = AND(" in b)
        print(f"[INFO] Saved .bench: {out_path} (nodes={num_nodes}, outputs={output_counter-1}, ANDs={and_count})")
        if and_count == 0:
            print("[WARN] No AND gates emitted. Upstream graph may be degenerate.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert PyG .pt -> valid .bench for ABC (AIG)")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()
    pt_to_bench(args.input, args.output)
