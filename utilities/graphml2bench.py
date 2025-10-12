# utilities/graphml2bench.py
from __future__ import annotations
import argparse
import os
import re
from typing import Dict, Iterable, List, Tuple

import networkx as nx

PI, PO, AND = 0, 1, 2

EDGE_INV_KEYS = ("edge_type", "inv", "inverted", "is_inverted")
NODE_ID_KEYS = ("node_id", "name", "label", "id")
NODE_TYPE_KEYS = ("node_type", "type", "nodetype")


# ------------------ naming ------------------
def _sanitize_name(name: str | None) -> str:
    if not name:
        base = "n"
    else:
        base = str(name).strip() or "n"
    base = re.sub(r"[^0-9a-zA-Z_]", "_", base)
    if base[0].isdigit():
        base = "n_" + base
    return base


class NameAllocator:
    def __init__(self):
        self.used = set()

    def reserve(self, name: str | None) -> str:
        base = _sanitize_name(name)
        if base in self.used:
            return self.get(base)
        self.used.add(base)
        return base

    def get(self, base: str | None) -> str:
        base = _sanitize_name(base)
        cand = base
        k = 1
        while cand in self.used:
            cand = f"{base}_{k}"
            k += 1
        self.used.add(cand)
        return cand


# ------------------ io ------------------
def _load_graph(graphml_path: str) -> nx.MultiDiGraph:
    g0 = nx.read_graphml(graphml_path)
    if isinstance(g0, nx.MultiDiGraph):
        return g0
    if isinstance(g0, nx.DiGraph):
        g = nx.MultiDiGraph()
        g.add_nodes_from(g0.nodes(data=True))
        for u, v, d in g0.edges(data=True):
            g.add_edge(u, v, **d)
        return g
    # 其他图转有向多重图
    return g0.to_directed(as_view=False).to_multigraph()


def _get_type(nd: Dict) -> int:
    for k in NODE_TYPE_KEYS:
        if k in nd:
            try:
                return int(nd[k])
            except Exception:
                pass
    return AND


def _get_inv(ed: Dict) -> int:
    for k in EDGE_INV_KEYS:
        if k in ed:
            try:
                return 1 if int(ed[k]) == 1 else 0
            except Exception:
                pass
    return 0


# ------------------ sanitize ------------------
def _sanitize_graph(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    gg = nx.MultiDiGraph()
    for n, nd in g.nodes(data=True):
        nd = dict(nd)
        nd["node_type"] = _get_type(nd)
        gg.add_node(n, **nd)

    # 去重：同一对(u,v)保留最多一条 edge_type=0 与一条 edge_type=1
    seen = set()
    for u, v, ed in g.edges(data=True):
        if u not in gg or v not in gg:
            continue
        if u == v:
            continue
        t_u = gg.nodes[u]["node_type"]
        t_v = gg.nodes[v]["node_type"]
        if t_u == PO:
            continue  # PO 不应作为源
        if t_v == PI:
            continue  # 不应指向 PI
        inv = _get_inv(ed)
        key = (u, v, inv)
        if key in seen:
            continue
        seen.add(key)
        gg.add_edge(u, v, edge_type=inv)

    # 清掉 PI 的入边、PO 的出边（保险）
    drops = []
    for n, nd in gg.nodes(data=True):
        t = nd["node_type"]
        if t == PI:
            for p in list(gg.predecessors(n)):
                drops.append((p, n))
        if t == PO:
            for q in list(gg.successors(n)):
                drops.append((n, q))
    for u, v in drops:
        if gg.has_edge(u, v):
            for k in list(gg.get_edge_data(u, v).keys()):
                gg.remove_edge(u, v, key=k)
    return gg


def _break_cycles(g: nx.MultiDiGraph, max_iter: int = 100000) -> nx.MultiDiGraph:
    """迭代删除回边直到成为 DAG；仅删除一条平行边，兼容 MultiDiGraph 四元组返回。"""
    g = g.copy()
    iters = 0
    while True:
        try:
            _ = list(nx.topological_sort(g))
            return g
        except nx.NetworkXUnfeasible:
            pass

        iters += 1
        if iters > max_iter:
            return g  # 放弃修复

        try:
            cyc = nx.find_cycle(g, orientation="original")
        except nx.NetworkXNoCycle:
            return g

        drop_uv = None
        for e in cyc:
            # e 可能是 (u,v,orient) 或 (u,v,key,orient)
            if len(e) == 3:
                u, v, orient = e
                key = None
            elif len(e) == 4:
                u, v, key, orient = e
            else:
                continue
            if orient != "forward":
                continue
            t_u = g.nodes[u]["node_type"]
            t_v = g.nodes[v]["node_type"]
            if t_u != PO and t_v != PI:
                drop_uv = (u, v)
                break
        if drop_uv is None:
            e = cyc[0]
            if len(e) == 3:
                drop_uv = (e[0], e[1])
            else:
                drop_uv = (e[0], e[1])

        # 只删 (u,v) 的一条边（尽量温和，优先删 edge_type==0 的）
        if g.has_edge(*drop_uv):
            edata = g.get_edge_data(*drop_uv)
            if edata:
                pick_key = None
                for k, d in edata.items():
                    if int(d.get("edge_type", 0)) == 0:
                        pick_key = k
                        break
                if pick_key is None:
                    pick_key = next(iter(edata.keys()))
                g.remove_edge(drop_uv[0], drop_uv[1], key=pick_key)


def _topo_nodes(g: nx.MultiDiGraph) -> List[str]:
    try:
        return list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        g2 = _break_cycles(g)
        try:
            return list(nx.topological_sort(g2))
        except nx.NetworkXUnfeasible:
            return list(g.nodes())


# ------------------ emit bench ------------------
def graphml_to_bench_lines(
    graphml_path: str,
    repair_dag: bool = True,
    diversify_pad: bool = True,
) -> List[str]:
    g0 = _load_graph(graphml_path)
    g = _sanitize_graph(g0)
    if repair_dag:
        g = _break_cycles(g)

    order = _topo_nodes(g)
    alloc = NameAllocator()
    node2name: Dict[str, str] = {}
    inv_cache: Dict[str, str] = {}

    inputs: List[str] = []
    and_logic: List[str] = []
    inv_logic: List[str] = []
    outputs: List[str] = []

    # 先为所有节点分配名字
    for n in order:
        node2name[n] = alloc.reserve(n if any(k in g.nodes[n] for k in NODE_ID_KEYS) else f"n{n}")

    # 记录 PI，并将其加入可用源
    available: List[str] = []
    for n in order:
        if g.nodes[n]["node_type"] == PI:
            inputs.append(f"INPUT({node2name[n]})")
            available.append(n)

    rr = 0  # 轮转指针
    pair_hit: Dict[Tuple[str, str], int] = {}

    def lit_of(u: str, inv: int) -> str:
        nm = node2name[u]
        if inv == 0:
            return nm
        key = f"{nm}#inv"
        if key in inv_cache:
            return inv_cache[key]
        inv_nm = alloc.get(f"{nm}_not")
        inv_logic.append(f"{inv_nm} = NOT({nm})")
        inv_cache[key] = inv_nm
        return inv_nm

    def pick_sources(k: int) -> List[str]:
        nonlocal rr
        pool = list(dict.fromkeys(available))
        if not pool:
            return []
        chosen: List[str] = []
        tries = 0
        while len(chosen) < k and tries < len(pool) * 2:
            u = pool[rr % len(pool)]
            rr += 1
            tries += 1
            if u not in chosen:
                chosen.append(u)
        return chosen

    def diversify_pair(curr_srcs: List[str]) -> List[str]:
        if not diversify_pad:
            return curr_srcs[:2] if len(curr_srcs) >= 2 else (curr_srcs + curr_srcs)[:2]
        if len(curr_srcs) >= 2:
            return curr_srcs[:2]
        if len(curr_srcs) == 0:
            picks = pick_sources(2)
            return picks if len(picks) == 2 else (picks + picks)[:2]
        a = curr_srcs[0]
        t_a = g.nodes[a]["node_type"]
        pool = [u for u in available if u != a and g.nodes[u]["node_type"] != t_a]
        if not pool:
            pool = [u for u in available if u != a]
        if not pool:
            return [a, a]
        # 挑组合使用次数最少的
        best = None
        best_hit = 1 << 30
        a_nm = node2name[a]
        for u in pool:
            key = tuple(sorted((a_nm, node2name[u])))
            cnt = pair_hit.get(key, 0)
            if cnt < best_hit:
                best_hit = cnt
                best = u
        return [a, best if best is not None else pool[0]]

    # 逐节点生成
    for n in order:
        t = g.nodes[n]["node_type"]
        nm = node2name[n]
        preds = list(g.in_edges(n, keys=True, data=True))  # MultiDiGraph: (u,v,key,attr)
        in_lits: List[Tuple[str, int]] = [(u, _get_inv(ed)) for (u, _, _k, ed) in preds]

        if t == AND:
            if len(in_lits) > 2:
                # 取两个不同源
                uniq = []
                seen_src = set()
                for u, inv in in_lits:
                    if u in seen_src:
                        continue
                    uniq.append((u, inv))
                    seen_src.add(u)
                    if len(uniq) == 2:
                        break
                in_lits = uniq

            elif len(in_lits) < 2:
                # 兜底补齐
                curr = [u for u, _ in in_lits]
                picks = diversify_pair(curr)
                in_lits = [(picks[0], 0), (picks[1], 0)]

            a_src, a_inv = in_lits[0]
            b_src, b_inv = in_lits[1]
            a_lit = lit_of(a_src, a_inv)
            b_lit = lit_of(b_src, b_inv)
            and_logic.append(f"{nm} = AND({a_lit},{b_lit})")
            available.append(n)

            key = tuple(sorted((node2name[a_src], node2name[b_src])))
            pair_hit[key] = pair_hit.get(key, 0) + 1


        elif t == PO:

            # 优先使用图里真实的前驱

            if len(in_lits) > 0:

                u, inv = in_lits[0]

                lit = lit_of(u, inv)

                outputs.append(f"OUTPUT({lit})")

            else:

                # 没有前驱：优先选择“最近出现的 AND”，保证输出依赖内部逻辑，避免被 ABC 全部清扫

                and_pool = [u for u in reversed(available) if g.nodes[u]["node_type"] == AND]

                if and_pool:

                    u = and_pool[0];
                    inv = 0

                    lit = lit_of(u, inv)

                    outputs.append(f"OUTPUT({lit})")

                else:

                    # 再退化选任意可用源；还没有就给常量 0

                    any_pool = list(reversed(available))

                    if any_pool:

                        u = any_pool[0];
                        inv = 0

                        lit = lit_of(u, inv)

                        outputs.append(f"OUTPUT({lit})")

                    else:

                        z = alloc.get("const0")

                        and_logic.append(f"{z} = AND(0,0)")

                        outputs.append(f"OUTPUT({z})")


        else:
            # PI 已处理；非 PI/PO 的“其它”默认缓冲不造 AND，但加入 available 以便后续可作为源
            if t != PI:
                available.append(n)

    # 去重输出，避免同一信号多次 OUTPUT
    def _dedup_keep_order(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    outputs = _dedup_keep_order(outputs)

    lines: List[str] = []
    lines.extend(inputs)
    lines.extend(inv_logic)   # 先 NOT
    lines.extend(and_logic)   # 再 AND
    lines.extend(outputs)
    return lines


def convert_graphml_to_bench(graphml_path: str,
                             bench_path: str,
                             repair_dag: bool = True,
                             diversify_pad: bool = True) -> None:
    """
    Convert a GraphML netlist to .bench.
    Node types:
      0 -> PI (primary input)
      1 -> PO (primary output)
      2 -> AND (2-input AND; if fanin<2 and repair_dag=True, we will repair)
    Edges may carry inversion, but we ignore inversion for .bench structure here.

    Key changes vs. older versions:
    - Do NOT immediately emit OUTPUTs when visiting PO; cache them first.
    - After all ANDs are emitted, map each PO to the nearest AND by backtracking
      (skipping PO→PO链). If still找不到，就从 AND 池里轮询分散一个 AND。
      若图中没有 AND，才回退到一个 PI。
    - 对于 AND 的缺失前驱（0或1个），在 repair_dag=True 时自动补齐；
      优先使用已有 AND/PI 池，必要时将唯一前驱复制成两路。
    - diversify_pad=True 时，对回退/补齐使用 round-robin 以分散扇入锥。

    Args:
        graphml_path: 输入 GraphML 路径
        bench_path:   输出 .bench 路径
        repair_dag:   是否修补 fanin<2 的 AND
        diversify_pad:是否在需要兜底时做轮询分散
    """
    import os
    from pathlib import Path
    import networkx as nx

    # 读取图
    g = nx.read_graphml(graphml_path)

    # 读取 node_type 的辅助函数（兼容 'd0' 或 'node_type' 命名）
    def node_type(u):
        data = g.nodes[u]
        # GraphML常见：键是 'd0'；也兼容 'node_type'
        v = data.get('d0', data.get('node_type', 0))
        try:
            return int(v)
        except Exception:
            return 0

    PI, PO, AND = 0, 1, 2

    # 把 id 看成整数排序，确保稳定可复现
    def as_int(u):
        try:
            return int(u)
        except Exception:
            # 如果不是纯数字id，就退回字符串排序
            return int.from_bytes(str(u).encode('utf-8'), 'little') % (10**9 + 7)

    # 名称映射：n{ID}
    def nname(u):
        return f"n{as_int(u)}"

    # 收集节点
    pi_nodes  = sorted([u for u in g.nodes if node_type(u) == PI], key=as_int)
    and_nodes = sorted([u for u in g.nodes if node_type(u) == AND], key=as_int)
    po_nodes  = sorted([u for u in g.nodes if node_type(u) == PO], key=as_int)

    # 便捷：所有前驱（按数值id排序，稳定）
    def preds_sorted(u):
        return sorted(list(g.predecessors(u)), key=as_int)

    # 1) 先写 INPUT
    bench_lines = []
    for u in pi_nodes:
        bench_lines.append(f"INPUT({nname(u)})")

    # 维护一个信号池：可用于 AND 修补
    # 初始放入所有 PI；随后每生成一个 AND，就把该 AND 放进去
    signal_pool = pi_nodes.copy()

    # 记录已经生成的 AND，作为 PO 兜底“and_pool”
    built_and_nodes = []

    # round-robin 指针
    rr_idx = 0

    def rr_pick(candidates):
        nonlocal rr_idx
        if not candidates:
            return None
        if diversify_pad:
            u = candidates[rr_idx % len(candidates)]
            rr_idx += 1
            return u
        # 不分散时取“最近生成”的那个
        return candidates[-1]

    # 2) 生成 AND（若缺失前驱且 repair_dag=True 则修补）
    for u in and_nodes:
        fins = preds_sorted(u)

        if len(fins) >= 2:
            a, b = fins[0], fins[1]
        elif repair_dag:
            if len(fins) == 1:
                a = fins[0]
                # 第二路用 AND/PI 池兜底；没有就复制 a
                b = rr_pick(built_and_nodes or signal_pool) or a
            else:
                # 没有任何前驱：两路都从池子兜底；实在没有，就用第一个PI或自连
                a = rr_pick(built_and_nodes or signal_pool)
                b = rr_pick(built_and_nodes or signal_pool)
                if a is None and b is None:
                    # 图里既无 PI 也无 AND —— 极端情况：用自己自连也行（ABC会清理）
                    a = u
                    b = u
                if a is None:
                    a = b
                if b is None:
                    b = a
        else:
            # 不修补时，若不足2个前驱，跳过/最小处理：复制已有前驱或自连
            if len(fins) == 1:
                a, b = fins[0], fins[0]
            else:
                a, b = u, u  # 自连，占位

        bench_lines.append(f"{nname(u)} = AND({nname(a)},{nname(b)})")
        built_and_nodes.append(u)
        signal_pool.append(u)  # 新 AND 也进入可用信号池

    # 3) 统一处理 PO：先缓存（不立即输出），这里做“最近 AND”回溯
    # 回溯：从某个节点往前找最近的 AND（跳过 PO→PO）；找不到返回 None
    def nearest_and_backtrack(start):
        seen = set()
        cur = start
        while cur is not None and cur not in seen:
            seen.add(cur)
            nt = node_type(cur)
            if nt == AND:
                return cur
            ps = preds_sorted(cur)
            if not ps:
                return None
            # 简单取第一个前驱继续后退（也可按深度/拓扑优化，这里保持确定性）
            cur = ps[0]
        return None

    outputs = []
    and_rr_idx = 0
    for v in po_nodes:
        fins = preds_sorted(v)
        target = None

        if fins:
            # 优先：按 PO 的前驱向后回溯到最近的 AND
            cand = nearest_and_backtrack(fins[0])
            if cand is not None:
                target = cand

        # 次优：从 AND 池兜底（轮询分散）
        if target is None and built_and_nodes:
            if diversify_pad:
                target = built_and_nodes[and_rr_idx % len(built_and_nodes)]
                and_rr_idx += 1
            else:
                target = built_and_nodes[-1]

        # 最后：图里居然没有 AND —— 回退到某个 PI（优先最后一个）
        if target is None:
            if pi_nodes:
                target = pi_nodes[-1]
            else:
                # 极端：连 PI 都没有，就把该 PO 自连（ABC 会清）
                target = v

        outputs.append(f"OUTPUT({nname(target)})")

    # 4) 写文件
    Path(os.path.dirname(bench_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        f.write("\n".join(bench_lines + outputs) + "\n")



def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert GraphML (AIG) to .bench with DAG repair & diversified padding")
    p.add_argument("--graphml", required=True, help="Input GraphML file")
    p.add_argument("--out", required=False, help="Output .bench path (default: same basename)")
    p.add_argument("--no-repair", action="store_true", help="Do not repair DAG (skip cycle breaking)")
    p.add_argument("--no-diversify", action="store_true", help="Do not diversify when padding missing preds")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    in_path = args.graphml
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    out_path = args.out
    if out_path is None:
        base, _ = os.path.splitext(in_path)
        out_path = base + ".bench"
    elif os.path.isdir(out_path):
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(out_path, base + ".bench")

    convert_graphml_to_bench(
        in_path, out_path,
        repair_dag=(not args.no_repair),
        diversify_pad=(not args.no_diversify),
    )
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()



"""
python utilities/graphml2bench.py --graphml results/ISCAS85/graphml_fake/sample_0.graphml --out results/ISCAS85/bench_fake/sample_0.bench
"""