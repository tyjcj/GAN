"""Utility to convert AIG GraphML files into .bench format."""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Iterable, List, Tuple

import networkx as nx

NODE_TYPE_PI = 0
NODE_TYPE_PO = 1
NODE_TYPE_AND = 2

EDGE_ATTR_INV_KEYS = ("edge_type", "inv", "inverted", "is_inverted")
NODE_ID_KEYS = ("node_id", "name", "label", "id")


def _sanitize_name(name: str) -> str:
    """Return a bench-safe identifier."""
    if name is None:
        base = "n"
    else:
        base = str(name)
    base = base.strip()
    if not base:
        base = "n"
    base = re.sub(r"[^0-9a-zA-Z_]", "_", base)
    if base[0].isdigit():
        base = f"n_{base}"
    return base


class NameAllocator:
    def __init__(self):
        self.used = set()

    def get(self, base: str) -> str:
        base = _sanitize_name(base)
        candidate = base
        idx = 1
        while candidate in self.used:
            candidate = f"{base}_{idx}"
            idx += 1
        self.used.add(candidate)
        return candidate

    def reserve(self, name: str) -> str:
        name = _sanitize_name(name)
        if name in self.used:
            return self.get(name)
        self.used.add(name)
        return name


def _load_graph(graphml_path: str) -> nx.MultiDiGraph:
    g_raw = nx.read_graphml(graphml_path)
    if not isinstance(g_raw, (nx.DiGraph, nx.MultiDiGraph)):
        g_raw = g_raw.to_directed()
    g = nx.MultiDiGraph()
    for n, data in g_raw.nodes(data=True):
        g.add_node(n, **data)
    for u, v, data in g_raw.edges(data=True):
        g.add_edge(u, v, **data)
    return g


def _detect_reverse(g: nx.MultiDiGraph) -> bool:
    pi_nodes = [n for n, d in g.nodes(data=True) if int(d.get("node_type", NODE_TYPE_AND)) == NODE_TYPE_PI]
    po_nodes = [n for n, d in g.nodes(data=True) if int(d.get("node_type", NODE_TYPE_AND)) == NODE_TYPE_PO]
    if not pi_nodes or not po_nodes:
        return False
    pi_in = sum(g.in_degree(n) for n in pi_nodes)
    pi_out = sum(g.out_degree(n) for n in pi_nodes)
    po_in = sum(g.in_degree(n) for n in po_nodes)
    po_out = sum(g.out_degree(n) for n in po_nodes)
    # Expect: PI in-degree ~ 0, PI out-degree >= 1
    #         PO out-degree ~ 0, PO in-degree >= 1
    reverse = False
    if pi_in > pi_out:
        reverse = True
    if po_out > po_in:
        reverse = True
    return reverse


def _ensure_forward(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if _detect_reverse(g):
        g = g.reverse(copy=True)
    return g


def _get_node_name(data: Dict[str, str], allocator: NameAllocator, fallback: str) -> str:
    for key in NODE_ID_KEYS:
        if key in data and data[key] is not None:
            return allocator.reserve(data[key])
    return allocator.reserve(fallback)


def _edge_inverters(data: Dict[str, str]) -> int:
    for key in EDGE_ATTR_INV_KEYS:
        if key in data:
            try:
                return int(data[key])
            except (TypeError, ValueError):
                pass
    return 0


def _topological_order(g: nx.MultiDiGraph) -> List[str]:
    simple = nx.DiGraph()
    simple.add_nodes_from(g.nodes(data=True))
    simple.add_edges_from([(u, v) for u, v, _ in g.edges(keys=True)])
    try:
        return list(nx.topological_sort(simple))
    except nx.NetworkXUnfeasible:
        # graph contains cycles; fall back to node iteration order
        return list(g.nodes())


def graphml_to_bench_lines(graphml_path: str) -> List[str]:
    g = _load_graph(graphml_path)
    g = _ensure_forward(g)
    order = _topological_order(g)

    allocator = NameAllocator()
    node_to_name: Dict[str, str] = {}
    inverter_cache: Dict[str, str] = {}
    lines_inputs: List[str] = []
    lines_outputs: List[Tuple[str, str]] = []
    lines_logic: List[str] = []

    def ensure_inverter(src_name: str) -> str:
        if src_name in inverter_cache:
            return inverter_cache[src_name]
        inv_name = allocator.get(f"{src_name}_not")
        lines_logic.append(f"{inv_name} = NOT({src_name})")
        inverter_cache[src_name] = inv_name
        return inv_name

    for node in order:
        attr = g.nodes[node]
        ntype = int(attr.get("node_type", NODE_TYPE_AND))
        assigned_name = _get_node_name(attr, allocator, fallback=f"n{node}")
        node_to_name[node] = assigned_name

        if ntype == NODE_TYPE_PI:
            lines_inputs.append(f"INPUT({assigned_name})")
            continue

        incoming = list(g.in_edges(node, keys=True, data=True))
        incoming.sort(key=lambda item: (node_to_name.get(item[0], str(item[0])), item[2]))

        if ntype == NODE_TYPE_AND:
            if len(incoming) < 2:
                # Attempt to pad missing predecessors by reusing the first available predecessor
                if incoming:
                    incoming = incoming + [incoming[0]] * (2 - len(incoming))
                else:
                    # floating AND -> drive with self-AND to create constant 0
                    dummy_src = assigned_name
                    lines_logic.append(f"{assigned_name} = AND({dummy_src},{dummy_src})")
                    continue
            literals: List[str] = []
            for src, _, _, data in incoming[:2]:
                src_name = node_to_name.get(src)
                if src_name is None:
                    # unseen source (due to cycle); allocate on demand
                    src_attr = g.nodes[src]
                    src_name = _get_node_name(src_attr, allocator, fallback=f"n{src}")
                    node_to_name[src] = src_name
                inv = _edge_inverters(data)
                if inv == 1:
                    literal = ensure_inverter(src_name)
                else:
                    literal = src_name
                literals.append(literal)
            lines_logic.append(f"{assigned_name} = AND({literals[0]},{literals[1]})")
        elif ntype == NODE_TYPE_PO:
            if not incoming:
                # Floating PO, attach to constant 0 by convention
                lines_logic.append(f"{assigned_name} = AND(0,0)")
                lines_outputs.append((assigned_name, assigned_name))
                continue
            src, _, _, data = incoming[0]
            src_name = node_to_name.get(src)
            if src_name is None:
                src_attr = g.nodes[src]
                src_name = _get_node_name(src_attr, allocator, fallback=f"n{src}")
                node_to_name[src] = src_name
            inv = _edge_inverters(data)
            literal = ensure_inverter(src_name) if inv == 1 else src_name
            lines_logic.append(f"{assigned_name} = AND({literal},{literal})")
            lines_outputs.append((assigned_name, assigned_name))
        else:
            # treat as buffer-like node with first predecessor
            if incoming:
                src, _, _, data = incoming[0]
                src_name = node_to_name.get(src, f"n{src}")
                inv = _edge_inverters(data)
                literal = ensure_inverter(src_name) if inv == 1 else src_name
                lines_logic.append(f"{assigned_name} = AND({literal},{literal})")
            else:
                lines_logic.append(f"{assigned_name} = AND({assigned_name},{assigned_name})")

    bench_lines: List[str] = []
    bench_lines.extend(lines_inputs)
    bench_lines.extend(f"OUTPUT({name})" for name, _ in lines_outputs)
    bench_lines.extend(lines_logic)
    return bench_lines


def convert_graphml_to_bench(graphml_path: str, bench_path: str) -> None:
    lines = graphml_to_bench_lines(graphml_path)
    os.makedirs(os.path.dirname(os.path.abspath(bench_path)), exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GraphML AIG into .bench format")
    parser.add_argument("--graphml", required=True, help="Input GraphML file")
    parser.add_argument("--out", required=False, help="Output .bench file (defaults to same name)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    graphml_path = args.graphml
    if not os.path.exists(graphml_path):
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")
    out_path = args.out
    if out_path is None:
        base, _ = os.path.splitext(graphml_path)
        out_path = base + ".bench"
    elif os.path.isdir(out_path):
        base_name = os.path.splitext(os.path.basename(graphml_path))[0]
        out_path = os.path.join(out_path, base_name + ".bench")
    convert_graphml_to_bench(graphml_path, out_path)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
