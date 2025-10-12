# utilities/scripts_check.py
import argparse
import os
import re
import networkx as nx


def check_aig_graphml(graphml_path: str) -> None:
    G = nx.read_graphml(graphml_path)
    print(f"[INFO] Checking GraphML: {graphml_path}")
    print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

    # AND 节点必须恰有 2 个前驱；PO 至少 1 个前驱
    and_bad = []
    po_bad = []
    iso = []

    for n, attr in G.nodes(data=True):
        t = int(attr.get("node_type", 2))
        if t == 2:  # AND
            preds = list(G.in_edges(n, data=True))  # DiGraph: (u,v,attr)
            if len(preds) != 2:
                and_bad.append((n, len(preds)))
        elif t == 1:  # PO
            preds = list(G.in_edges(n, data=True))
            if len(preds) < 1:
                po_bad.append(n)

    iso = [n for n, d in G.degree() if d == 0]

    if and_bad:
        for n, k in and_bad[:20]:
            print(f"[WARN] AND node {n} has {k} predecessors (expect 2)")
        if len(and_bad) > 20:
            print(f"... and {len(and_bad)-20} more AND nodes with wrong indegree")
    else:
        print("[OK] All AND nodes have exactly 2 predecessors.")

    if po_bad:
        for n in po_bad:
            print(f"[WARN] PO node {n} has 0 predecessors")
    else:
        print("[OK] All PO nodes have at least 1 predecessor.")

    if iso:
        print(f"[WARN] Isolated nodes: {iso[:30]}{' ...' if len(iso)>30 else ''}")
    else:
        print("[OK] No isolated nodes.")


def check_bench_file(bench_path: str) -> None:
    with open(bench_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # 统计 AND/OUTPUT（注意 AND 行一般是 “name = AND(a,b)”）
    and_count = sum(1 for ln in lines if " = AND(" in ln)
    out_count = sum(1 for ln in lines if ln.startswith("OUTPUT("))

    print(f"[INFO] .bench file: {bench_path}")
    print(f"AND gates (text count): {and_count}, OUTPUTs: {out_count}")

    # 简单抽样打印前 10 个 AND
    print_lines = [ln for ln in lines if " = AND(" in ln][:10]
    if print_lines:
        print("[SAMPLE AND]:")
        for ln in print_lines:
            print("  " + ln)


def main():
    ap = argparse.ArgumentParser(description="Quick checker for GraphML/.bench AIGs")
    ap.add_argument("--graphml", required=True, help="GraphML file to check")
    ap.add_argument("--bench", required=True, help=".bench file to check")
    args = ap.parse_args()

    if not os.path.exists(args.graphml):
        raise FileNotFoundError(args.graphml)
    if not os.path.exists(args.bench):
        raise FileNotFoundError(args.bench)

    check_aig_graphml(args.graphml)
    check_bench_file(args.bench)


if __name__ == "__main__":
    main()


"""
python utilities/scripts_check.py --pt_file results/ISCAS85/AIGfake/sample_0.pt --graphml_file results/ISCAS85/graphml_fake/sample_0.graphml --bench_file results/ISCAS85/bench_fake/sample_0.bench
"""
