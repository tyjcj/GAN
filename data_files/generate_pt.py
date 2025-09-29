 #!/usr/bin/env python3
"""
generate_pt.py

将 .bench 文件解析为 PyG Data 并保存为 .pt

输出 Data 包含字段:
    x: [N,2]  (node_type, num_inverted_predecessors)
    edge_index: [2,E]
    edge_attr: [E,1]  (0 buffer / 1 inverted)
    node_depth: [N]

"""
import os
import re
import argparse
from collections import defaultdict, deque
import torch
from torch_geometric.data import Data

def split_top_level_commas(s: str):
    parts = []
    cur = []
    depth = 0
    for ch in s:
        if ch == '(':
            depth += 1
            cur.append(ch)
        elif ch == ')':
            depth -= 1
            cur.append(ch)
        elif ch == ',' and depth == 0:
            part = ''.join(cur).strip()
            if part:
                parts.append(part)
            cur = []
        else:
            cur.append(ch)
    last = ''.join(cur).strip()
    if last:
        parts.append(last)
    return parts

def parse_expr(expr: str):
    """
    解析表达式，返回 [(name, inverted_flag), ...]
    支持:
      - !a
      - NOT(a) 或 NOT a
      - and(a,b) / AND(a,b)
      - infix a & b 等（抽取标识符）
    """
    expr = expr.strip()
    if expr == '':
        return []
    L = len(expr)

    def read_ident(j):
        start = j
        if j < L and (expr[j].isalpha() or expr[j] == '_' or expr[j].isdigit()):
            j += 1
            while j < L and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            return expr[start:j], j
        return None, j

    def skip_spaces(j):
        while j < L and expr[j].isspace():
            j += 1
        return j

    def parse_at(j):
        items = []
        while j < L:
            j = skip_spaces(j)
            if j >= L:
                break
            ch = expr[j]
            # '!' 前缀
            if ch == '!':
                j += 1
                j = skip_spaces(j)
                if j < L and expr[j] == '(':
                    # 拿括号内内容
                    k = j + 1
                    depth = 1
                    sub = []
                    while k < L and depth > 0:
                        if expr[k] == '(':
                            depth += 1; sub.append(expr[k])
                        elif expr[k] == ')':
                            depth -= 1
                            if depth > 0: sub.append(expr[k])
                        else:
                            sub.append(expr[k])
                        k += 1
                    inner = ''.join(sub)
                    args = split_top_level_commas(inner)
                    for a in args:
                        parsed = parse_expr(a)
                        for n, inv in parsed:
                            items.append((n, 1))
                    j = k
                    continue
                else:
                    name, j2 = read_ident(j)
                    if name:
                        items.append((name, 1))
                        j = j2
                        continue
                    else:
                        j += 1
                        continue
            # NOT keyword
            if expr[j:j+3].upper() == 'NOT':
                j2 = j + 3
                j2 = skip_spaces(j2)
                if j2 < L and expr[j2] == '(':
                    k = j2 + 1
                    depth = 1
                    sub = []
                    while k < L and depth > 0:
                        if expr[k] == '(':
                            depth += 1; sub.append(expr[k])
                        elif expr[k] == ')':
                            depth -= 1
                            if depth > 0: sub.append(expr[k])
                        else:
                            sub.append(expr[k])
                        k += 1
                    inner = ''.join(sub)
                    args = split_top_level_commas(inner)
                    for a in args:
                        parsed = parse_expr(a)
                        for n, inv in parsed:
                            items.append((n, 1))
                    j = k
                    continue
                else:
                    k = j2
                    name, k2 = read_ident(k)
                    if name:
                        items.append((name, 1))
                        j = k2
                        continue
                    else:
                        j = j2
                        continue
            # identifier or function call
            if ch.isalpha() or ch == '_' or ch.isdigit():
                name, j2 = read_ident(j)
                if name is None:
                    j += 1
                    continue
                j = j2
                j = skip_spaces(j)
                if j < L and expr[j] == '(':
                    k = j + 1
                    depth = 1
                    sub = []
                    while k < L and depth > 0:
                        if expr[k] == '(':
                            depth += 1; sub.append(expr[k])
                        elif expr[k] == ')':
                            depth -= 1
                            if depth > 0: sub.append(expr[k])
                        else:
                            sub.append(expr[k])
                        k += 1
                    inner = ''.join(sub)
                    args = split_top_level_commas(inner)
                    for a in args:
                        parsed = parse_expr(a)
                        items.extend(parsed)
                    j = k
                    continue
                else:
                    items.append((name, 0))
                    continue
            # else skip char
            j += 1
        return items, j

    parsed, _ = parse_at(0)
    # 合并同名变量，取 OR
    merged = {}
    for n, inv in parsed:
        if n in merged:
            merged[n] = merged[n] or bool(inv)
        else:
            merged[n] = bool(inv)
    return [(n, 1 if merged[n] else 0) for n in merged]

def parse_bench_file(path, verbose=False):
    node_types = {}      # name -> 0 PI,1 PO,2 AND/intermediate
    inverted_inputs = {} # name -> count
    edges = []           # (src_name, dst_name, inv_flag)
    nodes_order = []     # 保持首次出现顺序
    seen = set()
    lines = open(path, 'r', encoding='utf-8', errors='ignore').read().splitlines()

    def register_node(name):
        if name not in seen:
            seen.add(name)
            nodes_order.append(name)

    anon_output_counter = 0
    for ln_idx, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        # INPUT(name)
        m = re.match(r'INPUT\s*\(\s*([A-Za-z0-9_]+)\s*\)', line, flags=re.IGNORECASE)
        if m:
            name = m.group(1)
            node_types[name] = 0
            inverted_inputs.setdefault(name, 0)
            register_node(name)
            continue
        # OUTPUT(name_or_expr)
        m = re.match(r'OUTPUT\s*\(\s*(.*)\s*\)\s*$', line, flags=re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            idmatch = re.fullmatch(r'([A-Za-z0-9_]+)', inner)
            if idmatch:
                name = idmatch.group(1)
                node_types[name] = 1
                inverted_inputs.setdefault(name, 0)
                register_node(name)
            else:
                anon_name = f'__OUTPUT_ANON_{anon_output_counter}'
                anon_output_counter += 1
                node_types[anon_name] = 1
                inverted_inputs.setdefault(anon_name, 0)
                register_node(anon_name)
                parsed = parse_expr(inner)
                for src, inv in parsed:
                    edges.append((src, anon_name, inv))
                    register_node(src)
            continue
        # assignment: left = right
        if '=' in line:
            left, right = line.split('=', 1)
            left = left.strip()
            right = right.strip()
            node_types[left] = 2
            inverted_inputs.setdefault(left, 0)
            register_node(left)
            parsed = parse_expr(right)
            inv_count = sum(inv for _, inv in parsed)
            inverted_inputs[left] = inv_count
            for src, inv in parsed:
                edges.append((src, left, inv))
                register_node(src)
            continue
        # fallback parse expression for stray tokens
        parsed = parse_expr(line)
        for name, inv in parsed:
            if name not in node_types:
                node_types.setdefault(name, 0)
                inverted_inputs.setdefault(name, 0)
            register_node(name)

    # 确保所有在 edges 中出现但没出现在 nodes_order 的也加入（保持出现顺序）
    for u, v, _ in edges:
        if u not in seen:
            register_node(u)
        if v not in seen:
            register_node(v)
    # 也确保 node_types 的键存在于序列
    for n in list(node_types.keys()):
        if n not in seen:
            register_node(n)

    # 构造 node_to_idx 保持 nodes_order 顺序
    nodes = nodes_order
    node_to_idx = {n:i for i,n in enumerate(nodes)}

    # 构造 edge_index 和 edge_attr
    src_idx = []
    dst_idx = []
    edge_attrs = []
    for (u,v,inv) in edges:
        # 如果出现未知节点（非常罕见），跳过
        if u not in node_to_idx or v not in node_to_idx:
            if verbose:
                print(f"[WARN] skipping edge with unknown node: {u}->{v}")
            continue
        src_idx.append(node_to_idx[u])
        dst_idx.append(node_to_idx[v])
        edge_attrs.append([int(inv)])

    if len(src_idx) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,1), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    # x 特征矩阵
    x_list = []
    for n in nodes:
        nt = int(node_types.get(n, 0))
        inv_cnt = int(inverted_inputs.get(n, 0))
        x_list.append([nt, inv_cnt])
    x = torch.tensor(x_list, dtype=torch.float)

    # 计算 node_depth（拓扑）
    out_adj = defaultdict(list)
    indeg = {n:0 for n in nodes}
    for (u,v,inv) in edges:
        out_adj[u].append(v)
        indeg[v] = indeg.get(v, 0) + 1

    q = deque()
    depth = {n:0 for n in nodes}
    for n in nodes:
        if indeg.get(n,0) == 0:
            q.append(n); depth[n] = 0

    while q:
        u = q.popleft()
        for v in out_adj.get(u, []):
            depth[v] = max(depth.get(v, 0), depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    node_depth = torch.tensor([depth.get(n,0) for n in nodes], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_depth=node_depth)
    # 不在 data 上保存 node_names/name_to_idx（避免打印时展开大量内容）
    return data

def process_directory(bench_dir, out_dir, verbose=False):
    os.makedirs(out_dir, exist_ok=True)
    cnt = 0
    for root, dirs, files in os.walk(bench_dir):
        for fname in files:
            if not fname.endswith('.bench'):
                continue
            fpath = os.path.join(root, fname)
            try:
                data = parse_bench_file(fpath, verbose=verbose)
                outname = fname.replace('.bench', '.pt')
                outpath = os.path.join(out_dir, outname)
                torch.save(data, outpath)
                cnt += 1
                if verbose:
                    # 打印简洁信息（符合你需要的格式）
                    print(f"{fname.replace('.bench','')}:Data(x={tuple(data.x.shape)}, edge_index={tuple(data.edge_index.shape)}, edge_attr={tuple(data.edge_attr.shape)}, node_depth={tuple(data.node_depth.shape)})")
            except Exception as e:
                print(f"[ERR] Failed: {fpath} -> {e}")
    print(f"Saved {cnt} .pt files to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    process_directory(args.bench_dir, args.out_dir, verbose=args.verbose)
"""
python generate_pt.py --bench-dir ISCAS85/bench --out-dir --verbose
"""
