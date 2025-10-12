#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch process all .bench files in a directory and generate CSV for each.
Optimized version with parallel processing.
"""

import subprocess
import csv
import re
import os
import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from functools import lru_cache
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

# ---------------- Default config ----------------
ABC_BIN_DEFAULT = "/usr/local/bin/abc"
ABC_RC_DEFAULT = "/mnt/d/allproject/LSOformer/GraphMT-main/lsoformer/abc/abc.rc"
# BENCH_DIR_DEFAULT = "/mnt/d/allproject/LSOformer/GraphMT-main/lsoformer/datasets/EPFL"
BENCH_DIR_DEFAULT = "/mnt/d/allproject/LSOformer/GraphMT-main/lsoformer/datasets/EPFL_ctrl"
SCRIPTS_DIR_DEFAULT = "/mnt/d/allproject/LSOformer/GraphMT-main/lsoformer/data_files_epfl/referenceScripts"
OUTPUT_DIR_DEFAULT = "/mnt/d/allproject/LSOformer/GraphMT-main/lsoformer/data_files_epfl/dataset"
DEBUG_DIR_DEFAULT = "debug_abc_outputs"

SCRIPT_LINE_START = 3
SCRIPT_LINE_END = 22
TOTAL_STEPS = SCRIPT_LINE_END - SCRIPT_LINE_START + 1

SCRIPT_INDEX_RE = re.compile(r'(\d+)')
ALIAS_RE = re.compile(r"^alias\s+(\S+)\s+\"([^\"]+)\"")
LEV_PATTERN = re.compile(r"\blev(?:el)?s?\s*=\s*(\d+)", re.IGNORECASE)

# Thread-local storage for caching
_thread_local = threading.local()


# ---------------- Utility functions ----------------
def run_abc_batch(abc_bin: str, abc_rc: str, commands: List[str]) -> List[str]:
    """Run multiple abc commands in a single process to reduce startup overhead."""
    if not commands:
        return []

    # Combine all commands with semicolon separation
    batch_cmd = "; ".join(commands)
    full_cmd = f"source {abc_rc}; {batch_cmd}"

    try:
        cp = subprocess.run(
            [abc_bin, "-F", abc_rc, "-c", full_cmd],
            capture_output=True, text=True, timeout=300  # 5 minute timeout
        )
        output = (cp.stdout or "") + "\n" + (cp.stderr or "")
        # Split output by command boundaries (this is approximate)
        return [output] * len(commands)  # For now, return same output for all
    except subprocess.TimeoutExpired:
        return ["TIMEOUT"] * len(commands)
    except Exception as e:
        return [f"ERROR: {e}"] * len(commands)


def run_abc(abc_bin: str, abc_rc: str, cmd: str) -> subprocess.CompletedProcess:
    """Run abc with rc file and aliases active."""
    full_cmd = f"source {abc_rc}; {cmd}"
    try:
        cp = subprocess.run(
            [abc_bin, "-F", abc_rc, "-c", full_cmd],
            capture_output=True, text=True, timeout=60  # 1 minute timeout
        )
        return cp
    except subprocess.TimeoutExpired:
        # Create a mock CompletedProcess for timeout
        class MockProcess:
            def __init__(self):
                self.stdout = "TIMEOUT"
                self.stderr = "Command timed out"

        return MockProcess()


def parse_lev(out_text: str) -> Optional[int]:
    """Parse 'lev' from print_stats output."""
    if not out_text or "TIMEOUT" in out_text or "ERROR" in out_text:
        return None

    # First try the regex pattern
    m = LEV_PATTERN.findall(out_text)
    if m:
        return int(m[-1])

    # Fallback: look for lines containing 'lev'
    lines = out_text.splitlines()
    for ln in reversed(lines):
        if 'lev' in ln.lower():
            ints = re.findall(r"(\d+)", ln)
            if ints:
                return int(ints[-1])
    return None


def get_thread_cache():
    """Get thread-local cache."""
    if not hasattr(_thread_local, 'cache'):
        _thread_local.cache = {}
    return _thread_local.cache


def eval_prefix_get_lev_optimized(
        abc_bin: str, abc_rc: str, bench_path: str, prefix_key: str
) -> Tuple[Optional[int], str]:
    """Apply prefix steps to bench and return lev (with thread-local caching)."""
    cache = get_thread_cache()
    cache_key = f"{bench_path}:{prefix_key}"

    if cache_key in cache:
        return cache[cache_key]

    prefix_steps = prefix_key.split(';') if prefix_key else []
    bench_esc = bench_path.replace('"', '\\"')
    variants = [
        f'read_bench "{bench_esc}"; strash; ' + "; ".join(prefix_steps) + "; print_stats",
        f'read "{bench_esc}"; strash; ' + "; ".join(prefix_steps) + "; print_stats",
    ]
    last_out = ""
    for cmd in variants:
        try:
            cp = run_abc(abc_bin, abc_rc, cmd)
            out = (cp.stdout or "") + "\n" + (cp.stderr or "")
            last_out += "\n=== CMD: " + cmd + " ===\n" + out
            lev = parse_lev(out)
            if lev is not None and lev != 255:
                result = (lev, out)
                cache[cache_key] = result
                return result
        except Exception as e:
            last_out += f"\n=== ERROR: {cmd} - {e}\n"
            continue

    result = (None, last_out)
    cache[cache_key] = result
    return result


def read_script_lines(script_path: Path) -> List[str]:
    """Read lines 3–22 (inclusive) from script."""
    with script_path.open('r', encoding='utf-8') as f:
        lines = f.readlines()
    selected = lines[SCRIPT_LINE_START - 1:SCRIPT_LINE_END]
    return [ln.strip() for ln in selected if ln.strip()]


@lru_cache(maxsize=1)
def parse_alias_mapping(rc_path: str) -> Dict[str, str]:
    """Parse alias mapping from abc.rc file."""
    mapping = {}
    try:
        with open(rc_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("alias"):
                    m = ALIAS_RE.match(line)
                    if m:
                        alias, command = m.groups()
                        mapping[command] = alias
    except:
        pass
    return mapping


def build_custom_mapping() -> List[Tuple[re.Pattern, str]]:
    """Custom mapping rules for commands with parameters."""
    rules = [
        (re.compile(r"^balance$"), "b"),
        (re.compile(r"^rewrite\s*-z$"), "rwz"),
        (re.compile(r"^rewrite$"), "rw"),
        (re.compile(r"^refactor\s*-z$"), "rfz"),
        (re.compile(r"^refactor$"), "rf"),
        (re.compile(r"^resub\s*-z$"), "rsz"),
        (re.compile(r"^resub$"), "rs"),
        (re.compile(r"^rs\s+.*"), "rs"),
    ]
    return rules


def map_steps_to_alias(steps: List[str], alias_map: Dict[str, str], custom_rules: List[Tuple[re.Pattern, str]]) -> List[
    str]:
    """Replace each step with its alias if possible."""
    mapped = []
    for step in steps:
        replaced = None
        for pattern, alias in custom_rules:
            if pattern.match(step):
                replaced = alias
                break
        if replaced is None:
            replaced = alias_map.get(step, step)
        mapped.append(replaced)
    return mapped


def process_single_script(args_tuple):
    """Process a single script file - designed for parallel execution."""
    (idx, script_path, bench_path, abc_bin, abc_rc,
     alias_map, custom_rules, init_lev, debug_dir, bench_name) = args_tuple

    steps = read_script_lines(script_path)
    if len(steps) != TOTAL_STEPS:
        return None, f"Warning: Script {script_path.name} has {len(steps)} steps, expected {TOTAL_STEPS}"

    alias_steps = map_steps_to_alias(steps, alias_map, custom_rules)
    recipe_str = "; ".join(alias_steps)
    levels = []
    last_valid = init_lev if init_lev is not None else 0

    current_prefix = []
    for i, step in enumerate(steps, 1):
        current_prefix.append(step)
        prefix_key = ";".join(current_prefix)

        lev, raw_out = eval_prefix_get_lev_optimized(abc_bin, abc_rc, bench_path, prefix_key)

        if lev is None:
            if debug_dir:
                bench_debug_dir = Path(debug_dir) / bench_name
                os.makedirs(bench_debug_dir, exist_ok=True)
                dbgfile = bench_debug_dir / f"script{idx}_step{i}.log"
                dbgfile.write_text(raw_out)
            lev = last_valid
        else:
            last_valid = lev

        levels.append(lev)

    return [bench_name, recipe_str, TOTAL_STEPS] + levels, None


# ---------------- Core function ----------------
def process_single_bench(
        abc_bin: str,
        abc_rc: str,
        bench_path: str,
        scripts_dir: str,
        output_dir: str,
        debug_dir: str,
        max_workers: int = 4
):
    """Process a single bench file and generate CSV."""
    bench_name = Path(bench_path).stem  # Remove .bench extension
    out_csv = Path(output_dir) / f"{bench_name}.csv"
    bench_debug_dir = Path(debug_dir) / bench_name

    os.makedirs(bench_debug_dir, exist_ok=True)
    start_time = time.time()

    alias_map = parse_alias_mapping(abc_rc)
    custom_rules = build_custom_mapping()

    # Get initial level
    try:
        init_cp = run_abc(abc_bin, abc_rc, f'read_bench "{bench_path}"; strash; print_stats')
        init_text = (init_cp.stdout or "") + "\n" + (init_cp.stderr or "")
    except Exception as e:
        init_text = f"EXCEPTION: {e}"
    init_lev = parse_lev(init_text)
    if init_lev is None:
        Path(bench_debug_dir).joinpath(f"init_{bench_name}.txt").write_text(init_text)

    header = ["Design", "Recipe", "TotalSteps"] + [f"Level_{i + 1}" for i in range(TOTAL_STEPS)]
    rows = []

    # Collect .script files
    script_files = sorted(Path(scripts_dir).glob("*.script"),
                          key=lambda p: int(SCRIPT_INDEX_RE.search(p.name).group(1)))

    # Prepare arguments for parallel processing
    script_args = []
    for idx, script_path in enumerate(script_files):
        script_args.append((
            idx, script_path, bench_path, abc_bin, abc_rc,
            alias_map, custom_rules, init_lev, debug_dir, bench_name
        ))

    # Process scripts in parallel
    with ThreadPoolExecutor(max_workers=min(max_workers, len(script_files))) as executor:
        with tqdm(total=len(script_files), desc=f"Processing {bench_name}", unit="script", leave=False) as pbar:
            # Submit all tasks
            future_to_idx = {executor.submit(process_single_script, args): i
                             for i, args in enumerate(script_args)}

            # Collect results
            results = [None] * len(script_files)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result, warning = future.result()
                    if result:
                        results[idx] = result
                    if warning:
                        tqdm.write(warning)
                except Exception as e:
                    tqdm.write(f"Error processing script {idx}: {e}")

                pbar.update(1)
                pbar.set_postfix({"Completed": f"{sum(1 for r in results if r is not None)}/{len(script_files)}"})

    # Filter out None results and add to rows
    rows = [r for r in results if r is not None]

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    processing_time = time.time() - start_time
    print(f"Completed {bench_name} in {processing_time:.2f}s -> {out_csv}")


def process_bench_wrapper(args_tuple):
    """Wrapper function for multiprocessing."""
    (abc_bin, abc_rc, bench_path, scripts_dir, output_dir, debug_dir, max_workers) = args_tuple
    try:
        process_single_bench(abc_bin, abc_rc, str(bench_path), scripts_dir, output_dir, debug_dir, max_workers)
        return f"Success: {bench_path.name}"
    except Exception as e:
        return f"Error: {bench_path.name} - {e}"


# ---------------- Batch processing ----------------
def process_all_bench_files(
        abc_bin: str,
        abc_rc: str,
        bench_dir: str,
        scripts_dir: str,
        output_dir: str,
        debug_dir: str,
        max_processes: int = None,
        max_threads_per_process: int = 4
):
    """Process all .bench files in the directory with parallel processing."""
    bench_dir_path = Path(bench_dir)
    bench_files = list(bench_dir_path.glob("*.bench"))

    if not bench_files:
        print(f"No .bench files found in {bench_dir}")
        return

    if max_processes is None:
        max_processes = min(cpu_count(), len(bench_files))

    print(f"Found {len(bench_files)} .bench files to process")
    print(f"Using {max_processes} processes with {max_threads_per_process} threads each")

    total_start_time = time.time()
    processed_count = 0

    # Prepare arguments for multiprocessing
    process_args = [
        (abc_bin, abc_rc, bench_file, scripts_dir, output_dir, debug_dir, max_threads_per_process)
        for bench_file in bench_files
    ]

    # Process bench files in parallel
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        with tqdm(total=len(bench_files), desc="Processing bench files", unit="file") as pbar:
            # Submit all tasks
            future_to_file = {executor.submit(process_bench_wrapper, args): args[2]
                              for args in process_args}

            # Collect results
            for future in as_completed(future_to_file):
                bench_file = future_to_file[future]
                try:
                    result = future.result()
                    if result.startswith("Success"):
                        processed_count += 1
                    else:
                        tqdm.write(result)
                except Exception as e:
                    tqdm.write(f"Error processing {bench_file.name}: {e}")

                pbar.update(1)
                pbar.set_description(f"Processing bench files")
                pbar.set_postfix({"Processed": f"{processed_count}/{len(bench_files)}"})

    total_time = time.time() - total_start_time
    print(f"\n=== Batch processing completed ===")
    print(f"Processed: {processed_count}/{len(bench_files)} files")
    print(f"Total time: {total_time:.2f}s ({total_time / 60:.2f}min)")
    print(f"Average time per file: {total_time / len(bench_files):.2f}s")
    print(f"Output directory: {output_dir}")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Batch process all .bench files and generate CSVs (Optimized)")
    ap.add_argument("--abc", default=os.environ.get("ABC_BIN", ABC_BIN_DEFAULT), help="abc binary")
    ap.add_argument("--rc", default=os.environ.get("ABC_RC", ABC_RC_DEFAULT), help="abc.rc path")
    ap.add_argument("--bench-dir", default=BENCH_DIR_DEFAULT, help="directory containing .bench files")
    ap.add_argument("--scripts-dir", default=SCRIPTS_DIR_DEFAULT, help="directory with .script files")
    ap.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="output directory for CSV files")
    ap.add_argument("--debug-dir", default=DEBUG_DIR_DEFAULT, help="where to dump debug logs")
    ap.add_argument("--max-processes", type=int, default=8, help="max number of processes (default: CPU count)")
    ap.add_argument("--max-threads", type=int, default=6, help="max threads per process (default: 4)")
    args = ap.parse_args()

    # Create output and debug directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    if not Path(args.bench_dir).exists():
        raise SystemExit(f"Bench directory not found: {args.bench_dir}")
    if not Path(args.scripts_dir).exists():
        raise SystemExit(f"Scripts directory not found: {args.scripts_dir}")

    process_all_bench_files(
        abc_bin=args.abc,
        abc_rc=args.rc,
        bench_dir=args.bench_dir,
        scripts_dir=args.scripts_dir,
        output_dir=args.output_dir,
        debug_dir=args.debug_dir,
        max_processes=args.max_processes,
        max_threads_per_process=args.max_threads
    )


if __name__ == "__main__":
    main()
    print("Batch processing completed!")

"""


# Original command with parallelization:
python data_files/generate_csv_delay.py \
  --abc /usr/local/bin/abc \
  --rc data_files/abc/abc.rc \
  --bench-dir data_files/datasets/ISCAS85/bench  \
  --scripts-dir data_files/datasets/referenceScripts  \
  --output-dir data_files/datasets/ISCAS85/delay_csv \
  --debug-dir debug_logs \
  --max-processes 1 \
  --max-threads 4
"""

