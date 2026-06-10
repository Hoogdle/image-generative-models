#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

TICK_RE = re.compile(r"tick\s+(?P<tick>\d+)\s+kimg\s+(?P<kimg>[\d.]+).*?sec/kimg\s+(?P<sec>[\d.]+).*?augment\s+(?P<aug>[\d.]+)")


def latest_run(runs_dir: Path) -> Path:
    runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not runs:
        raise SystemExit(f"No run directories in {runs_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def read_last_tick(log_path: Path):
    if not log_path.exists():
        return None
    last = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = TICK_RE.search(line)
        if m:
            last = m.groupdict()
    return last


def read_fids(metric_path: Path):
    rows = []
    if not metric_path.exists():
        return rows
    for line in metric_path.read_text(errors="replace").splitlines():
        try:
            obj = json.loads(line)
            rows.append((obj["snapshot_pkl"], float(obj["results"]["fid50k_full"])))
        except Exception:
            pass
    return rows


def report(run_dir: Path) -> str:
    tick = read_last_tick(run_dir / "log.txt")
    fids = read_fids(run_dir / "metric-fid50k_full.jsonl")
    snaps = sorted(run_dir.glob("network-snapshot-*.pkl"))
    parts = [f"run={run_dir.name}"]
    if tick:
        parts.append(f"tick={tick['tick']} kimg={tick['kimg']} sec/kimg={tick['sec']} aug={tick['aug']}")
    else:
        parts.append("tick=not-yet")
    parts.append(f"snapshots={len(snaps)}")
    if fids:
        best = min(fids, key=lambda x: x[1])
        latest = fids[-1]
        parts.append(f"latest_fid={latest[1]:.4f}@{latest[0]}")
        parts.append(f"best_fid={best[1]:.4f}@{best[0]}")
    else:
        parts.append("fid=not-yet")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    root = Path(args.run_dir) if args.run_dir else latest_run(Path(args.runs_dir))
    while True:
        print(report(root), flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
