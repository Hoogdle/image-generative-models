#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def latest_run(runs_dir: Path) -> Path:
    runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not runs:
        raise SystemExit(f"No run directories in {runs_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--outdir", default="best")
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run(Path(args.runs_dir))
    metric = run_dir / "metric-fid50k_full.jsonl"
    rows = []
    if metric.exists():
        for line in metric.read_text(errors="replace").splitlines():
            obj = json.loads(line)
            rows.append((float(obj["results"]["fid50k_full"]), obj["snapshot_pkl"]))
    if rows:
        fid, snap_name = min(rows, key=lambda x: x[0])
        reason = f"fid{fid:.4f}"
    else:
        snaps = sorted(run_dir.glob("network-snapshot-*.pkl"))
        if not snaps:
            raise SystemExit(f"No snapshots found in {run_dir}")
        snap_name = snaps[-1].name
        fid = None
        reason = "latest_no_metric"

    src = run_dir / snap_name
    print(f"best_source={src}")
    print(f"reason={reason}")
    if args.copy:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        dst = outdir / f"stylegan2-ffhq256-mmcelebahq-best-{reason}-{snap_name}"
        shutil.copy2(src, dst)
        print(f"copied={dst}")


if __name__ == "__main__":
    main()
