#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


DEFAULT_KEYS = {
    "fid": ("fid50k_full", "min"),
    "kid": ("kid50k_full", "min"),
    "is": ("is50k_mean", "max"),
}


def latest_run(runs_dir: Path) -> Path:
    runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not runs:
        raise SystemExit(f"No run directories in {runs_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_rows(run_dir: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for metric_file in sorted(run_dir.glob("metric-*.jsonl")):
        for line in metric_file.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            snap = obj.get("snapshot_pkl")
            if not snap:
                continue
            rows.setdefault(snap, {})
            for key, value in obj.get("results", {}).items():
                try:
                    rows[snap][key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def best_for_key(rows: dict[str, dict[str, float]], key: str, mode: str) -> tuple[str, float] | None:
    candidates = [(snap, values[key]) for snap, values in rows.items() if key in values]
    if not candidates:
        return None
    reverse = mode == "max"
    return sorted(candidates, key=lambda item: item[1], reverse=reverse)[0]


def rank_scores(rows: dict[str, dict[str, float]], keys: dict[str, tuple[str, str]]) -> dict[str, float]:
    scores = {snap: 0.0 for snap in rows}
    used = 0
    for key, mode in keys.values():
        candidates = [(snap, values[key]) for snap, values in rows.items() if key in values]
        if not candidates:
            continue
        reverse = mode == "max"
        ordered = sorted(candidates, key=lambda item: item[1], reverse=reverse)
        for rank, (snap, _value) in enumerate(ordered, start=1):
            scores[snap] += rank
        worst_rank = len(ordered) + 1
        missing = set(rows) - {snap for snap, _value in ordered}
        for snap in missing:
            scores[snap] += worst_rank
        used += 1
    if used == 0:
        return {}
    return scores


def safe_value(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "_")


def copy_snapshot(run_dir: Path, outdir: Path, tag: str, snap_name: str, suffix: str) -> Path:
    src = run_dir / snap_name
    if not src.exists():
        raise SystemExit(f"Snapshot missing: {src}")
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / f"{tag}-{suffix}-{snap_name}"
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--outdir", default="best_metrics")
    parser.add_argument("--tag", default="stylegan-best")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--toppr-key", default="pr50k3_full_recall",
                        help="Metric key to treat as TOPPR/PR best. Default: pr50k3_full_recall")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run(Path(args.runs_dir))
    rows = load_rows(run_dir)
    if not rows:
        raise SystemExit(f"No metric rows found in {run_dir}")

    keys = dict(DEFAULT_KEYS)
    keys["toppr"] = (args.toppr_key, "max")
    outdir = Path(args.outdir)

    print(f"run_dir={run_dir}")
    print("available_snapshots_with_metrics=" + str(len(rows)))

    summary = []
    for label, (key, mode) in keys.items():
        result = best_for_key(rows, key, mode)
        if result is None:
            print(f"best_{label}=missing key={key}")
            continue
        snap, value = result
        print(f"best_{label}={snap} {key}={value}")
        copied = ""
        if args.copy:
            dst = copy_snapshot(run_dir, outdir, args.tag, snap, f"best-{label}-{safe_value(value)}")
            copied = str(dst)
            print(f"copied_{label}={dst}")
        summary.append({"best": label, "snapshot": snap, "metric_key": key, "value": value, "copied": copied})

    scores = rank_scores(rows, keys)
    if scores:
        snap, score = sorted(scores.items(), key=lambda item: item[1])[0]
        print(f"best_overall={snap} rank_sum={score}")
        copied = ""
        if args.copy:
            dst = copy_snapshot(run_dir, outdir, args.tag, snap, f"best-overall-rank{safe_value(score)}")
            copied = str(dst)
            print(f"copied_overall={dst}")
        summary.append({"best": "overall", "snapshot": snap, "metric_key": "rank_sum", "value": score, "copied": copied})

    if args.copy:
        csv_path = outdir / f"{args.tag}-best-metrics-summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["best", "snapshot", "metric_key", "value", "copied"])
            writer.writeheader()
            writer.writerows(summary)
        print(f"summary_csv={csv_path}")


if __name__ == "__main__":
    main()
