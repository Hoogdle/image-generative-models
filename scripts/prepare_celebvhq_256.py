#!/usr/bin/env python3
"""Download CelebV-HQ clips and build a StyleGAN-compatible 256x256 image zip."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image


def run(cmd, *, log_path=None):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("$ " + " ".join(map(str, cmd)) + "\n")
            f.write(proc.stdout + "\n")
    return proc.returncode, proc.stdout


def probe_size(video_path, ffprobe):
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ]
    code, out = run(cmd)
    if code != 0:
        raise RuntimeError(out.strip())
    streams = json.loads(out)["streams"]
    if not streams:
        raise RuntimeError("no video stream")
    return int(streams[0]["width"]), int(streams[0]["height"])


def crop_box(info, width, height):
    box = info["bbox"]
    top = max(float(box["top"]) - 0.02, 0.0)
    bottom = min(float(box["bottom"]) + 0.02, 1.0)
    left = max(float(box["left"]) - 0.02, 0.0)
    right = min(float(box["right"]) + 0.02, 1.0)

    top, bottom = round(top * height), round(bottom * height)
    left, right = round(left * width), round(right * width)
    h = bottom - top
    w = right - left
    half = max(1, min(h, w) // 2)
    cy = (top + bottom) / 2.0
    cx = (left + right) / 2.0

    top = int(round(cy - half))
    bottom = int(round(cy + half))
    left = int(round(cx - half))
    right = int(round(cx + half))

    if top < 0:
        bottom -= top
        top = 0
    if left < 0:
        right -= left
        left = 0
    if bottom > height:
        top -= bottom - height
        bottom = height
    if right > width:
        left -= right - width
        right = width

    top = max(top, 0)
    left = max(left, 0)
    return left, top, max(1, right - left), max(1, bottom - top)


def download_video(ytb_id, raw_dir, yt_dlp, log_path):
    for ext in ("mp4", "mkv", "webm"):
        candidate = raw_dir / f"{ytb_id}.{ext}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    output_tmpl = str(raw_dir / f"{ytb_id}.%(ext)s")
    cmd = [
        yt_dlp,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_tmpl,
        f"https://www.youtube.com/watch?v={ytb_id}",
    ]
    code, _ = run(cmd, log_path=log_path)
    if code != 0:
        return None
    for ext in ("mp4", "mkv", "webm"):
        candidate = raw_dir / f"{ytb_id}.{ext}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def extract_frames(clip_id, info, video_path, frames_dir, frames_per_clip, ffmpeg, ffprobe, log_path):
    width, height = probe_size(video_path, ffprobe)
    x, y, w, h = crop_box(info, width, height)
    start = float(info["duration"]["start_sec"])
    end = float(info["duration"]["end_sec"])
    duration = max(0.01, end - start)

    out_paths = []
    for idx in range(frames_per_clip):
        ts = start + duration * (idx + 0.5) / frames_per_clip
        out_path = frames_dir / f"{clip_id}_{idx:02d}.png"
        if out_path.exists() and out_path.stat().st_size > 0:
            out_paths.append(out_path)
            continue
        cmd = [
            ffmpeg,
            "-y",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", f"crop={w}:{h}:{x}:{y},scale=256:256:flags=lanczos",
            "-loglevel", "error",
            str(out_path),
        ]
        code, _ = run(cmd, log_path=log_path)
        if code == 0 and out_path.exists() and out_path.stat().st_size > 0:
            try:
                with Image.open(out_path) as im:
                    im.verify()
                out_paths.append(out_path)
            except Exception:
                out_path.unlink(missing_ok=True)
    return out_paths


def write_zip(frames_dir, zip_path):
    images = sorted(p for p in frames_dir.rglob("*.png") if p.is_file())
    if not images:
        raise RuntimeError(f"no PNG frames found in {frames_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, img_path in enumerate(images):
            arch = f"{idx // 1000:05d}/img{idx:08d}.png"
            zf.write(img_path, arch)
        zf.writestr("dataset.json", json.dumps({"labels": None}))
    tmp_path.replace(zip_path)
    return len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="/home/tykim/WorkSpace/CelebV-HQ/celebvhq_info.json")
    parser.add_argument("--workdir", default="/home/tykim/WorkSpace/datasets/celebv-hq")
    parser.add_argument("--zip", default="/home/tykim/WorkSpace/stylegan2_ffhq256_mmcelebahq/datasets/celebv-hq-256x256.zip")
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument("--max-clips", type=int, default=0, help="0 means all clips")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--yt-dlp", default="yt-dlp")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    raw_dir = workdir / "raw"
    frames_dir = workdir / "frames_256"
    log_path = workdir / "prepare_celebvhq_256.log"
    status_path = workdir / "prepare_status.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as f:
        clips = list(json.load(f)["clips"].items())
    if args.max_clips > 0:
        clips = clips[:args.max_clips]
    clips = clips[args.start_index:]

    done = set()
    if status_path.exists():
        with open(status_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add(row.get("clip_id"))

    fieldnames = ["time", "clip_id", "ytb_id", "status", "frames", "message"]
    write_header = not status_path.exists()
    with open(status_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        total = len(clips)
        for i, (clip_id, info) in enumerate(clips, start=1):
            if clip_id in done:
                continue
            ytb_id = info["ytb_id"]
            print(f"[{i}/{total}] {clip_id} ({ytb_id})", flush=True)
            try:
                if args.skip_download:
                    video_path = next((raw_dir / f"{ytb_id}.{ext}" for ext in ("mp4", "mkv", "webm") if (raw_dir / f"{ytb_id}.{ext}").exists()), None)
                else:
                    video_path = download_video(ytb_id, raw_dir, args.yt_dlp, log_path)
                if video_path is None:
                    raise RuntimeError("download failed or video missing")
                frames = extract_frames(clip_id, info, video_path, frames_dir, args.frames_per_clip, args.ffmpeg, args.ffprobe, log_path)
                if not frames:
                    raise RuntimeError("frame extraction failed")
                writer.writerow({"time": time.time(), "clip_id": clip_id, "ytb_id": ytb_id, "status": "ok", "frames": len(frames), "message": ""})
                f.flush()
            except Exception as exc:
                writer.writerow({"time": time.time(), "clip_id": clip_id, "ytb_id": ytb_id, "status": "fail", "frames": 0, "message": str(exc)[:500]})
                f.flush()

    if not args.no_zip:
        count = write_zip(frames_dir, Path(args.zip))
        print(f"Wrote {count} images to {args.zip}")


if __name__ == "__main__":
    main()
