"""
Multi-modal biodata aligner.
Aligns psychopy, PPG (finger + glass), eye-tracking video, and ultrasound PCM
data for AI training.

Usage:
    python aligner.py \
        --bio_dir   /path/to/bio_data \
        --video_csv /path/to/all_videos_select_1.csv \
        --pcm_dir   /path/to/pcm_folder \
        --run_dir   /path/to/run_output
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── ANSI colours ─────────────────────────────────────────────────────────────
RED   = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def red(s):   return f"{RED}{s}{RESET}"
def green(s): return f"{GREEN}{s}{RESET}"

# ── Stub / placeholder functions ──────────────────────────────────────────────

def read_ppg_csv(path: str) -> tuple[list, list]:
    """Return (timestamps, ir_values) arrays from a PPG CSV file."""
    timestamps, ir_values = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ir = row.get("ir_value") or row.get("irValue") or ""
                ts = row.get("pc_time_s") or row.get("arduino_time_ms") or ""
                if ir.strip():
                    ir_values.append(float(ir))
                    timestamps.append(float(ts) if ts.strip() else 0.0)
            except (ValueError, KeyError):
                continue
    return timestamps, ir_values


async def extract_eye_tracking(video_path: str) -> list[dict]:
    """
    Placeholder: analyse video and return per-frame eye-tracking data.
    Each dict has keys:
        frame_index, timestamp_sec, face_detected, ear, smoothed_ear,
        eye_closed, blink_frame, blink_event, blink_count_so_far
    """
    # TODO: implement with mediapipe / dlib blink detection
    return []


def get_pcm_duration(path: str) -> float:
    """Return duration of a raw PCM file in seconds (placeholder)."""
    # TODO: read actual header / metadata once format is known
    size = os.path.getsize(path)
    sample_rate = 48000
    bit_depth   = 32
    channels    = 1
    return size / (sample_rate * (bit_depth // 8) * channels)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ask_skip(step_name: str) -> bool:
    """Return True if the user wants to skip this step."""
    ans = input(f"\n[{step_name}] 跳过这一步? (Y=跳过 / N=运行): ").strip().upper()
    return ans == "Y"


def wait_for_continue(prompt: str = "检查完毕后请输入 Y 继续, 其他键退出: ") -> None:
    ans = input(prompt).strip().upper()
    if ans != "Y":
        print("已停止。")
        sys.exit(0)


def load_run_json(run_dir: Path) -> dict:
    p = run_dir / "trials.json"
    if p.exists():
        with open(p) as f:
            raw = json.load(f)
        # Support both list and dict storage
        if isinstance(raw, list):
            return {str(item["index"]): item for item in raw}
        return raw
    return {}


def save_run_json(run_dir: Path, data: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = sorted(data.values(), key=lambda x: int(x["index"]))
    with open(run_dir / "trials.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  → 已写入 {run_dir / 'trials.json'}")


def parse_timestamp_from_filename(filename: str) -> Optional[str]:
    """Extract the datetime suffix from a bio-data filename.

    Pattern example: psychopy_emotion_video_task_0_ywj_2026-05-31 14:50:16.150781
    Returns the timestamp string, e.g. '2026-05-31 14:50:16.150781'
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)$", filename)
    return m.group(1) if m else None


def parse_video_id_from_filename(filename: str) -> Optional[str]:
    """Extract video_id integer from bio-data filename.

    Pattern: ..._task_{video_id}_{participant}_{timestamp}
    """
    m = re.search(r"_task_(\d+)_", filename)
    return m.group(1) if m else None

# ── Step 1 ────────────────────────────────────────────────────────────────────

async def step1_scan_psychopy(bio_dir: Path, run_dir: Path) -> None:
    """Scan psychopy folder; pick best file per video_id; write trials.json."""

    if ask_skip("Step 1 – 扫描 psychopy"):
        print("  跳过 Step 1。")
        return

    psychopy_dir = bio_dir / "psychopy"
    csv_files = sorted(psychopy_dir.glob("psychopy_*.csv"))

    # Group files by video_id
    by_id: dict[str, list[Path]] = {}
    for p in csv_files:
        vid = parse_video_id_from_filename(p.stem)
        if vid is not None:
            by_id.setdefault(vid, []).append(p)

    failures = []
    trials: dict[str, dict] = {}

    for vid, files in sorted(by_id.items(), key=lambda kv: int(kv[0])):
        # Keep only files with at least one data row (>1 line after stripping BOM)
        valid = []
        for fp in files:
            with open(fp, encoding="utf-8-sig") as f:
                rows = [l for l in f if l.strip()]
            if len(rows) >= 2:           # header + ≥1 data row
                valid.append(fp)

        if not valid:
            failures.append(vid)
            continue

        # Pick the last valid file (latest attempt)
        chosen = sorted(valid, key=lambda p: parse_timestamp_from_filename(p.stem) or "")[-1]
        trials[vid] = {
            "index": int(vid),
            "ori_file_path": {
                "psychopy_path": str(chosen)
            }
        }

    save_run_json(run_dir, trials)

    if failures:
        print(red(f"\n以下 video_id 没有有效的 psychopy 数据:"))
        for f in failures:
            print(red(f"  ✗ video_id={f}"))
    else:
        print(green("  所有 video_id 均有有效 psychopy 数据。"))

    wait_for_continue()


# ── Step 2 ────────────────────────────────────────────────────────────────────

async def step2_find_ppg(bio_dir: Path, run_dir: Path) -> None:
    """Match PPG files to each trial by timestamp suffix."""

    if ask_skip("Step 2 – 匹配 PPG 文件"):
        print("  跳过 Step 2。")
        return

    trials = load_run_json(run_dir)
    ppg_dir = bio_dir / "ppg"
    ppg_files = list(ppg_dir.glob("ppg_*.csv"))

    errors = []

    for vid, trial in trials.items():
        psychopy_path = trial["ori_file_path"].get("psychopy_path", "")
        ts = parse_timestamp_from_filename(Path(psychopy_path).stem)
        if not ts:
            errors.append(f"video_id={vid}: 无法从 psychopy 路径解析时间戳")
            continue

        finger = [p for p in ppg_files if "ppg_finger_" in p.name and ts in p.name]
        glass  = [p for p in ppg_files if "ppg_glass_"  in p.name and ts in p.name]

        if not finger:
            errors.append(f"video_id={vid}: 找不到 finger PPG (ts={ts})")
        else:
            trial["ori_file_path"]["finger_ppg_path"] = str(finger[0])

        if not glass:
            errors.append(f"video_id={vid}: 找不到 glass PPG (ts={ts})")
        else:
            trial["ori_file_path"]["glass_ppg_path"] = str(glass[0])

    save_run_json(run_dir, trials)

    if errors:
        print(red("\n以下条目存在 PPG 匹配问题:"))
        for e in errors:
            print(red(f"  ✗ {e}"))
    else:
        print(green("  所有条目 PPG 匹配成功。"))

    wait_for_continue()


# ── Step 3 ────────────────────────────────────────────────────────────────────

async def step3_check_ppg_quality(run_dir: Path) -> None:
    """Check PPG signals for flat segments (10+ consecutive equal values)."""

    if ask_skip("Step 3 – PPG 质量检查"):
        print("  跳过 Step 3。")
        return

    trials = load_run_json(run_dir)

    def has_flat_segment(ir_values: list, window: int = 10) -> bool:
        for i in range(len(ir_values) - window + 1):
            if len(set(ir_values[i:i + window])) == 1:
                return True
        return False

    flagged = []

    for vid, trial in trials.items():
        errors = trial.get("error", [])
        fp = trial["ori_file_path"].get("finger_ppg_path")
        gp = trial["ori_file_path"].get("glass_ppg_path")

        for key, path in [("finger_ppg_error", fp), ("glass_ppg_error", gp)]:
            if not path:
                continue
            _, ir = read_ppg_csv(path)
            if not ir:
                flagged.append(f"video_id={vid} {key}: 文件为空")
                if key not in errors:
                    errors.append(key)
            elif has_flat_segment(ir):
                flagged.append(f"video_id={vid} {key}: 发现 10 个连续相同值")
                if key not in errors:
                    errors.append(key)

        trial["error"] = errors

    save_run_json(run_dir, trials)

    if flagged:
        print(red("\nPPG 质量问题:"))
        for f in flagged:
            print(red(f"  ✗ {f}"))
    else:
        print(green("  PPG 质量检查通过。"))

    wait_for_continue()


# ── Step 4 ────────────────────────────────────────────────────────────────────

EMOTION_COLS = [
    "emotion_Amusing", "emotion_Anger", "emotion_Disgust",
    "emotion_Fear", "emotion_Sad", "emotion_Happy", "emotion_Neutral"
]

async def step4_extract_psychopy_labels(run_dir: Path) -> None:
    """Extract ground-truth labels from psychopy CSV into trials.json."""

    if ask_skip("Step 4 – 提取 psychopy 标签"):
        print("  跳过 Step 4。")
        return

    trials = load_run_json(run_dir)

    for vid, trial in trials.items():
        psychopy_path = trial["ori_file_path"].get("psychopy_path")
        if not psychopy_path:
            continue

        with open(psychopy_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]

        if not rows:
            continue

        row = rows[-1]   # already validated ≥1 row in step 1

        # Safely read a float or leave None
        def _f(key):
            v = row.get(key, "").strip()
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        emotion_tag = {}
        for col in EMOTION_COLS:
            v = _f(col)
            if v is not None:
                short = col.replace("emotion_", "").lower()
                emotion_tag[short] = v

        trial["video_file"]  = row.get("video_file", "").strip()
        trial["participant"] = row.get("participant", "").strip()
        trial["ground_truth"] = {
            "valence":  _f("valence"),
            "arousal":  _f("arousal"),
            "emotion_tag": emotion_tag,
        }

    save_run_json(run_dir, trials)
    print(green("  Step 4 完成。"))
    wait_for_continue()


# ── Step 5 ────────────────────────────────────────────────────────────────────

async def step5_enrich_from_video_csv(video_csv: Path, run_dir: Path) -> None:
    """Look up tag and duration_seconds from the stimulation materials CSV."""

    if ask_skip("Step 5 – 从视频素材 CSV 补充信息"):
        print("  跳过 Step 5。")
        return

    # Build lookup: video_name (str) → row
    video_meta: dict[str, dict] = {}
    with open(video_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row.get("video_name", "").strip()
            if name:
                video_meta[name] = row

    # Also index by absolute_path basename for flexible matching
    path_meta: dict[str, dict] = {}
    for row in video_meta.values():
        ap = row.get("absolute_path", "")
        if ap:
            path_meta[Path(ap).stem] = row
            path_meta[Path(ap).name] = row

    trials = load_run_json(run_dir)

    for vid, trial in trials.items():
        vf = trial.get("video_file", "")
        if not vf:
            continue

        # Try matching by basename, then stem, then video_name field
        candidates = [
            Path(vf).name,
            Path(vf).stem,
            vf,
        ]
        meta = None
        for c in candidates:
            if c in path_meta:
                meta = path_meta[c]
                break
            if c in video_meta:
                meta = video_meta[c]
                break

        if meta:
            trial["duration_seconds"] = float(meta.get("duration_seconds") or 0)
            gt = trial.setdefault("ground_truth", {})
            gt["emotion_tag_reference"] = meta.get("tag", "").strip()
        else:
            trial.setdefault("error", [])
            if "video_meta_not_found" not in trial["error"]:
                trial["error"].append("video_meta_not_found")

    save_run_json(run_dir, trials)
    print(green("  Step 5 完成。"))
    wait_for_continue()


# ── Step 6 ────────────────────────────────────────────────────────────────────

async def step6_scan_pcm(pcm_dir: Path, run_dir: Path) -> None:
    """Scan PCM folder, compute durations, write sorted CSV for user review."""

    if ask_skip("Step 6 – 扫描 PCM 文件"):
        print("  跳过 Step 6。")
        return

    pcm_files = sorted(
        [p for p in Path(pcm_dir).glob("*.pcm")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "pcm_durations.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "duration_seconds"])
        for p in pcm_files:
            dur = get_pcm_duration(str(p))
            writer.writerow([p.name, f"{dur:.4f}"])

    print(f"  → PCM 信息已写入 {out_csv}")
    print(  "  请检查并根据需要调整行数 (空行表示该 PCM 缺失)，完成后输入 Y 继续。")
    wait_for_continue()


# ── Step 7 ────────────────────────────────────────────────────────────────────

async def step7_assign_ultrasound(run_dir: Path) -> None:
    """Map PCM rows to trials (row//2 == video_id; even=washout, odd=video)."""

    if ask_skip("Step 7 – 分配超声 PCM"):
        print("  跳过 Step 7。")
        return

    pcm_csv = run_dir / "pcm_durations.csv"
    if not pcm_csv.exists():
        print(red(f"  找不到 {pcm_csv}，请先完成 Step 6。"))
        return

    # Read rows (skip header); row index from 0
    with open(pcm_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)   # skip header
        rows = list(reader)   # each row: [filename, duration_seconds]

    trials = load_run_json(run_dir)
    errors_added: list[str] = []

    for row_idx, row in enumerate(rows):
        video_id = row_idx // 2
        is_washout = (row_idx % 2 == 0)
        vid = str(video_id)

        if vid not in trials:
            continue

        trial = trials[vid]
        filename = row[0].strip() if row else ""

        key = "ultra_sound_washout" if is_washout else "ultra_sound_video"
        if filename:
            trial["ori_file_path"][key] = filename
        else:
            trial["ori_file_path"][key] = None

    # Validate: flag trials missing either ultrasound file
    for vid, trial in trials.items():
        errors = trial.setdefault("error", [])
        fp = trial["ori_file_path"]
        if not fp.get("ultra_sound_washout"):
            if "ultra_sound_washout_missing" not in errors:
                errors.append("ultra_sound_washout_missing")
                errors_added.append(f"video_id={vid}: ultra_sound_washout 缺失")
        if not fp.get("ultra_sound_video"):
            if "ultra_sound_video_missing" not in errors:
                errors.append("ultra_sound_video_missing")
                errors_added.append(f"video_id={vid}: ultra_sound_video 缺失")

    save_run_json(run_dir, trials)

    if errors_added:
        print(red("\n以下条目超声文件缺失:"))
        for e in errors_added:
            print(red(f"  ✗ {e}"))
    else:
        print(green("  所有条目超声文件分配完成。"))

    wait_for_continue()


# ── Step 8 ────────────────────────────────────────────────────────────────────

async def step8_set_ultrasound_offset(run_dir: Path) -> None:
    """Add default ultrasound_offset fields; let user edit before continuing."""

    if ask_skip("Step 8 – 设置超声 offset"):
        print("  跳过 Step 8。")
        return

    trials = load_run_json(run_dir)

    for trial in trials.values():
        trial.setdefault("ultrasound_offset", {"washout": 0, "video": 0})

    save_run_json(run_dir, trials)

    print("  默认 offset 均为 0（秒）。")
    print("  如需调整，请现在打开 trials.json 修改对应条目的")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Bio-data aligner")
    parser.add_argument("--bio_dir",   required=True, help="biodata 文件夹路径")
    parser.add_argument("--video_csv", required=True, help="视频素材描述 CSV 路径")
    parser.add_argument("--pcm_dir",   required=True, help="PCM 文件夹路径")
    parser.add_argument("--run_dir",   required=True, help="运行输出文件夹路径")
    args = parser.parse_args()

    bio_dir   = Path(args.bio_dir)
    video_csv = Path(args.video_csv)
    pcm_dir   = Path(args.pcm_dir)
    run_dir   = Path(args.run_dir)

    print("\n===== Bio-data Aligner =====\n")

    await step1_scan_psychopy(bio_dir, run_dir)
    await step2_find_ppg(bio_dir, run_dir)
    await step3_check_ppg_quality(run_dir)
    await step4_extract_psychopy_labels(run_dir)
    await step5_enrich_from_video_csv(video_csv, run_dir)
    await step6_scan_pcm(pcm_dir, run_dir)
    await step7_assign_ultrasound(run_dir)
    await step8_set_ultrasound_offset(run_dir)

    print(green("\n全部步骤完成！最终结果在 trials.json"))


if __name__ == "__main__":
    asyncio.run(main())
