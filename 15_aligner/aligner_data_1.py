"""
aligner_data_1.py — Convert trials.json + raw bio-signals into per-trial .npz files.
Extends aligner_data.py with selective modality processing and a merged-write NPZ strategy.

Arrays inside each output .npz:
    Modality "ppg":
        washout_finger_ppg   washout_glass_ppg
        video_finger_ppg     video_glass_ppg
    Modality "ultrasound":
        washout_ultrasound_nodiff   washout_ultrasound_diff
        video_ultrasound_nodiff     video_ultrasound_diff
    Modality "ear":
        washout_ear   washout_blink
        video_ear     video_blink

Usage:
    # Run all modalities (equivalent to original aligner_data.py)
    python aligner_data_1.py \\
        --trials_json  /path/to/trials.json \\
        --video_path   /path/to/face_recording.mov \\
        --participants /path/to/participants.json \\
        --pcm_dir      /path/to/pcm_dir \\
        --run_dir      /path/to/run_dir \\
        --output_dir   /path/to/output \\
        --modalities ppg ultrasound ear

    # Re-process only ultrasound (merges into existing NPZ, skips PPG/ear)
    python aligner_data_1.py \\
        --trials_json  /path/to/trials.json \\
        --participants /path/to/participants.json \\
        --pcm_dir      /path/to/pcm_dir \\
        --run_dir      /path/to/run_dir \\
        --output_dir   /path/to/output \\
        --modalities ultrasound

    # Re-process ear only (--video_path required; PPG timing is computed but not saved)
    python aligner_data_1.py \\
        --trials_json  /path/to/trials.json \\
        --video_path   /path/to/face_recording.mov \\
        --participants /path/to/participants.json \\
        --run_dir      /path/to/run_dir \\
        --output_dir   /path/to/output \\
        --modalities ear
"""

import argparse
import asyncio
import csv
import time
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED   = "\033[91m"
RESET = "\033[0m"

def red(s): return f"{RED}{s}{RESET}"

# ── Modality constants ─────────────────────────────────────────────────────────

VALID_MODALITIES = {"ppg", "ultrasound", "ear"}

_MODALITY_PROGRESS_KEYS: dict[str, list[str]] = {
    "ppg":        ["finger_ppg", "glass_ppg"],
    "ultrasound": ["ultrasound_washout_nodiff", "ultrasound_washout_diff",
                   "ultrasound_video_nodiff",   "ultrasound_video_diff"],
    "ear":        ["eye_tracking"],
}


def _modalities_all_done(entry: dict, modalities: set[str]) -> bool:
    """Return True if every progress key for the requested modalities is True."""
    for m in modalities:
        for key in _MODALITY_PROGRESS_KEYS.get(m, []):
            if not entry.get(key, False):
                return False
    return True


# ── PPG segmentation ──────────────────────────────────────────────────────────

_WASHOUT_FALLBACK_DURATION_S = 60.0


class SkipTrial(Exception):
    pass


def _parse_session_datetime(ppg_path: str) -> datetime:
    stem = Path(ppg_path).stem
    m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)$', stem)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
    m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$', stem)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    raise ValueError(f"Cannot parse session datetime from filename: {Path(ppg_path).name}")


def segment_ppg(ppg_path: str, video_duration: float) -> tuple[
    np.ndarray, np.ndarray,
    datetime, datetime,
    datetime, datetime,
]:
    """Extract washout and video signals from a single PPG file."""
    session_start = _parse_session_datetime(ppg_path)

    with open(ppg_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    def row_time(row) -> datetime:
        return session_start + timedelta(seconds=float(row["pc_time_s"]))

    def row_marker(row) -> str:
        return (row.get("marker") or "").strip()

    pc_wo_start = pc_vid_start = None
    for i, row in enumerate(rows):
        m = row_marker(row)
        if m == "VIDEO_START_PC_WASHOUT" and pc_wo_start is None:
            pc_wo_start = i
        elif m.startswith("VIDEO_START_PC_") and "WASHOUT" not in m and pc_vid_start is None:
            pc_vid_start = i

    wo_section_end    = pc_vid_start if pc_vid_start is not None else len(rows)
    vid_section_start = pc_vid_start if pc_vid_start is not None else len(rows)

    def first_marker(val: str, lo: int, hi: int) -> Optional[int]:
        for i in range(lo, min(hi, len(rows))):
            if row_marker(rows[i]) == val:
                return i
        return None

    ard_wo_start  = first_marker("VIDEO_START", (pc_wo_start  or 0) + 1, wo_section_end)
    _wo_end_lo    = (ard_wo_start or 0) + 1
    ard_wo_end    = first_marker("VIDEO_END",   _wo_end_lo,              wo_section_end)

    ard_vid_start = first_marker("VIDEO_START", vid_section_start + 1,   len(rows))
    _vid_end_lo   = (ard_vid_start or vid_section_start) + 1
    ard_vid_end   = first_marker("VIDEO_END",   _vid_end_lo,             len(rows))

    def resolve_window(name, ard_s, ard_e, dur):
        if ard_s is not None and ard_e is not None:
            return row_time(rows[ard_s]), row_time(rows[ard_e])
        if ard_s is not None:
            print(red(f"  [{name}] 缺少 Arduino VIDEO_END，用 VIDEO_START + {dur:.1f}s 推算"))
            t = row_time(rows[ard_s])
            return t, t + timedelta(seconds=dur)
        if ard_e is not None:
            print(red(f"  [{name}] 缺少 Arduino VIDEO_START，用 VIDEO_END - {dur:.1f}s 推算"))
            t = row_time(rows[ard_e])
            return t - timedelta(seconds=dur), t
        print(red(
            f"\n  [{name}] 错误: Arduino VIDEO_START 和 VIDEO_END 均缺失。\n"
            f"  文件: {ppg_path}\n"
        ))
        ans = input("  输入 Y 跳过这整条数据: ").strip().upper()
        if ans == "Y":
            raise SkipTrial(f"Arduino VIDEO_START 和 VIDEO_END 均缺失，文件: {Path(ppg_path).name}")
        raise ValueError(f"segment_ppg [{name}]: Arduino 标记缺失且用户拒绝跳过，文件: {ppg_path}")

    washout_start_dt, washout_end_dt = resolve_window(
        "washout", ard_wo_start, ard_wo_end, _WASHOUT_FALLBACK_DURATION_S,
    )
    video_start_dt, video_end_dt = resolve_window(
        "video", ard_vid_start, ard_vid_end, video_duration,
    )

    def extract_signal(t_start, t_end):
        out = []
        for row in rows:
            pc_s = row.get("pc_time_s", "").strip()
            ir   = row.get("ir_value",  "").strip()
            if not pc_s or not ir:
                continue
            try:
                t = session_start + timedelta(seconds=float(pc_s))
                v = float(ir)
            except ValueError:
                continue
            if t_start <= t <= t_end:
                out.append(v)
        return np.array(out, dtype=np.float32)

    return (
        extract_signal(washout_start_dt, washout_end_dt),
        extract_signal(video_start_dt,   video_end_dt),
        washout_start_dt, washout_end_dt,
        video_start_dt,   video_end_dt,
    )


from tools.ultrasound.preprocess1 import fmcw_pro
def convert_pcm_to_matrix(pcm_path: str, offset_seconds: float = 0.0):
    return fmcw_pro(pcm_path, offset=offset_seconds)


from tools.ear_extractor.ear_extractor import detect_blinks
async def run_eye_tracking(
    video_path, output_dir, output_csv,
    time_windows_sec=None, segment_cooldown_seconds=30,
):
    detect_blinks(
        video_path, output_dir,
        output_video=True,
        output_csv=output_csv,
        time_windows=time_windows_sec,
        segment_cooldown_seconds=segment_cooldown_seconds,
    )
    return output_csv


# ── VideoRecording ────────────────────────────────────────────────────────────

_FILENAME_TIME_FORMAT  = "%Y-%m-%d %H-%M-%S"
_FILENAME_TIME_EXAMPLE = "2026-05-31 14-46-52.mov"
_MAX_DRIFT_SECONDS     = 5.0


class VideoRecording:
    def __init__(self, video_path: str, blink_cache_dir: Optional[str] = None):
        self.path = Path(video_path)
        self.blink_cache_dir = Path(blink_cache_dir) if blink_cache_dir else self.path.parent
        self._blink_df: Optional[list[dict]] = None
        self._creation_time: Optional[datetime] = None

    def _parse_filename_time(self) -> datetime:
        stem = self.path.stem
        try:
            return datetime.strptime(stem, _FILENAME_TIME_FORMAT)
        except ValueError:
            raise ValueError(
                f"视频文件名 '{self.path.name}' 无法解析为时间。\n"
                f"期望格式: YYYY-MM-DD HH-MM-SS（例如 {_FILENAME_TIME_EXAMPLE}）\n"
                f"实际文件名 stem: '{stem}'"
            )

    def _get_video_duration(self) -> Optional[float]:
        try:
            import cv2
            cap   = cv2.VideoCapture(str(self.path))
            fps   = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and count > 0:
                return count / fps
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", str(self.path)],
                capture_output=True, text=True, timeout=15,
            )
            import json as _json
            data = _json.loads(result.stdout)
            for stream in data.get("streams", []):
                if "duration" in stream:
                    return float(stream["duration"])
        except Exception:
            pass
        return None

    @property
    def creation_time(self) -> datetime:
        if self._creation_time is not None:
            return self._creation_time

        filename_time = self._parse_filename_time()

        ctime = datetime.fromtimestamp(os.path.getctime(str(self.path)))
        if abs((ctime - filename_time).total_seconds()) <= _MAX_DRIFT_SECONDS:
            self._creation_time = ctime
            return self._creation_time

        mtime    = datetime.fromtimestamp(os.path.getmtime(str(self.path)))
        duration = self._get_video_duration()
        computed: Optional[datetime] = None
        if duration is not None:
            computed = mtime - timedelta(seconds=duration)
            if abs((computed - filename_time).total_seconds()) <= _MAX_DRIFT_SECONDS:
                self._creation_time = computed
                return self._creation_time

        fallback = filename_time.replace(microsecond=500_000)
        ctime_str    = ctime.strftime("%H:%M:%S.%f")
        computed_str = computed.strftime("%H:%M:%S.%f") if computed else "无法计算（视频时长未知）"
        print(red(
            f"\n  [VideoRecording] ⚠ 警告: '{self.path.name}'\n"
            f"    文件名时间   = {filename_time.strftime('%H:%M:%S')}\n"
            f"    ctime        = {ctime_str}  （差 {abs((ctime - filename_time).total_seconds()):.1f} 秒）\n"
            f"    mtime-时长   = {computed_str}"
            + (f"  （差 {abs((computed - filename_time).total_seconds()):.1f} 秒）" if computed else "")
            + f"\n    两者误差均超过 {_MAX_DRIFT_SECONDS} 秒，回退到文件名时间（秒以下设为 0.5）"
        ))
        self._creation_time = fallback
        return self._creation_time

    def _trial_cache_csv_path(self, trial_idx: int) -> Path:
        return self.blink_cache_dir / f"blink_trial{trial_idx}.csv"

    async def load_blink_for_trial(
        self,
        trial_idx: int,
        time_windows_sec: list,
        segment_cooldown_seconds: float = 30,
    ) -> None:
        cache = self._trial_cache_csv_path(trial_idx)
        if not cache.exists():
            print(f"  [VideoRecording] trial {trial_idx}: running eye-tracking …")
            await run_eye_tracking(
                str(self.path), str(self.blink_cache_dir), str(cache),
                time_windows_sec=time_windows_sec,
                segment_cooldown_seconds=segment_cooldown_seconds,
            )
        self._load_blink_csv(cache)

    def _load_blink_csv(self, csv_path: Path) -> None:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            self._blink_df = list(csv.DictReader(f))

    def slice_window(self, start: datetime, end: datetime) -> tuple[np.ndarray, np.ndarray]:
        if not self._blink_df:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int8)
        t0 = self.creation_time
        ear_vals, blink_vals = [], []
        for row in self._blink_df:
            try:
                ts = float(row["timestamp_sec"])
            except (ValueError, KeyError):
                continue
            frame_abs = t0 + timedelta(seconds=ts)
            if start <= frame_abs <= end:
                try:
                    ear_vals.append(float(row["smoothed_ear"]))
                    closed = row["eye_closed"].strip().lower()
                    blink_vals.append(1 if closed in ("true", "1") else 0)
                except (ValueError, KeyError):
                    continue
        return (
            np.array(ear_vals,   dtype=np.float32),
            np.array(blink_vals, dtype=np.int8),
        )


# ── TrialData ─────────────────────────────────────────────────────────────────

@dataclass
class TrialData:
    trial_index:    int
    participant_id: int

    # washout period
    washout_finger_ppg:        np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    washout_glass_ppg:         np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    washout_ultrasound_nodiff: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 0), dtype=np.float32))
    washout_ultrasound_diff:   np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 0), dtype=np.float32))
    washout_ear:               np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    washout_blink:             np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))

    # video period
    video_finger_ppg:          np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    video_glass_ppg:           np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    video_ultrasound_nodiff:   np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 0), dtype=np.float32))
    video_ultrasound_diff:     np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 0), dtype=np.float32))
    video_ear:                 np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    video_blink:               np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))

    def save_npz(self, output_dir: Path, modalities: set[str]) -> Path:
        """Write modality arrays into the NPZ, merging with any existing arrays."""
        out = output_dir / f"p{self.participant_id}_{self.trial_index}.npz"

        existing: dict[str, np.ndarray] = {}
        if out.exists():
            with np.load(out) as f:
                existing = {k: f[k] for k in f.files}

        new_arrays: dict[str, np.ndarray] = {}
        if "ppg" in modalities:
            new_arrays.update({
                "washout_finger_ppg": self.washout_finger_ppg,
                "washout_glass_ppg":  self.washout_glass_ppg,
                "video_finger_ppg":   self.video_finger_ppg,
                "video_glass_ppg":    self.video_glass_ppg,
            })
        if "ultrasound" in modalities:
            new_arrays.update({
                "washout_ultrasound_nodiff": self.washout_ultrasound_nodiff,
                "washout_ultrasound_diff":   self.washout_ultrasound_diff,
                "video_ultrasound_nodiff":   self.video_ultrasound_nodiff,
                "video_ultrasound_diff":     self.video_ultrasound_diff,
            })
        if "ear" in modalities:
            new_arrays.update({
                "washout_ear":   self.washout_ear,
                "washout_blink": self.washout_blink,
                "video_ear":     self.video_ear,
                "video_blink":   self.video_blink,
            })

        existing.update(new_arrays)
        np.savez(str(out), **existing)
        return out

    def describe(self) -> str:
        lines = [f"Trial {self.trial_index}  participant={self.participant_id}"]
        for attr in [
            "washout_finger_ppg", "washout_glass_ppg",
            "washout_ultrasound_nodiff", "washout_ultrasound_diff",
            "washout_ear", "washout_blink",
            "video_finger_ppg", "video_glass_ppg",
            "video_ultrasound_nodiff", "video_ultrasound_diff",
            "video_ear", "video_blink",
        ]:
            arr = getattr(self, attr)
            lines.append(f"  {attr:32s}: shape={arr.shape}  dtype={arr.dtype}")
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_participants(path: str) -> dict[str, int]:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return {k: int(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return {item["name"]: int(item["id"]) for item in raw}
    raise ValueError(f"Unsupported participants.json format in {path}")


def load_trials(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _default_progress_entry(trial_index: int) -> dict:
    return {
        "index":                     trial_index,
        "finger_ppg":                False,
        "glass_ppg":                 False,
        "ultrasound_washout_nodiff": False,
        "ultrasound_washout_diff":   False,
        "ultrasound_video_nodiff":   False,
        "ultrasound_video_diff":     False,
        "eye_tracking":              False,
        "errors":                    [],
        "npz":                       None,
        "skipped":                   False,
    }


def _migrate_progress_entry(entry: dict) -> dict:
    """Migrate old-format entries (ultrasound_washout/video booleans) to new split keys."""
    if "ultrasound_washout" in entry and "ultrasound_washout_nodiff" not in entry:
        entry.pop("ultrasound_washout", None)
        entry.pop("ultrasound_video",   None)
        entry.setdefault("ultrasound_washout_nodiff", False)
        entry.setdefault("ultrasound_washout_diff",   False)
        entry.setdefault("ultrasound_video_nodiff",   False)
        entry.setdefault("ultrasound_video_diff",     False)
    return entry


# ── Per-trial processor ───────────────────────────────────────────────────────

async def process_trial(
    trial: dict,
    participant_id: int,
    video: Optional["VideoRecording"],
    pcm_dir: Path,
    output_dir: Path,
    progress_entry: dict,
    modalities: set[str],
    segment_cooldown_seconds: float = 30,
) -> Optional[Path]:
    idx          = trial["index"]
    ori          = trial.get("ori_file_path", {})
    us_offset    = trial.get("ultrasound_offset", {"washout": 0, "video": 0})
    vid_duration = float(trial.get("video_duration", 0.0))

    td = TrialData(trial_index=idx, participant_id=participant_id)

    run_ppg = "ppg" in modalities
    run_ear = "ear" in modalities
    run_us  = "ultrasound" in modalities

    washout_start = washout_end = video_start = video_end = None

    # ── PPG (also runs for timing when ear is requested) ─────────────────────
    if run_ppg or run_ear:
        finger_path = ori.get("finger_ppg_path")
        glass_path  = ori.get("glass_ppg_path")

        try:
            if finger_path and os.path.exists(finger_path):
                (
                    td.washout_finger_ppg,
                    td.video_finger_ppg,
                    washout_start, washout_end,
                    video_start,   video_end,
                ) = segment_ppg(finger_path, vid_duration)
                if run_ppg:
                    progress_entry["finger_ppg"] = True
            else:
                msg = f"缺少 finger PPG: {finger_path}"
                print(f"  [trial {idx}] {msg}")
                progress_entry["errors"].append(msg)

            if glass_path and os.path.exists(glass_path):
                (
                    td.washout_glass_ppg,
                    td.video_glass_ppg,
                    *_,
                ) = segment_ppg(glass_path, vid_duration)
                if run_ppg:
                    progress_entry["glass_ppg"] = True
            else:
                msg = f"缺少 glass PPG: {glass_path}"
                print(f"  [trial {idx}] {msg}")
                progress_entry["errors"].append(msg)

        except SkipTrial as exc:
            print(f"  [trial {idx}] 用户选择跳过，已放弃。")
            progress_entry["skipped"] = True
            progress_entry["errors"].append(str(exc) or "PPG 解析时用户选择跳过")
            return None

    # ── Eye tracking ──────────────────────────────────────────────────────────
    if run_ear:
        if washout_start is not None and video_start is not None and video is not None:
            t0 = video.creation_time
            windows_sec = []
            for period, t_s, t_e in [
                ("washout", washout_start, washout_end),
                ("video",   video_start,   video_end),
            ]:
                s = (t_s - t0).total_seconds()
                e = (t_e - t0).total_seconds()
                if 0 <= s < e:
                    windows_sec.append((s, e, f"trial{idx}_{period}"))

            if windows_sec:
                await video.load_blink_for_trial(idx, windows_sec, segment_cooldown_seconds)
                td.washout_ear, td.washout_blink = video.slice_window(washout_start, washout_end)
                td.video_ear,   td.video_blink   = video.slice_window(video_start,   video_end)
                if len(td.washout_ear) > 0 or len(td.video_ear) > 0:
                    progress_entry["eye_tracking"] = True
                else:
                    msg = "眼動切片返回空陣列（視頻時間窗口無匹配幀）"
                    print(f"  [trial {idx}] {msg}")
                    progress_entry["errors"].append(msg)
            else:
                msg = "PPG 時間窗口在視頻範圍之外"
                print(f"  [trial {idx}] {msg}")
                progress_entry["errors"].append(msg)
        else:
            msg = "无 PPG 时间窗口或无视频，跳过眼动切片"
            print(f"  [trial {idx}] {msg}")
            progress_entry["errors"].append(msg)

    # ── Ultrasound ────────────────────────────────────────────────────────────
    if run_us:
        for attr, key, offset_key in [
            ("washout_ultrasound", "ultra_sound_washout", "washout"),
            ("video_ultrasound",   "ultra_sound_video",   "video"),
        ]:
            pcm_name = ori.get(key)
            offset   = float(us_offset.get(offset_key, 0))
            if pcm_name:
                pcm_full = pcm_dir / pcm_name
                if pcm_full.exists():
                    no_diff, diff = convert_pcm_to_matrix(str(pcm_full), offset_seconds=offset)
                    setattr(td, f"{attr}_nodiff", no_diff)
                    setattr(td, f"{attr}_diff",   diff)
                    progress_entry[f"ultrasound_{offset_key}_nodiff"] = True
                    progress_entry[f"ultrasound_{offset_key}_diff"]   = True
                else:
                    msg = f"PCM 文件不存在: {pcm_full}"
                    print(f"  [trial {idx}] {msg}")
                    progress_entry["errors"].append(msg)
            else:
                msg = f"缺少 {key}"
                print(f"  [trial {idx}] {msg}")
                progress_entry["errors"].append(msg)

    # ── Save (merge with existing NPZ) ────────────────────────────────────────
    out_path = td.save_npz(output_dir, modalities)
    progress_entry["npz"] = out_path.name
    print(f"  [trial {idx}] 已保存 → {out_path.name}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="aligner_data_1: trials.json → .npz (selective modalities)")
    parser.add_argument("--trials_json",  required=True)
    parser.add_argument("--participants", required=True)
    parser.add_argument("--pcm_dir",      default=None,
                        help="PCM 文件夹路径 (ultrasound 模态必填)")
    parser.add_argument("--run_dir",      required=True,
                        help="运行输出文件夹路径（存放 ear cache CSV 等）")
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--video_path",   default=None,
                        help="面部录像视频路径 (ear 模态必填)")
    parser.add_argument("--modalities", nargs="+", choices=["ppg", "ultrasound", "ear"],
                        default=["ppg", "ultrasound", "ear"],
                        help="要处理的模态，可多选（默认: 全部）")
    parser.add_argument("--segment_cooldown", type=float, default=30.0)
    args = parser.parse_args()

    modalities = set(args.modalities)

    if "ear" in modalities and not args.video_path:
        parser.error("--video_path 是 'ear' 模态的必填参数")
    if "ultrasound" in modalities and not args.pcm_dir:
        parser.error("--pcm_dir 是 'ultrasound' 模态的必填参数")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pcm_dir = Path(args.pcm_dir) if args.pcm_dir else None

    print(f"运行模态: {sorted(modalities)}")

    # ── Load metadata ─────────────────────────────────────────────────────────
    trials       = load_trials(args.trials_json)
    participants = load_participants(args.participants)

    # ── Progress: load or initialise ──────────────────────────────────────────
    progress_path = output_dir / "progress.json"
    progress: dict[str, dict] = {}

    if progress_path.exists():
        with open(progress_path) as f:
            existing = json.load(f)
        for trial in trials:
            key = str(trial["index"])
            entry = existing.get(key, _default_progress_entry(trial["index"]))
            progress[key] = _migrate_progress_entry(entry)
        done_n    = sum(1 for e in progress.values() if e.get("npz") and _modalities_all_done(e, modalities))
        skipped_n = sum(1 for e in progress.values() if e.get("skipped"))
        pending_n = len(trials) - done_n - skipped_n
        print(f"載入已有進度 ({progress_path.name}): "
              f"完成 {done_n}  跳過 {skipped_n}  待處理 {pending_n}")
    else:
        for trial in trials:
            progress[str(trial["index"])] = _default_progress_entry(trial["index"])
        print("新建進度記錄。")

    def _save_progress() -> None:
        progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False))

    _save_progress()

    # ── VideoRecording (only when ear modality is active) ─────────────────────
    video: Optional[VideoRecording] = None
    if "ear" in modalities and args.video_path:
        video = VideoRecording(args.video_path, Path(args.run_dir) / "ear")
        print(f"\n視頻創建時間: {video.creation_time}")

    # ── Process trials ────────────────────────────────────────────────────────
    print(f"\n共 {len(trials)} 個 trial，模态 {sorted(modalities)}，開始逐條處理 …\n")

    saved:   list[Path] = []
    skipped: list[int]  = []

    for trial in trials:
        idx         = trial["index"]
        entry       = progress[str(idx)]
        participant = trial.get("participant", "")
        trial_errors = trial.get("error", [])

        # ── Resume: already skipped ──────────────────────────────────────────
        if entry.get("skipped"):
            print(f"  [trial {idx}] 已標記跳過，略過")
            skipped.append(idx)
            continue

        # ── Resume: all requested modalities done ────────────────────────────
        if entry.get("npz") and _modalities_all_done(entry, modalities):
            print(f"  [trial {idx}] 所有请求的模态已完成 ({entry['npz']})，略過")
            saved.append(output_dir / entry["npz"])
            continue

        print(f"\n{'─'*50}")
        print(f"  [trial {idx}] 開始處理 … (剩余模态: "
              f"{[m for m in sorted(modalities) if not all(entry.get(k, False) for k in _MODALITY_PROGRESS_KEYS[m])]})")

        # ── Participant lookup ────────────────────────────────────────────────
        participant_id = participants.get(participant)
        if participant_id is None:
            msg = f"找不到 participant '{participant}' 的 id"
            print(f"  [trial {idx}] {msg}，跳過")
            entry["errors"].append(msg)
            entry["skipped"] = True
            skipped.append(idx)
            _save_progress()
            continue

        # ── trial.json errors ─────────────────────────────────────────────────
        if trial_errors:
            print(red(f"\n  [trial {idx}] 存在以下错误，默认跳过:"))
            for e in trial_errors:
                print(red(f"    • {e}"))
            ans = input("  仍要处理这条数据? (Y=处理 / 其他=跳过，默认跳过): ").strip().upper()
            if ans != "Y":
                print(f"  [trial {idx}] 已跳过。")
                entry["errors"].extend([f"[trial.json] {e}" for e in trial_errors])
                entry["skipped"] = True
                skipped.append(idx)
                _save_progress()
                continue

        # ── Process ───────────────────────────────────────────────────────────
        out = await process_trial(
            trial, participant_id, video, pcm_dir, output_dir, entry,
            modalities=modalities,
            segment_cooldown_seconds=args.segment_cooldown,
        )
        if out:
            saved.append(out)
        else:
            if not entry.get("skipped"):
                entry["skipped"] = True
            skipped.append(idx)

        _save_progress()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"完成！共處理 {len(saved)} 個 trial，跳過 {len(skipped)} 個")
    if skipped:
        print(f"跳過的 trial index: {skipped}")
    print(f"輸出目錄: {output_dir}")
    print(f"進度記錄: {progress_path}")

    if saved:
        example = saved[0]
        print(f"\n─── 示例: {example.name} ───")
        with np.load(example) as npz:
            for key in npz.files:
                arr = npz[key]
                print(f"  {key:32s}: shape={arr.shape}  dtype={arr.dtype}")


if __name__ == "__main__":
    asyncio.run(main())
