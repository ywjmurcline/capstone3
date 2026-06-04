"""
aligner_data.py — Convert trials.json + raw bio-signals into per-trial .npz files.

Each output file is named:  p{participant_id}_{trial_index}.npz
Arrays inside the npz:
    washout_finger_ppg  washout_glass_ppg  washout_ultrasound
    washout_ear         washout_blink
    video_finger_ppg    video_glass_ppg    video_ultrasound
    video_ear           video_blink

Usage:
    python aligner_data.py \\
        --trials_json  /path/to/trials.json \\
        --video_path   /path/to/face_recording.mov \\
        --participants /path/to/participants.json \\
        --pcm_dir      /path/to/pcm_dir \\
        --output_dir   /path/to/output
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

# ── PPG segmentation ──────────────────────────────────────────────────────────

_WASHOUT_FALLBACK_DURATION_S = 60.0


class SkipTrial(Exception):
    """Raised when the user requests to discard the current trial entirely."""
    pass


def _parse_session_datetime(ppg_path: str) -> datetime:
    """Parse the session start datetime from the PPG filename."""
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
    """
    Extract washout and video signals from a single PPG file.

    Timing is derived exclusively from the Arduino-echoed VIDEO_START / VIDEO_END
    rows (the ones that carry actual ir_value readings).  The PC-sent markers
    (VIDEO_START_PC_*, VIDEO_END_PC_WASHOUT, VIDEO_END_PC) are used only to
    locate which section of the file is washout vs video.

    Returns
    -------
    washout_signal : 1-D float array  (ir_value samples during washout)
    video_signal   : 1-D float array  (ir_value samples during video)
    washout_start  : absolute datetime for washout onset
    washout_end    : absolute datetime for washout offset
    video_start    : absolute datetime for video onset
    video_end      : absolute datetime for video offset

    Raises
    ------
    SkipTrial
        When both Arduino markers are absent for a segment and the user types Y.
    """
    session_start = _parse_session_datetime(ppg_path)

    with open(ppg_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    def row_time(row) -> datetime:
        return session_start + timedelta(seconds=float(row["pc_time_s"]))

    def row_marker(row) -> str:
        return (row.get("marker") or "").strip()

    # ── Locate PC section-boundary indices (structure only, not for timing) ───
    pc_wo_start = pc_vid_start = None
    for i, row in enumerate(rows):
        m = row_marker(row)
        if m == "VIDEO_START_PC_WASHOUT" and pc_wo_start is None:
            pc_wo_start = i
        elif m.startswith("VIDEO_START_PC_") and "WASHOUT" not in m and pc_vid_start is None:
            pc_vid_start = i

    # Rows before pc_vid_start belong to the washout section
    wo_section_end    = pc_vid_start if pc_vid_start is not None else len(rows)
    vid_section_start = pc_vid_start if pc_vid_start is not None else len(rows)

    # ── Locate Arduino echo marker indices ────────────────────────────────────
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

    # ── Resolve window boundaries from Arduino timing only ────────────────────
    def resolve_window(
        name: str,
        ard_s: Optional[int],
        ard_e: Optional[int],
        dur: float,
    ) -> tuple[datetime, datetime]:
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

        # Both Arduino markers absent
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

    # ── Extract ir_value samples within each window ───────────────────────────
    def extract_signal(t_start: datetime, t_end: datetime) -> np.ndarray:
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

from tools.ultrasound.preprocess import fmcw_pro
def convert_pcm_to_matrix(pcm_path: str, offset_seconds: float = 0.0) -> np.ndarray:
    """
    Load a raw PCM file and return it as a 2-D float32 array.

    Shape: (n_channels, n_samples) or (n_samples, n_channels) — decide with
    the ultrasound hardware spec.

    offset_seconds: skip this many seconds from the start of the file before
                    returning data (comes from trial["ultrasound_offset"]).

    TODO: fill in sample_rate, bit_depth, channels.
    """

    return fmcw_pro(pcm_path, offset=offset_seconds)

from tools.ear_extractor.ear_extractor import detect_blinks
async def run_eye_tracking(
    video_path: str,
    output_dir: str,
    output_csv: str,
    time_windows_sec: Optional[list] = None,
    segment_cooldown_seconds: int = 30,
) -> str:
    """
    Run the blink-detection pipeline on *video_path* and write results to *output_csv*.

    time_windows_sec: list of (start_sec, end_sec) tuples relative to the video start.
                      When provided, only those frames are decoded (saves CPU / heat).
    segment_cooldown_seconds: seconds to pause between segments to let the CPU cool down.
    """
    detect_blinks(
        video_path, output_dir,
        output_video=True,
        output_csv=output_csv,
        time_windows=time_windows_sec,
        segment_cooldown_seconds=segment_cooldown_seconds,
    )
    return output_csv


# ── VideoRecording ────────────────────────────────────────────────────────────

_FILENAME_TIME_FORMAT = "%Y-%m-%d %H-%M-%S"   # e.g. 2026-05-31 14-46-52
_FILENAME_TIME_EXAMPLE = "2026-05-31 14-46-52.mov"
_MAX_DRIFT_SECONDS = 5.0


class VideoRecording:
    """Wraps a face-recording video file and its per-frame eye data."""

    def __init__(self, video_path: str, blink_cache_dir: Optional[str] = None):
        self.path = Path(video_path)
        self.blink_cache_dir = Path(blink_cache_dir) if blink_cache_dir else self.path.parent
        self._blink_df: Optional[list[dict]] = None   # loaded lazily
        self._creation_time: Optional[datetime] = None  # cached after first call

    # ── Filename parsing ──────────────────────────────────────────────────────

    def _parse_filename_time(self) -> datetime:
        """
        Parse the recording start time from the filename stem.
        Raises ValueError with a helpful message if the format is wrong.
        """
        stem = self.path.stem
        try:
            return datetime.strptime(stem, _FILENAME_TIME_FORMAT)
        except ValueError:
            raise ValueError(
                f"视频文件名 '{self.path.name}' 无法解析为时间。\n"
                f"期望格式: YYYY-MM-DD HH-MM-SS（例如 {_FILENAME_TIME_EXAMPLE}）\n"
                f"实际文件名 stem: '{stem}'"
            )

    # ── Video duration ────────────────────────────────────────────────────────

    def _get_video_duration(self) -> Optional[float]:
        """Return video duration in seconds, trying cv2 then ffprobe."""
        # Try opencv
        try:
            import cv2
            cap = cv2.VideoCapture(str(self.path))
            fps   = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and count > 0:
                return count / fps
        except Exception:
            pass

        # Try ffprobe
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

    # ── creation_time (main entry point) ─────────────────────────────────────

    @property
    def creation_time(self) -> datetime:
        """
        Return the absolute datetime when the recording started.

        Priority:
        1. Filesystem ctime         — if within 5 s of filename time
        2. mtime − video duration   — if within 5 s of filename time
        3. Filename time + 0.5 s    — fallback; prints a red warning
        Raises ValueError if the filename is not a valid time.
        """
        if self._creation_time is not None:
            return self._creation_time

        filename_time = self._parse_filename_time()   # raises if bad filename

        # ── Priority 1: filesystem ctime ──────────────────────────────────────
        ctime = datetime.fromtimestamp(os.path.getctime(str(self.path)))
        if abs((ctime - filename_time).total_seconds()) <= _MAX_DRIFT_SECONDS:
            self._creation_time = ctime
            return self._creation_time

        # ── Priority 2: mtime − video_duration ───────────────────────────────
        mtime    = datetime.fromtimestamp(os.path.getmtime(str(self.path)))
        duration = self._get_video_duration()
        computed: Optional[datetime] = None
        if duration is not None:
            computed = mtime - timedelta(seconds=duration)
            if abs((computed - filename_time).total_seconds()) <= _MAX_DRIFT_SECONDS:
                self._creation_time = computed
                return self._creation_time

        # ── Fallback: filename time with 0.5 s sub-second ────────────────────
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

    def _cache_csv_path(self) -> Path:
        return self.blink_cache_dir / f"blink_{self.path.stem}.csv"

    def _trial_cache_csv_path(self, trial_idx: int) -> Path:
        return self.blink_cache_dir / f"blink_trial{trial_idx}.csv"

    async def load_blink_for_trial(
        self,
        trial_idx: int,
        time_windows_sec: list,
        segment_cooldown_seconds: float = 30,
    ) -> None:
        """
        Run eye-tracking for a single trial's windows (washout + video).
        Cache is per-trial: blink_trial{idx}.csv
        If the cache already exists, just load it (resume-safe).
        """
        cache = self._trial_cache_csv_path(trial_idx)
        if not cache.exists():
            print(f"  [VideoRecording] trial {trial_idx}: running eye-tracking …")
            await run_eye_tracking(
                str(self.path), str(self.blink_cache_dir), str(cache),
                time_windows_sec=time_windows_sec,
                segment_cooldown_seconds=segment_cooldown_seconds,
            )
        self._load_blink_csv(cache)

    async def ensure_blink_data(
        self,
        time_windows_sec: Optional[list] = None,
        segment_cooldown_seconds: int = 30,
    ) -> None:
        """Run eye-tracking if the cache CSV is absent; otherwise load it.

        time_windows_sec: list of (start_sec, end_sec) relative to video start.
                          Passed to detect_blinks so only those segments are decoded.
        segment_cooldown_seconds: seconds to pause between segments (CPU cooling).
        """
        cache = self._cache_csv_path()
        if not cache.exists():
            print(f"  [VideoRecording] Running eye-tracking on {self.path.name} …")
            await run_eye_tracking(
                str(self.path), str(self.blink_cache_dir), cache,
                time_windows_sec=time_windows_sec,
                segment_cooldown_seconds=segment_cooldown_seconds,
            )
        self._load_blink_csv(cache)

    def _load_blink_csv(self, csv_path: Path) -> None:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            self._blink_df = list(csv.DictReader(f))

    def slice_window(
        self,
        start: datetime,
        end: datetime,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (smoothed_ear, eye_closed_01) for frames inside [start, end].

        Frame absolute time = creation_time + timestamp_sec
        """
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
    trial_index:   int
    participant_id: int

    # washout period
    washout_finger_ppg:  np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    washout_glass_ppg:   np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    washout_ultrasound:  np.ndarray = field(default_factory=lambda: np.zeros((1, 0), dtype=np.float32))
    washout_ear:         np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    washout_blink:       np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))

    # video period
    video_finger_ppg:    np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    video_glass_ppg:     np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    video_ultrasound:    np.ndarray = field(default_factory=lambda: np.zeros((1, 0), dtype=np.float32))
    video_ear:           np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    video_blink:         np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))

    def save_npz(self, output_dir: Path) -> Path:
        out = output_dir / f"p{self.participant_id}_{self.trial_index}.npz"
        np.savez(
            out,
            washout_finger_ppg = self.washout_finger_ppg,
            washout_glass_ppg  = self.washout_glass_ppg,
            washout_ultrasound = self.washout_ultrasound,
            washout_ear        = self.washout_ear,
            washout_blink      = self.washout_blink,
            video_finger_ppg   = self.video_finger_ppg,
            video_glass_ppg    = self.video_glass_ppg,
            video_ultrasound   = self.video_ultrasound,
            video_ear          = self.video_ear,
            video_blink        = self.video_blink,
        )
        return out

    def describe(self) -> str:
        lines = [f"Trial {self.trial_index}  participant={self.participant_id}"]
        for attr in [
            "washout_finger_ppg", "washout_glass_ppg", "washout_ultrasound",
            "washout_ear", "washout_blink",
            "video_finger_ppg",   "video_glass_ppg",   "video_ultrasound",
            "video_ear",          "video_blink",
        ]:
            arr = getattr(self, attr)
            lines.append(f"  {attr:28s}: shape={arr.shape}  dtype={arr.dtype}")
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_participants(path: str) -> dict[str, int]:
    """
    Return {name: participant_id} from participants.json.
    Accepts two formats:
      - {"ywj": 1, "abc": 2}
      - [{"name": "ywj", "id": 1}, ...]
    """
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


def _add_timedelta_safe(dt: datetime, seconds: float) -> datetime:
    return dt + timedelta(seconds=seconds)


# ── Per-trial processor ───────────────────────────────────────────────────────

async def process_trial(
    trial: dict,
    participant_id: int,
    video: VideoRecording,
    pcm_dir: Path,
    output_dir: Path,
    progress_entry: dict,
    segment_cooldown_seconds: float = 30,
) -> Optional[Path]:
    idx          = trial["index"]
    ori          = trial.get("ori_file_path", {})
    us_offset    = trial.get("ultrasound_offset", {"washout": 0, "video": 0})
    vid_duration = float(trial.get("video_duration", 0.0))

    td = TrialData(trial_index=idx, participant_id=participant_id)

    # ── PPG ─────────────────────────────────────────────────────────────────
    finger_path = ori.get("finger_ppg_path")
    glass_path  = ori.get("glass_ppg_path")

    washout_start = washout_end = video_start = video_end = None

    try:
        if finger_path and os.path.exists(finger_path):
            (
                td.washout_finger_ppg,
                td.video_finger_ppg,
                washout_start, washout_end,
                video_start,   video_end,
            ) = segment_ppg(finger_path, vid_duration)
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
            progress_entry["glass_ppg"] = True
        else:
            msg = f"缺少 glass PPG: {glass_path}"
            print(f"  [trial {idx}] {msg}")
            progress_entry["errors"].append(msg)

    except SkipTrial as exc:
        print(f"  [trial {idx}] 用户选择跳过，已放弃。")
        progress_entry["skipped"] = True
        progress_entry["errors"].append(str(exc) or "PPG 解析時用戶選擇跳過")
        return None

    # ── Eye tracking (per-trial, only this trial's windows) ──────────────────
    if washout_start is not None and video_start is not None:
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
        msg = "无 PPG 时间窗口，跳过眼动切片"
        print(f"  [trial {idx}] {msg}")
        progress_entry["errors"].append(msg)

    # ── Ultrasound ───────────────────────────────────────────────────────────
    for attr, key, offset_key in [
        ("washout_ultrasound", "ultra_sound_washout", "washout"),
        ("video_ultrasound",   "ultra_sound_video",   "video"),
    ]:
        pcm_name = ori.get(key)
        offset   = float(us_offset.get(offset_key, 0))
        if pcm_name:
            pcm_full = pcm_dir / pcm_name
            if pcm_full.exists():
                setattr(td, attr, convert_pcm_to_matrix(str(pcm_full), offset_seconds=offset))
                progress_entry[f"ultrasound_{offset_key}"] = True
            else:
                msg = f"PCM 文件不存在: {pcm_full}"
                print(f"  [trial {idx}] {msg}")
                progress_entry["errors"].append(msg)
        else:
            msg = f"缺少 {key}"
            print(f"  [trial {idx}] {msg}")
            progress_entry["errors"].append(msg)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = td.save_npz(output_dir)
    progress_entry["npz"] = out_path.name
    print(f"  [trial {idx}] 已保存 → {out_path.name}")
    return out_path


# ── Window pre-collection ──────────────────────────────────────────────────────

def _collect_time_windows(
    trials: list[dict],
    video_creation_time: datetime,
    progress: dict,
) -> list[tuple[float, float, str]]:
    """
    Parse PPG files for all trials to find the washout/video time windows,
    returning them as (start_sec, end_sec, label) relative to video_creation_time.

    When the user chooses to skip a trial (SkipTrial) during this phase, the
    trial's progress entry is marked immediately so the main loop won't process
    it again.  Other errors are silently ignored here — the main loop will
    encounter and report them properly.
    """
    windows: list[tuple[float, float, str]] = []
    for trial in trials:
        trial_idx    = trial.get("index", "?")
        ori          = trial.get("ori_file_path", {})
        vid_duration = float(trial.get("video_duration", 0.0))
        finger_path  = ori.get("finger_ppg_path")
        entry        = progress.get(str(trial_idx), {})

        if not finger_path or not os.path.exists(finger_path):
            continue

        try:
            _, _, washout_start, washout_end, video_start, video_end = segment_ppg(
                finger_path, vid_duration,
            )
        except SkipTrial as exc:
            msg = str(exc) or "Arduino VIDEO_START 和 VIDEO_END 均缺失（用戶選擇跳過）"
            entry["skipped"] = True
            entry["errors"].append(msg)
            continue
        except Exception:
            continue

        for period, (t_start, t_end) in [
            ("washout", (washout_start, washout_end)),
            ("video",   (video_start,   video_end)),
        ]:
            s = (t_start - video_creation_time).total_seconds()
            e = (t_end   - video_creation_time).total_seconds()
            if 0 <= s < e:
                label = f"trial{trial_idx}_{period}"
                windows.append((s, e, label))

    return windows


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_progress_entry(trial_index: int) -> dict:
    return {
        "index":              trial_index,
        "finger_ppg":         False,
        "glass_ppg":          False,
        "ultrasound_washout": False,
        "ultrasound_video":   False,
        "eye_tracking":       False,
        "errors":             [],
        "npz":                None,
        "skipped":            False,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="aligner_data: trials.json → .npz")
    parser.add_argument("--trials_json",  required=True, help="trials.json 路径")
    parser.add_argument("--video_path",   required=True, help="面部录像视频路径 (.mov/.mp4)")
    parser.add_argument("--participants", required=True, help="participants.json 路径")
    parser.add_argument("--pcm_dir",      required=True, help="PCM 文件夹路径")
    parser.add_argument("--run_dir",      required=True, help="运行输出文件夹路径")
    parser.add_argument("--output_dir",   required=True, help="npz 输出文件夹")
    parser.add_argument("--segment_cooldown", type=float, default=30.0,
                        help="眼動片段之間的冷卻秒數（默認 30，設為 0 關閉）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pcm_dir = Path(args.pcm_dir)

    # ── Load metadata ────────────────────────────────────────────────────────
    trials       = load_trials(args.trials_json)
    participants = load_participants(args.participants)

    # ── Progress: load existing or initialise fresh ──────────────────────────
    progress_path = output_dir / "progress.json"
    progress: dict[str, dict] = {}

    if progress_path.exists():
        with open(progress_path) as f:
            existing = json.load(f)
        # Keep existing data; add skeleton for any new trials not yet recorded
        for trial in trials:
            key = str(trial["index"])
            progress[key] = existing.get(key, _default_progress_entry(trial["index"]))
        done_n    = sum(1 for e in progress.values() if e.get("npz"))
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

    # ── VideoRecording ────────────────────────────────────────────────────────
    video = VideoRecording(args.video_path, Path(args.run_dir) / "ear")
    print(f"\n視頻創建時間: {video.creation_time}")

    # ── Process trials one by one ─────────────────────────────────────────────
    print(f"\n共 {len(trials)} 個 trial，開始逐條處理 …\n")

    saved:   list[Path] = []
    skipped: list[int]  = []

    for trial in trials:
        idx          = trial["index"]
        entry        = progress[str(idx)]
        participant  = trial.get("participant", "")
        trial_errors = trial.get("error", [])

        # ── Resume: already done ─────────────────────────────────────────────
        if entry.get("npz"):
            print(f"  [trial {idx}] 已完成 ({entry['npz']})，略過")
            saved.append(output_dir / entry["npz"])
            continue

        # ── Resume: already skipped ──────────────────────────────────────────
        if entry.get("skipped"):
            print(f"  [trial {idx}] 已標記跳過，略過")
            skipped.append(idx)
            continue

        print(f"\n{'─'*50}")
        print(f"  [trial {idx}] 開始處理 …")

        # ── Participant lookup ───────────────────────────────────────────────
        participant_id = participants.get(participant)
        if participant_id is None:
            msg = f"找不到 participant '{participant}' 的 id"
            print(f"  [trial {idx}] {msg}，跳過")
            entry["errors"].append(msg)
            entry["skipped"] = True
            skipped.append(idx)
            _save_progress()
            continue

        # ── trial.json errors ────────────────────────────────────────────────
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

        # ── Process (PPG → eye-tracking → ultrasound → save) ─────────────────
        out = await process_trial(
            trial, participant_id, video, pcm_dir, output_dir, entry,
            segment_cooldown_seconds=args.segment_cooldown,
        )
        if out:
            saved.append(out)
        else:
            if not entry.get("skipped"):
                entry["skipped"] = True
            skipped.append(idx)

        _save_progress()

    # ── Summary ──────────────────────────────────────────────────────────────
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
                print(f"  {key:28s}: shape={arr.shape}  dtype={arr.dtype}")


if __name__ == "__main__":
    asyncio.run(main())
    # print(segment_ppg("/Users/lily/Documents/myApps/Capstone_Saveme/1_data_archive/534/bio_data/ppg/ppg_glass_emotion_video_task_18_ywj_2026-05-31 16:14:52.292866.csv", ))
