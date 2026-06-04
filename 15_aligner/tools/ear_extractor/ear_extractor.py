import argparse
import subprocess
import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE_OUTLINE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_OUTLINE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(points):
    """
    EAR = (vertical_1 + vertical_2) / (2 * horizontal)
    points order: p1, p2, p3, p4, p5, p6
    """
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_landmark_points(face_landmarks, image_width, image_height, indices):
    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        points.append((int(lm.x * image_width), int(lm.y * image_height)))
    return points


def draw_eye_outline(frame, points, color):
    for p in points:
        cv2.circle(frame, p, 1, color, -1)


def _add_audio_to_segment(
    source_video: Path,
    silent_video: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    """
    Mux the audio track from source_video[start_sec..end_sec] into silent_video,
    replacing the file in-place.  Falls back to keeping the silent video if ffmpeg
    is unavailable or the source has no audio track.
    """
    tmp = silent_video.with_suffix(".tmp.mp4")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(silent_video),                   # annotated video (no audio)
            "-ss", str(start_sec), "-to", str(end_sec),
            "-i", str(source_video),                   # original recording (audio source)
            "-map", "0:v",                             # video from annotated
            "-map", "1:a",                             # audio from original
            "-c:v", "copy",
            "-c:a", "aac",
            str(tmp),
        ],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        silent_video.unlink()
        tmp.rename(silent_video)
    else:
        tmp.unlink(missing_ok=True)
        print(f"  [audio mux] 失敗，保留無聲視頻: {silent_video.name}")


def detect_blinks(
    input_video,
    output_dir=".",
    output_video=None,
    output_csv=None,
    ear_threshold=0.21,
    min_closed_frames=2,
    max_closed_frames=10,
    smoothing_window=3,
    max_video_seconds=300,
    time_windows=None,
    segment_cooldown_seconds=30,
):
    """
    Detect blinks and compute EAR for every analysed frame.

    output_video : None  → auto-name as <stem>_with_ear<ext>  (full-video mode only)
                   False → skip video output entirely           (full-video mode only)
    output_csv   : path for the per-frame CSV (auto-named if None)
    max_video_seconds : cap on annotated video length in full-video mode (default 5 min)

    time_windows : list of (start_sec, end_sec, label) tuples, relative to video start.
                   When provided, only those segments are decoded — the rest of the video
                   is skipped via seek so the CPU stays cool on long recordings.
                   Each segment produces its own annotated video:
                       <output_dir>/<stem>_<label>_with_ear.mp4   (with audio)
                   EAR smoothing state is reset at each segment boundary.
                   The 'output_video' parameter is ignored in this mode.
    """
    input_video = Path(input_video)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_csv is None:
        output_csv = output_dir / (input_video.stem + "_ear.csv")
    else:
        output_csv = output_dir / Path(output_csv).name

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")

    # ── Parse / prepare segments ──────────────────────────────────────────────
    if time_windows is not None:
        # Each element may be (start, end) or (start, end, label).
        segments = []
        for i, w in enumerate(time_windows):
            s, e   = w[0], w[1]
            label  = w[2] if len(w) > 2 else str(i)
            segments.append((s, e, label))
        segments.sort(key=lambda x: x[0])
    else:
        segments = None

    # ── Full-video mode: single writer ────────────────────────────────────────
    max_video_frames = int(fps * max_video_seconds) if fps else 0
    full_writer      = None
    full_video_path  = None
    if segments is None and output_video is not False:
        if output_video is None or output_video is True:
            full_video_path = output_dir / (input_video.stem + "_with_ear" + input_video.suffix)
        else:
            full_video_path = output_dir / Path(output_video).name
        full_writer = cv2.VideoWriter(str(full_video_path), fourcc, fps, (width, height))

    # ── Per-frame processing (shared between modes) ───────────────────────────
    rows          = []
    blink_counter = 0

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        def _process_one_frame(
            frame, frame_index, ear_history, closed_counter, in_closed_segment, seg_writer
        ):
            nonlocal blink_counter

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            face_detected  = False
            raw_ear        = None
            smoothed_ear   = None
            is_eye_closed  = False
            is_blink_frame = False
            blink_event    = False

            if result.multi_face_landmarks:
                face_detected  = True
                face_landmarks = result.multi_face_landmarks[0]

                left_eye_points  = get_landmark_points(face_landmarks, width, height, LEFT_EYE)
                right_eye_points = get_landmark_points(face_landmarks, width, height, RIGHT_EYE)
                left_ear         = eye_aspect_ratio(left_eye_points)
                right_ear        = eye_aspect_ratio(right_eye_points)
                raw_ear          = (left_ear + right_ear) / 2.0

                ear_history.append(raw_ear)
                if len(ear_history) > smoothing_window:
                    ear_history.pop(0)

                smoothed_ear  = float(np.mean(ear_history))
                is_eye_closed = smoothed_ear < ear_threshold

                if is_eye_closed:
                    closed_counter   += 1
                    in_closed_segment = True
                    is_blink_frame    = True
                else:
                    if in_closed_segment:
                        if min_closed_frames <= closed_counter <= max_closed_frames:
                            blink_counter += 1
                            blink_event    = True
                    closed_counter    = 0
                    in_closed_segment = False

                if seg_writer is not None:
                    left_outline  = get_landmark_points(face_landmarks, width, height, LEFT_EYE_OUTLINE)
                    right_outline = get_landmark_points(face_landmarks, width, height, RIGHT_EYE_OUTLINE)
                    eye_color = (0, 0, 255) if is_eye_closed else (0, 255, 0)
                    draw_eye_outline(frame, left_outline,  eye_color)
                    draw_eye_outline(frame, right_outline, eye_color)

            timestamp_sec = frame_index / fps if fps else 0.0

            if seg_writer is not None:
                status_text = (
                    "BLINK / CLOSED" if is_blink_frame
                    else ("NO FACE" if not face_detected else "OPEN")
                )
                cv2.putText(frame, f"Frame: {frame_index}",         (20,  35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Status: {status_text}",        (20,  70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if is_blink_frame else (0, 255, 0), 2)
                cv2.putText(frame, f"EAR: {smoothed_ear:.3f}" if smoothed_ear is not None else "EAR: N/A",
                                                                     (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Blinks: {blink_counter}",      (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                seg_writer.write(frame)

            rows.append({
                "frame_index":        frame_index,
                "timestamp_sec":      timestamp_sec,
                "face_detected":      face_detected,
                "ear":                raw_ear,
                "smoothed_ear":       smoothed_ear,
                "eye_closed":         is_eye_closed,
                "blink_frame":        is_blink_frame,
                "blink_event":        blink_event,
                "blink_count_so_far": blink_counter,
            })

            return ear_history, closed_counter, in_closed_segment

        # ── Full-video mode ───────────────────────────────────────────────────
        if segments is None:
            frame_index       = 0
            ear_history       = []
            closed_counter    = 0
            in_closed_segment = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                w = full_writer if frame_index < max_video_frames else None
                ear_history, closed_counter, in_closed_segment = _process_one_frame(
                    frame, frame_index, ear_history, closed_counter, in_closed_segment, w,
                )
                frame_index += 1
                if frame_index % 200 == 0:
                    print(f"Processed {frame_index}/{total_frames} frames...")

        # ── Windowed mode: one annotated video file per segment ───────────────
        else:
            n = len(segments)
            for seg_idx, (seg_start, seg_end, label) in enumerate(segments):
                start_frame = max(0, int(seg_start * fps))
                end_frame   = min(total_frames - 1, int(seg_end * fps))

                seg_video_path = output_dir / f"{input_video.stem}_{label}_with_ear.mp4"
                seg_writer     = cv2.VideoWriter(str(seg_video_path), fourcc, fps, (width, height))

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_index       = start_frame
                ear_history       = []
                closed_counter    = 0
                in_closed_segment = False
                seg_total         = end_frame - start_frame

                print(f"Segment {seg_idx + 1}/{n} '{label}': "
                      f"frames {start_frame}–{end_frame} ({seg_start:.1f}s–{seg_end:.1f}s)")

                while frame_index <= end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    ear_history, closed_counter, in_closed_segment = _process_one_frame(
                        frame, frame_index, ear_history, closed_counter, in_closed_segment, seg_writer,
                    )
                    frame_index += 1
                    done = frame_index - start_frame
                    if done % 200 == 0:
                        print(f"  {done}/{seg_total} frames in segment...")

                seg_writer.release()
                _add_audio_to_segment(input_video, seg_video_path, seg_start, seg_end)
                print(f"  → {seg_video_path.name}")

                if segment_cooldown_seconds > 0 and seg_idx < n - 1:
                    print(f"  [冷卻] 休息 {segment_cooldown_seconds} 秒後繼續下一段 …")
                    time.sleep(segment_cooldown_seconds)

    cap.release()
    if full_writer is not None:
        full_writer.release()

    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print("Done.")
    if segments is None and full_video_path is not None:
        written_sec = min(len(rows), max_video_frames) / fps if fps else 0
        print(f"Output video: {full_video_path}  ({written_sec:.0f}s written)")
    print(f"Output CSV:   {output_csv}")
    print(f"Total blinks: {blink_counter}")


if __name__ == "__main__":
    detect_blinks(
        input_video="/Users/lily/Downloads/2026-05-30 11-09-07.mov",
        output_dir="/Users/lily/Documents/myApps/Capstone_Saveme/15_aligner/tools/ear_extractor",
        output_video=True,
        output_csv="ear.csv",
        ear_threshold=0.15,
        min_closed_frames=2,
        max_closed_frames=15,
        smoothing_window=3,
        max_video_seconds=5,
    )
