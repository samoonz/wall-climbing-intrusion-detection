"""
Demo canh bao xam nhap (wall-climbing intrusion alert) dung model DD-Net da train.

Chay tren video file hoac webcam, ve skeleton, hien thi canh bao khi phat hien
hanh vi treo tuong, va ghi log canh bao (console + file) kem am bao (Windows beep).

Vi du:
    python demo_intrusion_alert.py --source video_test.mp4
    python demo_intrusion_alert.py --source 0 --model best_with_attention.pth
    python demo_intrusion_alert.py --source video_test.mp4 --output out.mp4 --no-display
"""
import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
from scipy.ndimage import zoom as scipy_zoom
from scipy.signal import medfilt
from ultralytics import YOLO

from ddnet_model import Config, DDNetOriginal
from pose_tracking import track_people

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TRACKER_PATH = os.path.join(BASE_PATH, "pose_bytetrack.yaml")
TRACK_STATE_TTL_FRAMES = 30

FONT = cv2.FONT_HERSHEY_SIMPLEX
YOLO_IMGSZ = 480
INFER_STRIDE = 2
MAX_SKIP_PER_TICK = 10


def norm_scale(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.mean(x)


def zoom_sequence(p: np.ndarray, target_l: int, joints_num: int, joints_dim: int) -> np.ndarray:
    l = p.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:, m, n] = medfilt(p[:, m, n], 3)
            p_new[:, m, n] = scipy_zoom(p[:, m, n], target_l / l)[:target_l]
    return p_new


def get_cg(p: np.ndarray, config: Config, device: torch.device) -> np.ndarray:
    iu = np.triu_indices(config.joint_n, 1)
    frames = []
    for f in range(config.frame_l):
        pf = torch.from_numpy(p[f]).float().to(device)
        d_m = torch.cdist(pf, pf, p=2)
        frames.append(d_m[iu].cpu().numpy())
    return norm_scale(np.stack(frames))


def draw_bbox(frame: np.ndarray, bbox: np.ndarray, is_climb: bool, track_id: int) -> None:
    x1, y1, x2, y2 = bbox.astype(int)
    color = (0, 0, 255) if is_climb else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id} | {'CLIMB' if is_climb else 'NO CLIMB'}"
    cv2.putText(frame, label, (x1, max(20, y1 - 8)), FONT, 0.55, color, 2)


def draw_label(frame: np.ndarray, is_climb: bool, first_alert_time: float | None, fps: float) -> None:
    if is_climb:
        text = "CANH BAO: XAM NHAP - TREO TUONG!"
        color = (0, 0, 255)
        (tw, th), baseline = cv2.getTextSize(text, FONT, 1, 2)
        cv2.rectangle(frame, (40, 20), (60 + tw, 60 + th), color, 2)
        cv2.putText(frame, text, (50, 50), FONT, 1, color, 2)
    else:
        cv2.putText(frame, "No Climb", (50, 50), FONT, 1, (0, 255, 0), 2)

    if first_alert_time is not None:
        cv2.putText(frame, f"First alert at: {first_alert_time:.2f}s", (50, 100), FONT, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (50, 140), FONT, 0.8, (255, 0, 0), 2)


def play_alert_sound() -> None:
    if sys.platform != "win32":
        return
    try:
        import winsound
        winsound.Beep(1500, 300)
    except Exception:
        pass


def log_alert(log_path: str, source_name: str, elapsed: float) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | ALERT climb detected | source={source_name} | t={elapsed:.2f}s"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_demo(
    source: str,
    model_path: str,
    output_path: str | None,
    log_path: str,
    alert_cooldown: float,
    display: bool,
) -> None:
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    pose_model = YOLO("yolov8l-pose.pt")

    use_attention = "attention" in os.path.basename(model_path) and "no_attention" not in os.path.basename(model_path)
    model = DDNetOriginal(config, use_attention=use_attention).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Da nap model: {model_path} (attention={use_attention})", flush=True)

    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Khong the mo nguon video: {source}")

    source_name = "webcam" if isinstance(cap_source, int) else os.path.basename(source)
    is_file_source = isinstance(cap_source, str)
    track_buffers: dict[int, deque[np.ndarray]] = {}
    track_last_seen: dict[int, int] = {}
    track_climb: dict[int, bool] = {}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 1:
        video_fps = 30.0
    frame_interval = 1.0 / video_fps
    playback_clock_start = time.time()
    frame_idx = 0
    use_half = device.type == "cuda"

    out_writer = None
    if output_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, video_fps, (w, h))

    start_time = time.time()
    prev_time = start_time
    first_alert_time = None
    last_alert_time = -alert_cooldown
    is_climb = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            tracked_people = track_people(
                pose_model,
                frame,
                TRACKER_PATH,
                conf=0.25,
                imgsz=YOLO_IMGSZ,
                quantize=16 if use_half else None,
            )
            seen_ids = set()

            for tracked in tracked_people:
                track_id = tracked.track_id
                seen_ids.add(track_id)
                track_last_seen[track_id] = frame_idx
                pose_buffer = track_buffers.setdefault(
                    track_id, deque(maxlen=config.frame_l)
                )
                pose_buffer.append(tracked.keypoints_xyn)

                if len(pose_buffer) == config.frame_l and frame_idx % INFER_STRIDE == 0:
                    seq = zoom_sequence(
                        np.array(pose_buffer),
                        config.frame_l,
                        config.joint_n,
                        config.joint_d,
                    )
                    cg = get_cg(seq, config, device)

                    m_tensor = torch.from_numpy(cg).unsqueeze(0).float().to(device)
                    p_tensor = torch.from_numpy(np.expand_dims(seq, axis=0)).float().to(device)

                    with torch.no_grad():
                        outputs = model(m_tensor, p_tensor)
                        track_climb[track_id] = torch.argmax(outputs, dim=1).item() == 1

            for track_id in list(track_buffers):
                if frame_idx - track_last_seen.get(track_id, frame_idx) > TRACK_STATE_TTL_FRAMES:
                    track_buffers.pop(track_id, None)
                    track_last_seen.pop(track_id, None)
                    track_climb.pop(track_id, None)

            is_climb = any(track_climb.get(track_id, False) for track_id in seen_ids)

            elapsed = time.time() - start_time
            if is_climb:
                if first_alert_time is None:
                    first_alert_time = elapsed
                if elapsed - last_alert_time >= alert_cooldown:
                    log_alert(log_path, source_name, elapsed)
                    play_alert_sound()
                    last_alert_time = elapsed

            for tracked in tracked_people:
                draw_bbox(
                    frame,
                    tracked.bbox,
                    track_climb.get(tracked.track_id, False),
                    tracked.track_id,
                )

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-8)
            prev_time = curr_time
            draw_label(frame, is_climb, first_alert_time, fps)

            if display:
                cv2.imshow("Intrusion Alert Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if out_writer is not None:
                out_writer.write(frame)

            if is_file_source:
                expected_elapsed = frame_idx * frame_interval
                actual_elapsed = time.time() - playback_clock_start
                skips = 0
                while (actual_elapsed - expected_elapsed) > frame_interval and skips < MAX_SKIP_PER_TICK:
                    if not cap.grab():
                        break
                    frame_idx += 1
                    skips += 1
                    expected_elapsed = frame_idx * frame_interval
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if display:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo canh bao xam nhap treo tuong dung DD-Net")
    parser.add_argument("--source", default="0", help="Duong dan video hoac chi so webcam (mac dinh 0)")
    parser.add_argument("--model", default="best_with_attention.pth", help="Ten file checkpoint trong thu muc project/")
    parser.add_argument("--output", default=None, help="Duong dan luu video ket qua (tuy chon)")
    parser.add_argument("--log", default="alerts_log.txt", help="File log canh bao")
    parser.add_argument("--cooldown", type=float, default=5.0, help="So giay toi thieu giua 2 lan canh bao lien tiep")
    parser.add_argument("--no-display", action="store_true", help="Khong hien cua so xem truc tiep (chi luu file/log)")
    args = parser.parse_args()

    model_path = args.model if os.path.isabs(args.model) else os.path.join(BASE_PATH, args.model)
    log_path = args.log if os.path.isabs(args.log) else os.path.join(BASE_PATH, args.log)

    run_demo(
        source=args.source,
        model_path=model_path,
        output_path=args.output,
        log_path=log_path,
        alert_cooldown=args.cooldown,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
