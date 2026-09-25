"""
Dashboard web giam sat NHIEU camera/video cung luc, canh bao xam nhap treo tuong
(DD-Net + YOLOv8-pose). Danh sach camera duoc khai bao trong config.py.

Chay:
    python app.py

Sau do mo trinh duyet: http://127.0.0.1:5000
"""
import argparse
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
from flask import Flask, Response, abort, jsonify, render_template
from scipy.ndimage import zoom as scipy_zoom
from scipy.signal import medfilt
from ultralytics import YOLO

import config

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from ddnet_model import Config, DDNetOriginal  # noqa: E402
from pose_tracking import track_people  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
TRACKER_PATH = os.path.join(PROJECT_DIR, "pose_bytetrack.yaml")
TRACK_STATE_TTL_FRAMES = 30
STREAM_PUSH_INTERVAL = 1 / 20  # gioi han toc do day MJPEG toi trinh duyet, doc lap voi toc do suy luan

app = Flask(__name__)

STATE_LOCK = threading.Lock()
FRAME_LOCK = threading.Lock()
STATES = {}          # cam_id -> dict trang thai hien tai cua camera
LATEST_FRAMES = {}   # cam_id -> bytes JPEG moi nhat (None neu chua co / mat ket noi)
REPLAY_EVENTS = {}   # cam_id -> threading.Event, bao worker phat lai video da ket thuc
GLOBAL_ALERTS = deque(maxlen=100)
TOTAL_ALERTS_ALL = 0


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


def get_cg(p: np.ndarray, cfg: Config, device: torch.device) -> np.ndarray:
    iu = np.triu_indices(cfg.joint_n, 1)
    frames = []
    for f in range(cfg.frame_l):
        pf = torch.from_numpy(p[f]).float().to(device)
        d_m = torch.cdist(pf, pf, p=2)
        frames.append(d_m[iu].cpu().numpy())
    return norm_scale(np.stack(frames))


def draw_corner_box(frame: np.ndarray, bbox: np.ndarray, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox.astype(int)
    lx = max(int((x2 - x1) * 0.22), 12)
    ly = max(int((y2 - y1) * 0.22), 12)
    t = 3
    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * lx, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy * ly), color, t)


def draw_hud(frame: np.ndarray, cam_name: str, is_climb: bool, person_detected: bool, fps: float) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)

    cv2.putText(frame, cam_name, (12, 26), FONT, 0.6, (200, 205, 212), 2)

    if is_climb:
        status_text, status_color = "ALERT", (0, 0, 255)
    elif person_detected:
        status_text, status_color = "MONITORING", (60, 165, 210)
    else:
        status_text, status_color = "SECURE", (110, 180, 80)

    (tw, _), _ = cv2.getTextSize(status_text, FONT, 0.6, 2)
    cv2.putText(frame, status_text, (w - tw - 12, 26), FONT, 0.6, status_color, 2)
    cv2.putText(frame, f"{fps:4.1f} FPS", (12, h - 10), FONT, 0.5, (170, 170, 170), 1)

    if is_climb:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)


def init_camera_state(cam_id: str, name: str) -> None:
    with STATE_LOCK:
        STATES[cam_id] = {
            "id": cam_id,
            "name": name,
            "connected": False,
            "error": None,
            "is_climb": False,
            "person_detected": False,
            "ended": False,
            "fps": 0.0,
            "first_alert_time": None,
            "total_alerts": 0,
            "start_time": time.time(),
        }
    with FRAME_LOCK:
        LATEST_FRAMES[cam_id] = None
    REPLAY_EVENTS[cam_id] = threading.Event()


def camera_worker(cam_id: str, name: str, source, model_path: str) -> None:
    """Vong lap rieng cho 1 camera: doc frame, suy luan DD-Net, ghi vao STATES/LATEST_FRAMES."""
    global TOTAL_ALERTS_ALL

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_attention = False

    try:
        pose_model = YOLO(os.path.join(PROJECT_DIR, "yolov8l-pose.pt"))

        fname = os.path.basename(model_path)
        use_attention = "attention" in fname and "no_attention" not in fname
        model = DDNetOriginal(cfg, use_attention=use_attention).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        cap_source = int(source) if (isinstance(source, str) and source.isdigit()) else source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise RuntimeError(f"Khong mo duoc nguon video: {source}")
    except Exception as exc:
        print(f"[{cam_id}] LOI khoi tao: {exc}", flush=True)
        with STATE_LOCK:
            STATES[cam_id]["error"] = str(exc)
            STATES[cam_id]["connected"] = False
        return

    print(f"[{cam_id}] device={device} model={os.path.basename(model_path)} attention={use_attention}", flush=True)

    is_file_source = isinstance(cap_source, str)
    with STATE_LOCK:
        STATES[cam_id]["connected"] = True
        STATES[cam_id]["start_time"] = time.time()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 1:
        video_fps = 30.0
    frame_interval = 1.0 / video_fps
    playback_clock_start = time.time()
    frame_idx = 0

    track_buffers: dict[int, deque[np.ndarray]] = {}
    track_last_seen: dict[int, int] = {}
    track_climb: dict[int, bool] = {}
    prev_time = time.time()
    last_alert_time = -config.ALERT_COOLDOWN_SEC
    is_climb = False
    use_half = device.type == "cuda"

    replay_event = REPLAY_EVENTS[cam_id]

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_file_source:
                with STATE_LOCK:
                    STATES[cam_id]["ended"] = True
                    STATES[cam_id]["is_climb"] = False
                    STATES[cam_id]["person_detected"] = False
                replay_event.wait()
                replay_event.clear()

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                playback_clock_start = time.time()
                frame_idx = 0
                track_buffers.clear()
                track_last_seen.clear()
                track_climb.clear()
                is_climb = False
                last_alert_time = -config.ALERT_COOLDOWN_SEC
                with STATE_LOCK:
                    STATES[cam_id]["ended"] = False
                    STATES[cam_id]["start_time"] = time.time()
                    STATES[cam_id]["first_alert_time"] = None
                continue
            break
        frame_idx += 1

        tracked_people = track_people(
            pose_model,
            frame,
            TRACKER_PATH,
            conf=0.25,
            imgsz=config.YOLO_IMGSZ,
            quantize=16 if use_half else None,
        )
        seen_ids = set()

        for tracked in tracked_people:
            track_id = tracked.track_id
            seen_ids.add(track_id)
            track_last_seen[track_id] = frame_idx
            pose_buffer = track_buffers.setdefault(
                track_id, deque(maxlen=cfg.frame_l)
            )
            pose_buffer.append(tracked.keypoints_xyn)

            if len(pose_buffer) == cfg.frame_l and frame_idx % config.INFER_STRIDE == 0:
                seq = zoom_sequence(
                    np.array(pose_buffer),
                    cfg.frame_l,
                    cfg.joint_n,
                    cfg.joint_d,
                )
                cg = get_cg(seq, cfg, device)
                m_t = torch.from_numpy(cg).unsqueeze(0).float().to(device)
                p_t = torch.from_numpy(np.expand_dims(seq, axis=0)).float().to(device)

                with torch.no_grad():
                    outputs = model(m_t, p_t)
                    track_climb[track_id] = torch.argmax(outputs, dim=1).item() == 1

        for track_id in list(track_buffers):
            if frame_idx - track_last_seen.get(track_id, frame_idx) > TRACK_STATE_TTL_FRAMES:
                track_buffers.pop(track_id, None)
                track_last_seen.pop(track_id, None)
                track_climb.pop(track_id, None)

        is_climb = any(track_climb.get(track_id, False) for track_id in seen_ids)
        person_detected = bool(tracked_people)

        elapsed = time.time() - STATES[cam_id]["start_time"]
        with STATE_LOCK:
            if is_climb:
                if STATES[cam_id]["first_alert_time"] is None:
                    STATES[cam_id]["first_alert_time"] = round(elapsed, 1)
                if elapsed - last_alert_time >= config.ALERT_COOLDOWN_SEC:
                    GLOBAL_ALERTS.appendleft({
                        "cam_id": cam_id,
                        "cam_name": name,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "elapsed": round(elapsed, 1),
                    })
                    STATES[cam_id]["total_alerts"] += 1
                    TOTAL_ALERTS_ALL += 1
                    last_alert_time = elapsed

        with STATE_LOCK:
            STATES[cam_id]["is_climb"] = is_climb
            STATES[cam_id]["person_detected"] = person_detected

        for tracked in tracked_people:
            tracked_is_climb = track_climb.get(tracked.track_id, False)
            draw_corner_box(
                frame,
                tracked.bbox,
                (0, 0, 255) if tracked_is_climb else (125, 175, 76),
            )
            x1, y1, _, _ = tracked.bbox.astype(int)
            cv2.putText(
                frame,
                f"ID {tracked.track_id} | {'CLIMB' if tracked_is_climb else 'NO CLIMB'}",
                (x1, max(55, y1 - 8)),
                FONT,
                0.5,
                (0, 0, 255) if tracked_is_climb else (125, 175, 76),
                2,
            )

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-8)
        prev_time = curr_time
        with STATE_LOCK:
            STATES[cam_id]["fps"] = fps

        draw_hud(frame, name, is_climb, keypoints is not None, fps)

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            with FRAME_LOCK:
                LATEST_FRAMES[cam_id] = buffer.tobytes()

        if is_file_source:
            expected_elapsed = frame_idx * frame_interval
            actual_elapsed = time.time() - playback_clock_start
            skips = 0
            while (actual_elapsed - expected_elapsed) > frame_interval and skips < config.MAX_SKIP_PER_TICK:
                if not cap.grab():
                    break
                frame_idx += 1
                skips += 1
                expected_elapsed = frame_idx * frame_interval

    cap.release()
    with STATE_LOCK:
        STATES[cam_id]["connected"] = False


def stream_camera(cam_id: str):
    while True:
        with FRAME_LOCK:
            frame = LATEST_FRAMES.get(cam_id)
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(STREAM_PUSH_INTERVAL)


@app.route("/")
def index():
    return render_template("index.html", cameras=config.CAMERAS)


@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    if cam_id not in STATES:
        abort(404)
    return Response(stream_camera(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with STATE_LOCK:
        cams = []
        for cam in config.CAMERAS:
            s = STATES[cam["id"]]
            uptime = time.time() - s["start_time"] if s["connected"] else 0
            cams.append({
                "id": s["id"],
                "name": s["name"],
                "connected": s["connected"],
                "error": s["error"],
                "is_climb": s["is_climb"],
                "person_detected": s["person_detected"],
                "ended": s["ended"],
                "fps": round(s["fps"], 1),
                "first_alert_time": s["first_alert_time"],
                "total_alerts": s["total_alerts"],
                "uptime": round(uptime, 1),
            })
        return jsonify({
            "cameras": cams,
            "total_alerts_all": TOTAL_ALERTS_ALL,
            "any_alert": any(c["is_climb"] for c in cams),
        })


@app.route("/api/alerts")
def api_alerts():
    with STATE_LOCK:
        return jsonify(list(GLOBAL_ALERTS))


@app.route("/api/replay/<cam_id>", methods=["POST"])
def api_replay(cam_id):
    if cam_id not in REPLAY_EVENTS:
        abort(404)
    REPLAY_EVENTS[cam_id].set()
    return jsonify({"ok": True})


@app.route("/api/alerts/clear", methods=["POST"])
def api_alerts_clear():
    global TOTAL_ALERTS_ALL
    with STATE_LOCK:
        GLOBAL_ALERTS.clear()
        TOTAL_ALERTS_ALL = 0
        for state in STATES.values():
            state["total_alerts"] = 0
    return jsonify({"ok": True})


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard giam sat nhieu camera - canh bao xam nhap treo tuong")
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    args = parser.parse_args()

    if not config.CAMERAS:
        raise RuntimeError("config.py: CAMERAS dang rong, hay khai bao it nhat 1 camera")

    for cam in config.CAMERAS:
        model_path = cam.get("model") or config.DEFAULT_MODEL
        init_camera_state(cam["id"], cam["name"])
        threading.Thread(
            target=camera_worker,
            args=(cam["id"], cam["name"], cam["source"], model_path),
            daemon=True,
        ).start()

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
