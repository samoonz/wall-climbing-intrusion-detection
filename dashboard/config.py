"""
Cau hinh danh sach camera/video duoc giam sat boi dashboard nhieu man hinh.

Them/bot camera bang cach sua danh sach CAMERAS ben duoi. Moi camera la 1 dict:
    id     : ma dinh danh duy nhat, khong dau/khong cach, dung trong URL (/video_feed/<id>)
    name   : ten hien thi tren giao dien
    source : duong dan file video, URL RTSP/HTTP, hoac chi so webcam (vd: 0)
    model  : (tuy chon) duong dan checkpoint DD-Net rieng cho camera nay;
             de None de dung DEFAULT_MODEL chung
"""
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL = os.path.join(PROJECT_DIR, "best_with_attention.pth")

CAMERAS = [
    {
        "id": "cam1",
        "name": "Camera 1 - San test",
        "source": os.path.join(PROJECT_DIR, "v1.MP4"),
        "model": None,
    },
    {
        "id": "cam2",
        "name": "Camera 2 - Webcam",
        "source": os.path.join(PROJECT_DIR, "v2.MP4"),
        "model": None,
    },
]

# Suy luan / xu ly anh - dung chung cho tat ca camera
YOLO_IMGSZ = 480
INFER_STRIDE = 2
MAX_SKIP_PER_TICK = 10
ALERT_COOLDOWN_SEC = 5.0

# May chu web
HOST = "127.0.0.1"
PORT = 5000
