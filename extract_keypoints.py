"""
Trich xuat pose keypoints (YOLOv8-pose) tu video Data/{Train,Valid,Test}/{climb,no_climb}
Du lieu goc duoc chia thanh 3 thu muc "Bo data_final-00X" (do gioi han dung luong upload),
script nay gop chung lai truoc khi trich xuat, giu nguyen logic cua load_data.ipynb goc.
"""
import os
import sys
import time
import numpy as np
from ultralytics import YOLO

ROOT = "d:/Projects/CV/climb"
PARTS = ["Bộ data_final-001", "Bộ data_final-002", "Bộ data_final-003"]
OUT_DIR = os.path.join(ROOT, "project")

SPLITS = ["Train", "Valid", "Test"]
CLASSES = ["climb", "no_climb"]

OUT_NAME = {
    ("Train", "climb"): "train_climb_keypoints.npy",
    ("Train", "no_climb"): "train_no_climb_keypoints.npy",
    ("Valid", "climb"): "val_climb_keypoints.npy",
    ("Valid", "no_climb"): "val_no_climb_keypoints.npy",
    ("Test", "climb"): "test_climb_keypoints.npy",
    ("Test", "no_climb"): "test_no_climb_keypoints.npy",
}


def collect_video_paths(split, cls):
    paths = []
    for part in PARTS:
        folder = os.path.join(ROOT, part, "Bộ data_final", "Data", split, cls)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            paths.append(os.path.join(folder, fname))
    return paths


def extract_pose_sequences(video_paths, model, conf_threshold=0.7, min_sequence_len=40, step=5):
    list_data = []
    for idx, input_path in enumerate(video_paths):
        list_results = []
        try:
            results = model(input_path, conf=conf_threshold, stream=True, save=False, verbose=False)
            for r in results:
                if r.keypoints is None or len(r.keypoints.xyn) == 0 or len(r.keypoints.xyn[0]) == 0:
                    continue
                list_results.append(np.array(r.keypoints.xyn[0].tolist()))
        except Exception as e:
            print(f"  [{idx+1}/{len(video_paths)}] LOI xu ly {os.path.basename(input_path)}: {e}", flush=True)
            continue

        if len(list_results) < min_sequence_len:
            print(f"  [{idx+1}/{len(video_paths)}] BO QUA {os.path.basename(input_path)} "
                  f"(chi co {len(list_results)} frame co pose < {min_sequence_len})", flush=True)
            continue

        n_seq = 0
        for i in range(0, len(list_results) - min_sequence_len + 1, step):
            sequence = list_results[i:i + min_sequence_len]
            list_data.append(sequence)
            n_seq += 1

        print(f"  [{idx+1}/{len(video_paths)}] OK {os.path.basename(input_path)} "
              f"-> {len(list_results)} frame, {n_seq} chuoi", flush=True)

    return list_data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Dang tai model YOLOv8l-pose ...", flush=True)
    model = YOLO("yolov8l-pose.pt")

    summary = []
    for split in SPLITS:
        for cls in CLASSES:
            out_name = OUT_NAME[(split, cls)]
            out_path = os.path.join(OUT_DIR, out_name)
            if os.path.exists(out_path):
                print(f"=== {split}/{cls}: da ton tai {out_name}, bo qua ===", flush=True)
                continue

            video_paths = collect_video_paths(split, cls)
            print(f"=== {split}/{cls}: {len(video_paths)} video ===", flush=True)
            if not video_paths:
                continue

            t0 = time.time()
            pose_sequences = extract_pose_sequences(video_paths, model)
            elapsed = time.time() - t0

            if not pose_sequences:
                print(f"  CANH BAO: khong co chuoi keypoint nao cho {split}/{cls}", flush=True)
                summary.append((split, cls, 0, elapsed))
                continue

            pose_array = np.array(pose_sequences)
            np.save(out_path, pose_array)
            print(f"  Da luu {out_path} shape={pose_array.shape} ({elapsed:.1f}s)", flush=True)
            summary.append((split, cls, pose_array.shape[0], elapsed))

    print("\n=== TOM TAT ===", flush=True)
    for split, cls, n_seq, elapsed in summary:
        print(f"{split}/{cls}: {n_seq} chuoi, {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
