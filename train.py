"""
Huan luyen DD-Net (skeleton-based) de phan loai climb / no_climb tu cac chuoi
keypoint (.npy) da duoc extract_keypoints.py trich xuat.
Chuyen the truc tiep tu logic cua train.ipynb goc, chay duoc nhu mot script doc lap.
"""
import os
import random

import numpy as np
from scipy.signal import medfilt
from scipy.spatial.distance import cdist
from scipy.ndimage import zoom as scipy_zoom
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score

from ddnet_model import Config, DDNetOriginal

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


# ---------------- Data preprocessing ----------------
def zoom(p, target_l=40, joints_num=17, joints_dim=2):
    l = p.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:, m, n] = medfilt(p[:, m, n], 3)
            p_new[:, m, n] = scipy_zoom(p[:, m, n], target_l / l)[:target_l]
    return p_new


def norm_scale(x):
    return (x - np.mean(x)) / np.mean(x)


def get_CG(p, C):
    M = []
    iu = np.triu_indices(C.joint_n, 1)
    for f in range(C.frame_l):
        d_m = cdist(p[f], p[f], "euclidean")
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    return norm_scale(M)


# ---------------- Load data ----------------
def load_pose_with_label(file_path, label):
    data = np.load(file_path)
    labels = np.full((len(data),), label, dtype=np.int32)
    print(f"Da load: {os.path.basename(file_path)} - Du lieu: {data.shape}, Label: {label}")
    return data, labels


def load_all_data(base_path):
    files = {
        "train_climb": os.path.join(base_path, "train_climb_keypoints.npy"),
        "train_no_climb": os.path.join(base_path, "train_no_climb_keypoints.npy"),
        "val_climb": os.path.join(base_path, "val_climb_keypoints.npy"),
        "val_no_climb": os.path.join(base_path, "val_no_climb_keypoints.npy"),
        "test_climb": os.path.join(base_path, "test_climb_keypoints.npy"),
        "test_no_climb": os.path.join(base_path, "test_no_climb_keypoints.npy"),
    }

    def concat_data(f_climb, f_no_climb):
        X1, y1 = load_pose_with_label(f_climb, 1)
        X2, y2 = load_pose_with_label(f_no_climb, 0)
        return np.concatenate([X1, X2]), np.concatenate([y1, y2])

    X_train, y_train = concat_data(files["train_climb"], files["train_no_climb"])
    X_val, y_val = concat_data(files["val_climb"], files["val_no_climb"])
    X_test, y_test = concat_data(files["test_climb"], files["test_no_climb"])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class PoseDataset(Dataset):
    def __init__(self, X, y, config):
        self.X0 = []
        self.X1 = []
        self.Y = []

        for i in tqdm(range(len(X)), desc="Creating dataset"):
            p = zoom(np.copy(X[i]), target_l=config.frame_l, joints_num=config.joint_n, joints_dim=config.joint_d)
            label = np.eye(config.cls_num)[y[i]]
            self.X0.append(get_CG(p, config))
            self.X1.append(p)
            self.Y.append(label)

        self.X0 = torch.from_numpy(np.array(self.X0)).float()
        self.X1 = torch.from_numpy(np.array(self.X1)).float()
        self.Y = torch.from_numpy(np.array(self.Y)).float()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X0[idx], self.X1[idx]), self.Y[idx]


def create_dataset(X, y, config, batch_size=32, shuffle=True):
    dataset = PoseDataset(X, y, config)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(42)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------- Train / eval ----------------
@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for (X0, X1), Y in data_loader:
        X0 = X0.to(device)
        X1 = X1.to(device)
        labels = torch.argmax(Y, dim=1).to(device)

        outputs = model(X0, X1)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    precision = precision_score(all_labels.numpy(), all_preds.numpy(), average="binary", zero_division=0)
    recall = recall_score(all_labels.numpy(), all_preds.numpy(), average="binary", zero_division=0)
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="binary", zero_division=0)
    avg_loss = total_loss / len(data_loader)

    return accuracy, precision, recall, f1, avg_loss


def train_model(model, train_loader, val_loader, device, epochs=50, save_path="best_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for (X0, X1), Y in tqdm(train_loader):
            X0 = X0.to(device).float()
            X1 = X1.to(device).float()
            labels = torch.argmax(Y, dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(X0, X1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc, val_prec, val_rec, val_f1, val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Train Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            print(f"Luu mo hinh tot nhat voi do chinh xac validation: {val_acc:.4f}")
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    return best_val_acc


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_all_data(BASE_PATH)

    batch_size = 32
    train_loader = create_dataset(X_train, y_train, config, batch_size=batch_size, shuffle=True)
    val_loader = create_dataset(X_val, y_val, config, batch_size=batch_size, shuffle=False)
    test_loader = create_dataset(X_test, y_test, config, batch_size=batch_size, shuffle=False)

    model_no_att = DDNetOriginal(config, use_attention=False).to(device)
    train_model(
        model=model_no_att,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=50,
        save_path=os.path.join(BASE_PATH, "best_no_attention.pth"),
    )

    model_with_att = DDNetOriginal(config, use_attention=True).to(device)
    train_model(
        model=model_with_att,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=50,
        save_path=os.path.join(BASE_PATH, "best_with_attention.pth"),
    )

    model_no_att.load_state_dict(torch.load(os.path.join(BASE_PATH, "best_no_attention.pth")))
    model_with_att.load_state_dict(torch.load(os.path.join(BASE_PATH, "best_with_attention.pth")))

    criterion = nn.CrossEntropyLoss()
    acc_no_att, prec_no_att, rec_no_att, f1_no_att, _ = evaluate(model_no_att, test_loader, criterion, device)
    acc_att, prec_att, rec_att, f1_att, _ = evaluate(model_with_att, test_loader, criterion, device)

    print("\n=== Ket qua tren Test Set ===")
    print(f"Khong attention: Accuracy={acc_no_att:.4f}, Precision={prec_no_att:.4f}, Recall={rec_no_att:.4f}, F1={f1_no_att:.4f}")
    print(f"Co attention:    Accuracy={acc_att:.4f}, Precision={prec_att:.4f}, Recall={rec_att:.4f}, F1={f1_att:.4f}")


if __name__ == "__main__":
    main()
