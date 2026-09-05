#!/usr/bin/env python
# coding: utf-8

"""
ヘーゼルナッツ画像を用いた Autoencoder 異常検知（講義用・簡易版）

学習：
    正常画像（good）のみ

テスト：
    正常画像 + 異常画像

異常スコア：
    画像全体の平均再構成誤差

主な出力：
    nutcheck_results/
    ├── training_loss.png
    ├── reconstruction_normal.png
    ├── reconstruction_anomaly.png
    ├── anomaly_score_histogram.png
    ├── anomaly_score_kde.png
    └── anomaly_detection_results.csv

想定するフォルダ構成：
    current_folder/
    ├── nutcheck_easy_clean_jp.py
    └── hazelnutData/
        ├── train/
        │   └── good/
        └── test/
            ├── good/
            ├── crack/
            ├── cut/
            ├── hole/
            └── print/

必要なパッケージ：
    torch torchvision pillow matplotlib numpy pandas scikit-learn scipy
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics import roc_auc_score


# ============================================================
# 1. 基本設定
# ============================================================

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 利用可能なデバイスを自動選択する。
# CUDA : NVIDIA GPU
# MPS  : Apple Silicon Mac
# CPU  : 上記が利用できない場合
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("device =", DEVICE)


# ============================================================
# 2. データフォルダと出力フォルダ
# ============================================================

# このスクリプトを hazelnutData/ と同じ階層から実行する。
#
# hazelnutData/
# ├── train/
# │   └── good/
# └── test/
#     ├── good/       -> 正常
#     ├── crack/      -> 異常
#     ├── cut/        -> 異常
#     ├── hole/       -> 異常
#     └── print/      -> 異常

DATA_ROOT = Path.cwd() / "hazelnutData"

TRAIN_GOOD_DIR = DATA_ROOT / "train" / "good"
TEST_DIR = DATA_ROOT / "test"
TEST_GOOD_DIR = TEST_DIR / "good"

# 図や CSV はこのフォルダにまとめて保存する。
OUTPUT_DIR = Path.cwd() / "nutcheck_results"
OUTPUT_DIR.mkdir(exist_ok=True)

if not TRAIN_GOOD_DIR.exists():
    raise FileNotFoundError(f"学習フォルダが見つかりません: {TRAIN_GOOD_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"テストフォルダが見つかりません: {TEST_DIR}")

print("\n=== フォルダ ===")
print("データ       :", DATA_ROOT)
print("学習データ   :", TRAIN_GOOD_DIR)
print("テストデータ :", TEST_DIR)
print("結果保存先   :", OUTPUT_DIR)


# ============================================================
# 3. 画像ファイルを集める
# ============================================================

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def image_files(folder):
    """フォルダ直下の画像ファイルを名前順に返す。"""
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


train_good_paths = image_files(TRAIN_GOOD_DIR)
test_good_paths = image_files(TEST_GOOD_DIR)

# crack/, cut/, hole/, print/ などの異常フォルダを取得する。
# good/ は正常データなので除外する。
anomaly_dirs = sorted(
    folder for folder in TEST_DIR.iterdir()
    if folder.is_dir() and folder.name != "good"
)

# すべての異常画像を1つのリストにまとめる。
test_anomaly_paths = [
    path
    for folder in anomaly_dirs
    for path in image_files(folder)
]

print("\n=== 画像枚数 ===")
print("train good  :", len(train_good_paths))
print("test good   :", len(test_good_paths))
print("test anomaly:", len(test_anomaly_paths))


# ============================================================
# 4. 画像の前処理
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),       # 128 x 128 にリサイズ
    transforms.Grayscale(num_output_channels=1),  # RGB -> グレースケール
    transforms.ToTensor(),                        # Tensor に変換し [0, 1] にする
])


class HazelnutDataset(Dataset):
    """1枚ずつ画像・ラベル・ファイルパスを返す Dataset。"""

    def __init__(self, image_paths, labels):
        self.image_paths = list(image_paths)
        self.labels = list(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]

        image = Image.open(path).convert("RGB")
        image = transform(image)

        label = self.labels[index]

        return image, label, str(path)


# 学習データは正常画像のみ。
# 正常 = 0
train_dataset = HazelnutDataset(
    train_good_paths,
    [0] * len(train_good_paths),
)

# テストデータは正常 + 異常。
test_paths = test_good_paths + test_anomaly_paths

# 正解ラベル：
# 0 = normal
# 1 = anomaly
test_labels = (
    [0] * len(test_good_paths)
    + [1] * len(test_anomaly_paths)
)

test_dataset = HazelnutDataset(
    test_paths,
    test_labels,
)


# 学習用 DataLoader。
# shuffle=True により、各 epoch で画像の順番を変える。
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# 学習画像の異常スコア計算用。
# 順番を固定するため shuffle=False。
train_score_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# テスト結果と test_paths の順番を対応させるため shuffle=False。
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# 5. Convolutional Autoencoder
# ============================================================

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        #
        # 1 x 128 x 128
        #      ↓
        # 16 x 64 x 64
        #      ↓
        # 32 x 32 x 32
        #      ↓
        # 64 x 16 x 16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Decoder
        #
        # 64 x 16 x 16
        #      ↓
        # 32 x 32 x 32
        #      ↓
        # 16 x 64 x 64
        #      ↓
        # 1 x 128 x 128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


model = ConvAutoEncoder().to(DEVICE)

# 入力画像と再構成画像の MSE を学習誤差として使用する。
criterion = nn.MSELoss()

# Autoencoder のパラメータを Adam で更新する。
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n=== Model ===")
print(model)


# ============================================================
# 6. 正常画像のみで学習
# ============================================================

loss_history = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, _, _ in train_loader:
        images = images.to(DEVICE)

        optimizer.zero_grad()

        # Autoencoder で画像を再構成する。
        reconstructed = model(images)

        # Autoencoder では入力画像自身が正解データになる。
        loss = criterion(reconstructed, images)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    epoch_loss = total_loss / len(train_dataset)
    loss_history.append(epoch_loss)

    print(f"Epoch {epoch + 1:02d}/{EPOCHS}  loss = {epoch_loss:.6f}")

# ============================================================
# 6-1. 学習済みモデルを保存
# ============================================================

MODEL_FILE = OUTPUT_DIR / "autoencoder_model.pth"

torch.save(model.state_dict(), MODEL_FILE)

print(f"モデルを保存しました: {MODEL_FILE}")


# ============================================================
# 7. 学習誤差を保存
# ============================================================

plt.figure(figsize=(6, 3.5))
plt.plot(range(1, EPOCHS + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Training progress")
plt.tight_layout()

training_loss_file = OUTPUT_DIR / "training_loss.png"
plt.savefig(training_loss_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"保存しました: {training_loss_file}")


# ============================================================
# 8. 再構成画像と誤差マップ
# ============================================================

def reconstruct(image_path):
    """入力画像、再構成画像、画素ごとの再構成誤差を返す。"""
    image = Image.open(image_path).convert("RGB")

    # 1 x 128 x 128 -> 1 x 1 x 128 x 128
    # 先頭に batch 次元を追加する。
    x = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():
        x_hat = model(x)

    input_image = x[0, 0].cpu().numpy()
    reconstructed = x_hat[0, 0].cpu().numpy()

    # 各画素の二乗再構成誤差。
    error_map = (reconstructed - input_image) ** 2

    return input_image, reconstructed, error_map


def save_reconstruction(image_path, output_file, label_text):
    """入力・再構成・誤差マップを1枚の図として保存する。"""
    input_image, reconstructed, error_map = reconstruct(image_path)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    axes[0].imshow(input_image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")

    axes[1].imshow(reconstructed, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reconstruction")

    axes[2].imshow(input_image, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(error_map, cmap="jet", alpha=0.55)
    axes[2].set_title("Error map")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{label_text}: {Path(image_path).name}", fontsize=10)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"保存しました: {output_file}")
    print(f"  元画像: {Path(image_path).name}")


# 正常画像・異常画像から1枚ずつ例を選び、図として保存する。
normal_example = random.choice(test_good_paths)
anomaly_example = random.choice(test_anomaly_paths)


# 直接に指定する場合は、以下のように書き換えることもできる。
# normal_example = TEST_GOOD_DIR / "007.png"
# anomaly_example = TEST_DIR / "crack" / "003.png"

save_reconstruction(
    normal_example,
    OUTPUT_DIR / "reconstruction_normal.png",
    "Normal",
)

save_reconstruction(
    anomaly_example,
    OUTPUT_DIR / "reconstruction_anomaly.png",
    "Anomaly",
)


# ============================================================
# 9. 異常スコアを計算
# ============================================================

# この簡易版では画像全体の平均再構成誤差を使用する。
#
# anomaly score = mean[(x_hat - x)^2]
#
# したがって、1枚の画像につき1つの異常スコアが得られる。


def calculate_scores(loader):
    """各画像について1つの再構成誤差スコアを計算する。"""
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(DEVICE)
            reconstructed = model(images)

            # 画素ごとの二乗再構成誤差。
            error = (reconstructed - images).pow(2)

            # channel, height, width の平均を取り、
            # 1画像につき1つの異常スコアにする。
            image_scores = error.mean(dim=(1, 2, 3))

            all_scores.extend(image_scores.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.asarray(all_scores), np.asarray(all_labels)


train_scores, _ = calculate_scores(train_score_loader)
test_scores, y_true = calculate_scores(test_loader)

print("\n=== 異常スコア数 ===")
print("train:", len(train_scores))
print("test :", len(test_scores))


# ============================================================
# 10. しきい値と予測
# ============================================================

# 正常学習データの異常スコアの95パーセンタイルをしきい値とする。
threshold = np.percentile(train_scores, 95)

# threshold 以上なら anomaly と判定する。
# 0 = normal
# 1 = anomaly
y_pred = (test_scores >= threshold).astype(int)

print("\n=== しきい値と平均スコア ===")
print(f"threshold          = {threshold:.6f}")
print(f"normal mean score  = {test_scores[y_true == 0].mean():.6f}")
print(f"anomaly mean score = {test_scores[y_true == 1].mean():.6f}")


# ============================================================
# 11. 簡単な評価
# ============================================================

accuracy = (y_pred == y_true).mean()
roc_auc = roc_auc_score(y_true, test_scores)

print("\n=== 評価 ===")
print(f"accuracy = {accuracy:.4f}")
print(f"ROC-AUC  = {roc_auc:.4f}")


# ============================================================
# 12. テスト結果を CSV に保存
# ============================================================

# test_loader は shuffle=False なので、
# test_paths, y_true, y_pred, test_scores の順番は対応している。

results_df = pd.DataFrame({
    "file": [path.name for path in test_paths],
    "relative_path": [
        str(path.relative_to(DATA_ROOT))
        for path in test_paths
    ],
    "defect_type": [path.parent.name for path in test_paths],
    "y_true": y_true,                  # 0 = normal, 1 = anomaly
    "true_status": np.where(y_true == 0, "normal", "anomaly"),
    "anomaly_score": test_scores,
    "threshold": threshold,
    "y_pred": y_pred,                  # 0 = normal, 1 = anomaly
    "pred_status": np.where(y_pred == 0, "normal", "anomaly"),
    "correct": y_true == y_pred,
})

results_csv = OUTPUT_DIR / "anomaly_detection_results.csv"
results_df.to_csv(results_csv, index=False)

print(f"\n保存しました: {results_csv}")
print("CSV columns:")
print(", ".join(results_df.columns))
print("\n先頭5件:")
print(results_df.head())


# ============================================================
# 13. 異常スコアのヒストグラム
# ============================================================

normal_scores = test_scores[y_true == 0]
anomaly_scores = test_scores[y_true == 1]

plt.figure(figsize=(7, 4))

plt.hist(normal_scores, bins=25, alpha=0.6, label="normal")
plt.hist(anomaly_scores, bins=25, alpha=0.6, label="anomaly")
plt.axvline(threshold, linestyle="--", label="threshold")

plt.xlabel("Anomaly score")
plt.ylabel("Count")
plt.title("Distribution of anomaly scores")
plt.legend()
plt.tight_layout()

histogram_file = OUTPUT_DIR / "anomaly_score_histogram.png"
plt.savefig(histogram_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"保存しました: {histogram_file}")


# ============================================================
# 14. 異常スコアの KDE
# ============================================================

plt.figure(figsize=(7, 4))

# density=True とすると縦軸は度数ではなく確率密度になる。
plt.hist(normal_scores, bins=25, density=True, alpha=0.4, label="normal")
plt.hist(anomaly_scores, bins=25, density=True, alpha=0.4, label="anomaly")

# KDE（カーネル密度推定）で滑らかな分布曲線を描く。
x_min = min(normal_scores.min(), anomaly_scores.min())
x_max = max(normal_scores.max(), anomaly_scores.max())
x_grid = np.linspace(x_min, x_max, 300)

normal_kde = gaussian_kde(normal_scores)
anomaly_kde = gaussian_kde(anomaly_scores)

plt.plot(x_grid, normal_kde(x_grid), label="normal KDE")
plt.plot(x_grid, anomaly_kde(x_grid), label="anomaly KDE")
plt.axvline(threshold, linestyle="--", label="threshold")

plt.xlabel("Anomaly score")
plt.ylabel("Density")
plt.title("Distribution of anomaly scores with KDE")
plt.legend()
plt.tight_layout()

kde_file = OUTPUT_DIR / "anomaly_score_kde.png"
plt.savefig(kde_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"保存しました: {kde_file}")


# ============================================================
# 15. 出力ファイル一覧
# ============================================================

print("\n=== 出力ファイル ===")

for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        print(f"  {path.name}")

print(f"\n結果は次のフォルダに保存されました:\n{OUTPUT_DIR}")


# ============================================================
# End
# ============================================================

# この簡易版では、以下は意図的に省略している。
#
# - 再構成誤差上位5%のみを使うスコア
# - 画像周辺部分の除外
# - classification_report
# - confusion matrix
# - 異常種類ごとの詳細集計
# - ground-truth mask
# - pixel-level ROC-AUC
