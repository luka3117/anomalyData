#!/usr/bin/env python
# coding: utf-8

"""
学習済み Autoencoder を使って、選択した画像だけを異常判定する。

事前に必要なもの
----------------
nutcheck_results/
├── autoencoder_model.pth
└── threshold.npy

使い方
------
1. SELECTED_IMAGES に調べたい画像を指定する
2. このスクリプトを実行する
3. reconstruction_selected/ と selected_results.csv を確認する
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


# ============================================================
# 1. 基本設定
# ============================================================

IMG_SIZE = 128

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("device =", DEVICE)


# ============================================================
# 2. ファイルの場所
# ============================================================

DATA_ROOT = Path.cwd() / "hazelnutData"
OUTPUT_DIR = Path.cwd() / "nutcheck_results"

MODEL_FILE = OUTPUT_DIR / "autoencoder_model.pth"
THRESHOLD_FILE = OUTPUT_DIR / "threshold.npy"

SELECTED_OUTPUT_DIR = OUTPUT_DIR / "reconstruction_selected"
SELECTED_OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. 判定したい画像を選ぶ
# ============================================================

# ここだけ変更すればよい。
#
# 例：
#   good   -> 正常画像
#   crack  -> 異常画像
#   hole   -> 異常画像

SELECTED_IMAGES = [
    DATA_ROOT / "test" / "good" / "007.png",
    DATA_ROOT / "test" / "crack" / "003.png",
    DATA_ROOT / "test" / "hole" / "005.png",
    DATA_ROOT / "test" / "print" / "000.png",
    DATA_ROOT / "test" / "print" / "002.png",
]


# ============================================================
# 4. 画像の前処理
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


# ============================================================
# 5. Autoencoder の構造
# ============================================================

# state_dict() で保存したファイルには「重み」だけが入っている。
# そのため、読み込み側でも同じネットワーク構造を定義する必要がある。

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

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


# ============================================================
# 6. 学習済みモデルを読み込む
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"モデルが見つかりません: {MODEL_FILE}")

model = ConvAutoEncoder().to(DEVICE)

state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print(f"モデルを読み込みました: {MODEL_FILE}")


# # ============================================================
# # 7. しきい値を読み込む
# # ============================================================

# if not THRESHOLD_FILE.exists():
#     raise FileNotFoundError(
#         f"しきい値が見つかりません: {THRESHOLD_FILE}\n"
#         "学習用スクリプトで threshold.npy を保存してください。"
#     )

# threshold = float(np.load(THRESHOLD_FILE))

# print(f"しきい値を読み込みました: {THRESHOLD_FILE}")
# print(f"threshold = {threshold:.6f}")


# ============================================================
# 7. しきい値
# ============================================================

threshold = 0.000786

print(f"threshold = {threshold:.6f}")




# ============================================================
# 8. 1枚の画像を再構成して異常スコアを計算
# ============================================================

def analyze_image(image_path):
    """1枚の画像について再構成・異常スコア・判定結果を返す。"""

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x_hat = model(x)

    input_image = x[0, 0].cpu().numpy()
    reconstructed = x_hat[0, 0].cpu().numpy()

    error_map = (reconstructed - input_image) ** 2

    # easy version と同じ：
    # 画像全体の平均再構成誤差を異常スコアとする。
    anomaly_score = error_map.mean()

    # 0 = normal, 1 = anomaly
    y_pred = int(anomaly_score >= threshold)

    return input_image, reconstructed, error_map, anomaly_score, y_pred


# ============================================================
# 9. 正解ラベルが分かる場合だけ取得
# ============================================================

def infer_true_status(image_path):
    """
    hazelnutData/test/ 以下の既知フォルダなら正解を推定する。

    good              -> normal
    crack/cut/hole/... -> anomaly

    それ以外の任意画像なら unknown とする。
    """

    parent = image_path.parent.name

    if parent == "good":
        return 0, "normal"

    known_anomaly_dirs = {"crack", "cut", "hole", "print"}

    if parent in known_anomaly_dirs:
        return 1, "anomaly"

    return None, "unknown"


# ============================================================
# 10. 選択した画像を順番に判定
# ============================================================

results = []

for image_path in SELECTED_IMAGES:

    if not image_path.exists():
        print(f"画像が見つかりません: {image_path}")
        continue

    input_image, reconstructed, error_map, score, y_pred = analyze_image(
        image_path
    )

    pred_status = "anomaly" if y_pred == 1 else "normal"
    y_true, true_status = infer_true_status(image_path)

    # --------------------------------------------------------
    # 再構成結果を図として保存
    # --------------------------------------------------------

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

    fig.suptitle(
        f"{image_path.name}   score={score:.6f}   pred={pred_status}",
        fontsize=10,
    )

    plt.tight_layout()

    # 同じファイル名が別フォルダに存在しても上書きしないよう、
    # 親フォルダ名も出力ファイル名に含める。
    figure_file = (
        SELECTED_OUTPUT_DIR
        / f"{image_path.parent.name}_{image_path.stem}_result.png"
    )

    plt.savefig(figure_file, dpi=300, bbox_inches="tight")
    plt.close()

    print()
    print(f"画像     : {image_path}")
    print(f"score    : {score:.6f}")
    print(f"prediction: {pred_status}")
    print(f"保存しました: {figure_file}")

    # --------------------------------------------------------
    # CSV 用の結果を保存
    # --------------------------------------------------------

    correct = None if y_true is None else (y_true == y_pred)

    results.append({
        "file": image_path.name,
        "path": str(image_path),
        "source_type": image_path.parent.name,
        "y_true": y_true,
        "true_status": true_status,
        "anomaly_score": score,
        "threshold": threshold,
        "y_pred": y_pred,
        "pred_status": pred_status,
        "correct": correct,
        "figure_file": figure_file.name,
    })


# ============================================================
# 11. 選択した画像の結果を CSV に保存
# ============================================================

results_df = pd.DataFrame(results)

results_csv = OUTPUT_DIR / "selected_results.csv"
results_df.to_csv(results_csv, index=False)

print()
print("=== 選択画像の判定結果 ===")
print(results_df)

print()
print(f"CSVを保存しました: {results_csv}")
print(f"図の保存先: {SELECTED_OUTPUT_DIR}")
