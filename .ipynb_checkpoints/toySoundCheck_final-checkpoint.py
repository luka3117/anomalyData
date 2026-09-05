#!/usr/bin/env python
# coding: utf-8

"""
ToyADMOS ToyCar を用いた異常音検知

比較する方法
-------------
1. PCA
   正常データを低次元空間で表現し、
   再構成できない程度を異常スコアとする。

2. kNN
   正常データからの距離を異常スコアとする。

3. One-Class SVM
   正常データのまとまりから外れる程度を異常スコアとする。

学習には正常データのみを使用し、
テストでは正常音と異常音の両方を評価する。
"""


# ============================================================
# 1. ライブラリ
# ============================================================

from pathlib import Path
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ============================================================
# 2. 乱数シード
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 3. データフォルダ
# ============================================================

# 想定するフォルダ構成
#
# soundData/
# └── ToyCar/
#     ├── train/
#     └── test/
#
# このスクリプト / Notebook を
# soundData と同じ階層から実行することを想定する。

BASE_DIR = Path.cwd() / "soundData"

TRAIN_DIR = BASE_DIR / "ToyCar" / "train"
TEST_DIR = BASE_DIR / "ToyCar" / "test"

print("BASE_DIR :", BASE_DIR)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR :", TEST_DIR)


# フォルダが存在するか確認
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"学習データが見つかりません: {TRAIN_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"テストデータが見つかりません: {TEST_DIR}")


# ============================================================
# 4. 使用する機械 ID
# ============================================================

# ToyADMOS には複数の machine ID がある。
# ここでは id_02 の音声だけを使用する。
#
# 学習データ：
#   正常音のみ
#
# テストデータ：
#   正常音 + 異常音

TARGET_ID = "id_02"

train_files = sorted(
    str(path) for path in TRAIN_DIR.glob(f"normal_{TARGET_ID}_*.wav")
)

test_normal_files = sorted(
    str(path) for path in TEST_DIR.glob(f"normal_{TARGET_ID}_*.wav")
)

test_anomaly_files = sorted(
    str(path) for path in TEST_DIR.glob(f"anomaly_{TARGET_ID}_*.wav")
)


print("\n=== ファイル数 ===")
print("train       :", len(train_files))
print("test normal :", len(test_normal_files))
print("test anomaly:", len(test_anomaly_files))


# ファイル名を1件ずつ確認
print("\n=== ファイル例 ===")
print("train       :", train_files[0] if train_files else "ファイルなし")
print("test normal :", test_normal_files[0] if test_normal_files else "ファイルなし")
print("test anomaly:", test_anomaly_files[0] if test_anomaly_files else "ファイルなし")


# ============================================================
# 5. 音声から特徴量を作る
# ============================================================

def extract_feature(
    wav_path,          # sound file
    sr=16000,          # sampling rate [Hz]
    n_mels=128,        # number of Mel-frequency bands
    n_fft=1024,        # FFT window size
    hop_length=512,    # frame shift between successive windows
):    
    """
    1つの wav ファイルから特徴ベクトルを作成する。

    処理の流れ

    wav
      ↓
    waveform
      ↓
    Mel spectrogram
      ↓
    log-Mel spectrogram
      ↓
    各 Mel 周波数帯について
      ・時間平均
      ・時間標準偏差
    を計算
      ↓
    256次元特徴ベクトル

    n_mels=128 の場合、

        mean : 128次元
        std  : 128次元

    よって

        128 + 128 = 256次元
    """

    # --------------------------------------------------------
    # wav ファイルを読み込む
    # --------------------------------------------------------

    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    # y は1次元の音声波形
    #
    # 例：
    # [0.01, 0.03, -0.02, ...]


    # --------------------------------------------------------
    # Mel spectrogram を作る
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
    y=y,                    # waveform data
    sr=sr,                  # sampling rate [Hz]
    n_fft=n_fft,            # samples used for each FFT
    hop_length=hop_length,  # samples shifted to the next frame
    n_mels=n_mels,          # number of Mel-frequency bands
    )

    # mel.shape
    #
    # (Mel 周波数帯, 時間フレーム)
    #
    # 例：
    # (128, 313)


    # --------------------------------------------------------
    # Power → dB
    # --------------------------------------------------------

    logmel = librosa.power_to_db(mel, ref=np.max)

    # ref=np.max により、
    # その spectrogram 内の最大値が 0 dB になる。
    #
    # その他の値は最大値に対する相対的な dB 値になる。


    # --------------------------------------------------------
    # 時間方向を要約する
    # --------------------------------------------------------

    feat_mean = logmel.mean(axis=1)
    feat_std = logmel.std(axis=1)

    # axis=1 なので、
    # 各 Mel 周波数帯について時間方向に要約する。


    # --------------------------------------------------------
    # 128 + 128 = 256 次元
    # --------------------------------------------------------

    feature = np.hstack([feat_mean, feat_std])

    return feature.astype(np.float32)


def load_features(file_list):
    """
    複数の wav ファイルを特徴量行列に変換する。

    1 wav
        ↓ extract_feature()
    1 × 256 の特徴ベクトル

    n 個の wav
        ↓
    n × 256 の特徴量行列
    """

    features = []

    for wav_path in tqdm(file_list):
        feature = extract_feature(wav_path)
        features.append(feature)

    return np.array(features, dtype=np.float32)


# ============================================================
# 6. 全 wav ファイルを特徴量に変換
# ============================================================

X_train = load_features(train_files)

X_test_normal = load_features(test_normal_files)
X_test_anomaly = load_features(test_anomaly_files)


# 正常 + 異常を1つのテストデータにまとめる
X_test = np.vstack([X_test_normal, X_test_anomaly])


# 正解ラベル
#
# normal  = 0
# anomaly = 1

y_test = np.r_[
    np.zeros(len(X_test_normal), dtype=int),
    np.ones(len(X_test_anomaly), dtype=int),
]


print("\n=== 特徴量行列 ===")
print("X_train       :", X_train.shape)
print("X_test_normal :", X_test_normal.shape)
print("X_test_anomaly:", X_test_anomaly.shape)
print("X_test        :", X_test.shape)


# ============================================================
# 7. 標準化
# ============================================================

scaler = StandardScaler()

# 学習データだけを使って平均と標準偏差を推定する
X_train_scaled = scaler.fit_transform(X_train)

# テストデータには、
# 学習データから求めた平均と標準偏差を使用する
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 8. PCA による異常検知
# ============================================================

# ------------------------------------------------------------
# 8-1. 累積寄与率 95% となる主成分数を決める
# ------------------------------------------------------------

pca_full = PCA().fit(X_train_scaled)

cum_ratio = np.cumsum(pca_full.explained_variance_ratio_)

n_components = np.searchsorted(cum_ratio, 0.95) + 1


# ------------------------------------------------------------
# 8-2. PCA モデルを学習
# ------------------------------------------------------------

pca = PCA(
    n_components=n_components,
    random_state=SEED,
).fit(X_train_scaled)


# ------------------------------------------------------------
# 8-3. PCA 空間へ射影し、元の空間へ再構成
# ------------------------------------------------------------

train_recon = pca.inverse_transform(pca.transform(X_train_scaled))
test_recon = pca.inverse_transform(pca.transform(X_test_scaled))


# ------------------------------------------------------------
# 8-4. 再構成誤差を異常スコアとする
# ------------------------------------------------------------

train_pca_score = ((X_train_scaled - train_recon) ** 2).mean(axis=1)
test_pca_score = ((X_test_scaled - test_recon) ** 2).mean(axis=1)


# ------------------------------------------------------------
# 8-5. 学習正常データの上位5%を異常とする
# ------------------------------------------------------------

pca_threshold = np.percentile(train_pca_score, 95)

pca_pred = (test_pca_score > pca_threshold).astype(int)


# ============================================================
# 9. kNN による異常検知
# ============================================================

k = 5

knn = NearestNeighbors(
    n_neighbors=k,
    metric="euclidean",
).fit(X_train_scaled)


# ------------------------------------------------------------
# 学習データの異常スコア
# ------------------------------------------------------------
#
# 学習データ自身を検索すると、
# 最も近い点は自分自身になり距離は 0 になる。
#
# そこで k+1 個取得し、
# 最初の自分自身を除外する。

train_dists, _ = knn.kneighbors(
    X_train_scaled,
    n_neighbors=k + 1,
)

train_knn_score = train_dists[:, 1:].mean(axis=1)


# ------------------------------------------------------------
# テストデータの異常スコア
# ------------------------------------------------------------

test_dists, _ = knn.kneighbors(
    X_test_scaled,
    n_neighbors=k,
)

test_knn_score = test_dists.mean(axis=1)


# ------------------------------------------------------------
# 正常学習データの95パーセンタイルを閾値とする
# ------------------------------------------------------------

knn_threshold = np.percentile(train_knn_score, 95)

knn_pred = (test_knn_score > knn_threshold).astype(int)


# ============================================================
# 10. One-Class SVM による異常検知
# ============================================================

ocsvm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05,
).fit(X_train_scaled)


# decision_function は
# 正常領域の内側ほど大きい値を返す。
#
# 今回は
#
#   大きい値 = より異常
#
# に統一したいのでマイナスを付ける。

train_svm_score = -ocsvm.decision_function(X_train_scaled).ravel()
test_svm_score = -ocsvm.decision_function(X_test_scaled).ravel()


svm_threshold = np.percentile(train_svm_score, 95)

svm_pred = (test_svm_score > svm_threshold).astype(int)


# ============================================================
# 11. 評価結果を表示する関数
# ============================================================

def print_evaluation(
    method_name,
    y_true,
    score,
    pred,
    threshold,
    extra_info=None,
):
    """
    異常検知結果をまとめて表示する。
    """

    print()
    print("=" * 60)
    print(method_name)
    print("=" * 60)

    if extra_info is not None:
        print(extra_info)

    print("threshold        :", threshold)
    print("ROC-AUC          :", roc_auc_score(y_true, score))
    print("Average Precision:", average_precision_score(y_true, score))
    print()

    print(
        classification_report(
            y_true,
            pred,
            target_names=["normal", "anomaly"],
        )
    )


# ============================================================
# 12. PCA / kNN / One-Class SVM の評価
# ============================================================

print_evaluation(
    method_name="PCA",
    y_true=y_test,
    score=test_pca_score,
    pred=pca_pred,
    threshold=pca_threshold,
    extra_info=f"n_components     : {n_components}",
)

print_evaluation(
    method_name="kNN",
    y_true=y_test,
    score=test_knn_score,
    pred=knn_pred,
    threshold=knn_threshold,
    extra_info=f"k                : {k}",
)

print_evaluation(
    method_name="One-Class SVM",
    y_true=y_test,
    score=test_svm_score,
    pred=svm_pred,
    threshold=svm_threshold,
)


# ============================================================
# 13. 異常スコア分布を図示
# ============================================================

def plot_score_distribution(
    normal_score,
    anomaly_score,
    threshold,
    title,
):
    """
    正常音と異常音の異常スコア分布を
    ヒストグラムとして保存する。
    """

    # 極端な外れ値によって横軸が広がりすぎないよう、
    # 全スコアの99パーセンタイルを表示上限とする。
    upper = np.percentile(
        np.r_[normal_score, anomaly_score],
        99,
    )

    bins = np.linspace(0, upper, 30)

    plt.figure(figsize=(6, 3))

    plt.hist(
        normal_score,
        bins=bins,
        alpha=0.6,
        label="normal",
    )

    plt.hist(
        anomaly_score,
        bins=bins,
        alpha=0.6,
        label="anomaly",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label="threshold",
    )

    plt.title(title)
    plt.xlabel("anomaly score")
    plt.ylabel("count")
    plt.legend()

    # 例：
    #
    # "One-Class SVM"
    #       ↓
    # "one-class_svm.png"

    filename = title.lower().replace(" ", "_") + ".png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"saved: {filename}")


# ============================================================
# 14. PCA / kNN / One-Class SVM の分布を保存
# ============================================================

plot_score_distribution(
    normal_score=test_pca_score[y_test == 0],
    anomaly_score=test_pca_score[y_test == 1],
    threshold=pca_threshold,
    title="PCA",
)

plot_score_distribution(
    normal_score=test_knn_score[y_test == 0],
    anomaly_score=test_knn_score[y_test == 1],
    threshold=knn_threshold,
    title="kNN",
)

plot_score_distribution(
    normal_score=test_svm_score[y_test == 0],
    anomaly_score=test_svm_score[y_test == 1],
    threshold=svm_threshold,
    title="One-Class SVM",
)



# ============================================================
# 15. Save file information, true labels, scores, and predictions
# ============================================================

import pandas as pd


# Keep only the filename, not the full path
test_files = test_normal_files + test_anomaly_files
test_filenames = [Path(f).name for f in test_files]


results_df = pd.DataFrame({
    "file": test_filenames,
    "y_true": y_test,                 # 0 = normal, 1 = anomaly

    "pca_score": test_pca_score,
    "pca_pred": pca_pred,             # 0 = normal, 1 = anomaly

    "knn_score": test_knn_score,
    "knn_pred": knn_pred,

    "svm_score": test_svm_score,
    "svm_pred": svm_pred,
})


# Save as CSV
output_csv = "toycar_anomaly_results.csv"

results_df.to_csv(
    output_csv,
    index=False,
)


print(f"saved: {output_csv}")

# Check first few rows
print(results_df.head())