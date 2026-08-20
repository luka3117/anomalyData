# # 異常音の異常検知を体験しよう
# このノートブックでは、ToyADMOS の **ToyCar** データを使って、機械の音から異常を見つける流れを体験します。
# 
# 今回は次の 3 つの方法を試します。
# 
# - **PCA**：正常データをうまく表せない音を異常とみなす
# - **kNN**：正常データから遠い音を異常とみなす
# - **One-Class SVM**：正常データのまとまりから外れる音を異常とみなす
# 
# 講義では、まず **1つの section だけ** に絞って進めます。

# %% [markdown]
# # 1. データセット
# 使用するデータセットは **ToyADMOS** です。小型機械の動作音を集めた、異常音検知の学習用データセットです。
# 
# - データセット配布ページ：Zenodo
# - 参考 GitHub：ToyADMOS-dataset
# - 論文：Koizumi ら, *ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection*, WASPAA 2019
# 
# このノートブックでは、**ToyCar** の一部だけを使います。

# %%
# Colabで使うライブラリをインストール
!pip -q install librosa soundfile scikit-learn tqdm

import glob
import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

# %% [markdown]
# # 2. データをダウンロードする
# Google Drive に作業用フォルダを作り、ToyCar の zip ファイルをダウンロードして展開します。
# 
# 今回は講義用に、**train / test の wav ファイルだけ** を使います。

# %%
from google.colab import drive

drive.mount("/content/drive")

BASE_DIR = "/content/drive/MyDrive/ToyADMOS_work"
RAW_DIR = os.path.join(BASE_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

%cd {RAW_DIR}

# ToyCar をダウンロード
!wget -nc https://zenodo.org/records/3678171/files/dev_data_ToyCar.zip

# 展開
!mkdir -p {BASE_DIR}
!unzip -q -o dev_data_ToyCar.zip -d {BASE_DIR}

# 展開結果の確認
!find {BASE_DIR}/ToyCar -maxdepth 2 -type d | sort

# %% [markdown]
# # 3. 使うファイルを決める
# 講義ではまず **section_00** だけを使います。
# 
# - 学習データ：正常音のみ
# - テストデータ：正常音と異常音

# %%
TRAIN_DIR = os.path.join(BASE_DIR, "ToyCar", "train")
TEST_DIR = os.path.join(BASE_DIR, "ToyCar", "test")

TARGET_SECTION = "id_04"

train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, f"*{TARGET_SECTION}*.wav")))
test_normal_files = sorted(glob.glob(os.path.join(TEST_DIR, f"normal_*{TARGET_SECTION}*.wav")))
test_anomaly_files = sorted(glob.glob(os.path.join(TEST_DIR, f"anomaly_*{TARGET_SECTION}*.wav")))

print("train:", len(train_files))
print("test normal:", len(test_normal_files))
print("test anomaly:", len(test_anomaly_files))

# %% [markdown]
# # 4. 音声から特徴量を作る
# そのままの波形では扱いにくいので、音声を **log-mel spectrogram** に変換します。
# 
# 今回はさらに、各周波数帯について
# 
# - 時間方向の平均
# - 時間方向の標準偏差
# 
# を計算し、1本の音声を **固定長ベクトル** にします。

# %%
def extract_feature(wav_path, sr=16000, n_mels=128, n_fft=1024, hop_length=512):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    feat_mean = logmel.mean(axis=1)
    feat_std = logmel.std(axis=1)
    return np.hstack([feat_mean, feat_std]).astype(np.float32)


def load_features(file_list):
    features = []
    for wav_path in tqdm(file_list):
        features.append(extract_feature(wav_path))
    return np.array(features, dtype=np.float32)

# %% [markdown]
# # 5. 学習データとテストデータを読み込む
# 異常検知では、学習に **正常データだけ** を使うことが多いです。
# 
# ここでも、
# 
# - `X_train`：正常音だけ
# - `X_test`：正常音 + 異常音
# - `y_test`：正常なら 0、異常なら 1
# 
# という形にします。

# %%
X_train = load_features(train_files)
X_test_normal = load_features(test_normal_files)
X_test_anomaly = load_features(test_anomaly_files)

X_test = np.vstack([X_test_normal, X_test_anomaly])
y_test = np.r_[np.zeros(len(X_test_normal), dtype=int),
               np.ones(len(X_test_anomaly), dtype=int)]

print("X_train:", X_train.shape)
print("X_test_normal:", X_test_normal.shape)
print("X_test_anomaly:", X_test_anomaly.shape)
print("X_test:", X_test.shape)

# %% [markdown]
# # 6. 標準化
# 特徴量の尺度をそろえるために標準化します。
# 
# **学習データで平均と標準偏差を求めて、同じ変換をテストデータにも適用する** のが大切です。

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# # 7. PCA による異常検知
# PCA は、正常データを少ない次元で表す方法です。
# 
# ここでは、
# 
# 1. 正常データで PCA を学習
# 2. 元の特徴量を PCA で圧縮してから復元
# 3. **復元誤差** が大きいほど異常らしい
# 
# と考えます。

# %%
# 累積寄与率 95% になる主成分数を選ぶ
pca_full = PCA().fit(X_train_scaled)
cum_ratio = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.searchsorted(cum_ratio, 0.95) + 1

pca = PCA(n_components=n_components, random_state=42).fit(X_train_scaled)

train_recon = pca.inverse_transform(pca.transform(X_train_scaled))
test_recon = pca.inverse_transform(pca.transform(X_test_scaled))

train_pca_score = ((X_train_scaled - train_recon) ** 2).mean(axis=1)
test_pca_score = ((X_test_scaled - test_recon) ** 2).mean(axis=1)

pca_threshold = np.percentile(train_pca_score, 95)
pca_pred = (test_pca_score > pca_threshold).astype(int)

print("=== PCA ===")
print("n_components:", n_components)
print("threshold:", pca_threshold)
print("ROC-AUC:", roc_auc_score(y_test, test_pca_score))
print("Average Precision:", average_precision_score(y_test, test_pca_score))
print(classification_report(y_test, pca_pred, target_names=["normal", "anomaly"]))

# %% [markdown]
# # 8. kNN による異常検知
# kNN では、**正常データの近くにあれば正常、遠ければ異常** と考えます。
# 
# ここでは、各データについて **近い k 個の訓練データまでの距離の平均** を異常スコアにします。

# %%
k = 5

knn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X_train_scaled)

# 学習データは自分自身が最も近くなるので、k+1 個取り最初を除く
train_dists, _ = knn.kneighbors(X_train_scaled, n_neighbors=k + 1)
train_knn_score = train_dists[:, 1:].mean(axis=1)

test_dists, _ = knn.kneighbors(X_test_scaled, n_neighbors=k)
test_knn_score = test_dists.mean(axis=1)

knn_threshold = np.percentile(train_knn_score, 95)
knn_pred = (test_knn_score > knn_threshold).astype(int)

print("=== kNN ===")
print("k:", k)
print("threshold:", knn_threshold)
print("ROC-AUC:", roc_auc_score(y_test, test_knn_score))
print("Average Precision:", average_precision_score(y_test, test_knn_score))
print(classification_report(y_test, knn_pred, target_names=["normal", "anomaly"]))

# %% [markdown]
# # 9. One-Class SVM による異常検知
# One-Class SVM は、正常データが分布する範囲を囲むように学習します。
# 
# `decision_function` は **大きいほど正常らしい** 値なので、ここでは符号を反転して  
# **大きいほど異常らしいスコア** に変換します。

# %%
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_train_scaled)

train_svm_score = -ocsvm.decision_function(X_train_scaled).ravel()
test_svm_score = -ocsvm.decision_function(X_test_scaled).ravel()

svm_threshold = np.percentile(train_svm_score, 95)
svm_pred = (test_svm_score > svm_threshold).astype(int)

print("=== One-Class SVM ===")
print("threshold:", svm_threshold)
print("ROC-AUC:", roc_auc_score(y_test, test_svm_score))
print("Average Precision:", average_precision_score(y_test, test_svm_score))
print(classification_report(y_test, svm_pred, target_names=["normal", "anomaly"]))

# %% [markdown]
# # 10. スコア分布を見てみる
# 正常音と異常音でスコアの分布が分かれていれば、うまく検知できている可能性があります。
# 
# 点線は、**学習用の正常データの 95 パーセンタイル** を使って決めたしきい値です。

# %%
def plot_score_distribution(normal_score, anomaly_score, threshold, title):
    upper = np.percentile(np.r_[normal_score, anomaly_score], 99)
    bins = np.linspace(0, upper, 30)

    plt.figure(figsize=(6, 3))
    plt.hist(normal_score, bins=bins, alpha=0.6, label="normal")
    plt.hist(anomaly_score, bins=bins, alpha=0.6, label="anomaly")
    plt.axvline(threshold, linestyle="--", label="threshold")
    plt.title(title)
    plt.xlabel("anomaly score")
    plt.ylabel("count")
    plt.legend()
    plt.show()


plot_score_distribution(test_pca_score[y_test == 0], test_pca_score[y_test == 1], pca_threshold, "PCA")
plot_score_distribution(test_knn_score[y_test == 0], test_knn_score[y_test == 1], knn_threshold, "kNN")
plot_score_distribution(test_svm_score[y_test == 0], test_svm_score[y_test == 1], svm_threshold, "One-Class SVM")

# %% [markdown]
# # 11. まとめ
# このノートブックでは、機械の正常音だけを使って異常検知を行いました。
# 
# 今回のポイントは次の通りです。
# 
# - 異常検知では **正常データだけで学習する** 場面が多い
# - 音声はそのままではなく、**特徴量に変換してから** 扱う
# - PCA・kNN・One-Class SVM など、教師なし学習の方法で異常度を計算できる
# - しきい値を決めることで、正常 / 異常の判定ができる
# 
# 講義では、結果を見比べながら「どの方法がどんな異常を見つけやすいか」を考えてみましょう。

# %%



