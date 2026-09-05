#!/usr/bin/env python
# coding: utf-8

"""
Hazelnut image anomaly detection with a convolutional Autoencoder.

Training:
    normal ("good") images only

Testing:
    normal + anomaly images

Anomaly score:
    mean pixel-wise reconstruction error

Expected folder structure
-------------------------
current_folder/
├── nutcheck_easy_clean.py
└── hazelnutData/
    ├── train/
    │   └── good/
    └── test/
        ├── good/
        ├── crack/
        ├── cut/
        ├── hole/
        └── print/

Required packages
-----------------
torch torchvision pillow matplotlib numpy scikit-learn scipy
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics import roc_auc_score


# ============================================================
# 1. Basic settings
# ============================================================

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Use CUDA on supported Windows/Linux machines,
# MPS on Apple Silicon Macs, otherwise CPU.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("device =", DEVICE)


# ============================================================
# 2. Data folders
# ============================================================

# Run this file from the folder that contains hazelnutData/.
#
# hazelnutData/
# ├── train/
# │   └── good/
# └── test/
#     ├── good/      -> normal
#     ├── crack/     -> anomaly
#     ├── cut/       -> anomaly
#     ├── hole/      -> anomaly
#     └── print/     -> anomaly

DATA_ROOT = Path.cwd() / "hazelnutData"

TRAIN_GOOD_DIR = DATA_ROOT / "train" / "good"
TEST_DIR = DATA_ROOT / "test"
TEST_GOOD_DIR = TEST_DIR / "good"

if not TRAIN_GOOD_DIR.exists():
    raise FileNotFoundError(f"Training folder not found: {TRAIN_GOOD_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

print("data folder :", DATA_ROOT)
print("train folder:", TRAIN_GOOD_DIR)
print("test folder :", TEST_DIR)


# ============================================================
# 3. Collect image paths
# ============================================================

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def image_files(folder):
    """Return image files directly inside folder in sorted order."""
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


train_good_paths = image_files(TRAIN_GOOD_DIR)
test_good_paths = image_files(TEST_GOOD_DIR)

# Find anomaly folders such as crack/, cut/, hole/, print/.
anomaly_dirs = sorted(
    folder for folder in TEST_DIR.iterdir()
    if folder.is_dir() and folder.name != "good"
)

# Combine all anomaly images into one list.
test_anomaly_paths = [
    path
    for folder in anomaly_dirs
    for path in image_files(folder)
]

print("\n=== Number of images ===")
print("train good  :", len(train_good_paths))
print("test good   :", len(test_good_paths))
print("test anomaly:", len(test_anomaly_paths))


# ============================================================
# 4. Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),      # resize to 128 x 128
    transforms.Grayscale(num_output_channels=1), # RGB -> grayscale
    transforms.ToTensor(),                       # image -> tensor [0, 1]
])


class HazelnutDataset(Dataset):
    """Dataset that returns (image, label, path)."""

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


# Training data: normal images only.
train_dataset = HazelnutDataset(
    train_good_paths,
    [0] * len(train_good_paths),
)

# Test data: normal + anomaly.
test_paths = test_good_paths + test_anomaly_paths

# 0 = normal, 1 = anomaly
test_labels = (
    [0] * len(test_good_paths)
    + [1] * len(test_anomaly_paths)
)

test_dataset = HazelnutDataset(
    test_paths,
    test_labels,
)


# DataLoader for model training.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Same training images without shuffle.
# Used only when calculating training anomaly scores.
train_score_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

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

        # Encoder:
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

        # Decoder:
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

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n=== Model ===")
print(model)


# ============================================================
# 6. Train using normal images only
# ============================================================

loss_history = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, _, _ in train_loader:
        images = images.to(DEVICE)

        optimizer.zero_grad()

        reconstructed = model(images)

        # Autoencoder target = input image itself.
        loss = criterion(reconstructed, images)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    epoch_loss = total_loss / len(train_dataset)
    loss_history.append(epoch_loss)

    print(f"Epoch {epoch + 1:02d}/{EPOCHS}  loss = {epoch_loss:.6f}")


# ============================================================
# 7. Plot training loss
# ============================================================

plt.figure(figsize=(6, 3.5))
plt.plot(range(1, EPOCHS + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Training progress")
plt.tight_layout()
plt.show()


# ============================================================
# 8. Reconstruction examples
# ============================================================

def reconstruct(image_path):
    """Return input image, reconstructed image, and pixel-wise error map."""
    image = Image.open(image_path).convert("RGB")

    # 1 x 128 x 128 -> 1 x 1 x 128 x 128
    x = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():
        x_hat = model(x)

    input_image = x[0, 0].cpu().numpy()
    reconstructed = x_hat[0, 0].cpu().numpy()

    # Pixel-wise squared reconstruction error.
    error_map = (reconstructed - input_image) ** 2

    return input_image, reconstructed, error_map


def show_reconstruction(image_path):
    """Display input, reconstruction, and reconstruction-error map."""
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

    plt.suptitle(Path(image_path).name, fontsize=10)
    plt.tight_layout()
    plt.show()


print("\n=== Normal examples ===")
show_reconstruction(random.choice(test_good_paths))
show_reconstruction(test_good_paths[3])

print("\n=== Anomaly examples ===")
show_reconstruction(random.choice(test_anomaly_paths))
show_reconstruction(test_anomaly_paths[3])


# ============================================================
# 9. Anomaly score
# ============================================================

# Simple score used in this teaching version:
#
#     anomaly score = mean[(x_hat - x)^2]
#
# That is, one mean reconstruction-error value for each image.
#
# A more practical alternative could use only the largest pixel errors
# or ignore image borders, but those extensions are omitted here.


def calculate_scores(loader):
    """Calculate one reconstruction-error anomaly score per image."""
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(DEVICE)
            reconstructed = model(images)

            error = (reconstructed - images).pow(2)

            # Mean over channel, height, and width.
            image_scores = error.mean(dim=(1, 2, 3))

            all_scores.extend(image_scores.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.asarray(all_scores), np.asarray(all_labels)


train_scores, _ = calculate_scores(train_score_loader)
test_scores, y_true = calculate_scores(test_loader)

print("\n=== Number of scores ===")
print("train:", len(train_scores))
print("test :", len(test_scores))


# ============================================================
# 10. Threshold
# ============================================================

# Use the 95th percentile of NORMAL TRAINING scores.
threshold = np.percentile(train_scores, 95)

# 0 = normal, 1 = anomaly
y_pred = (test_scores >= threshold).astype(int)

print("\n=== Threshold and mean scores ===")
print(f"threshold          = {threshold:.6f}")
print(f"normal mean score  = {test_scores[y_true == 0].mean():.6f}")
print(f"anomaly mean score = {test_scores[y_true == 1].mean():.6f}")


# ============================================================
# 11. Simple evaluation
# ============================================================

accuracy = (y_pred == y_true).mean()
roc_auc = roc_auc_score(y_true, test_scores)

print("\n=== Evaluation ===")
print(f"accuracy = {accuracy:.4f}")
print(f"ROC-AUC  = {roc_auc:.4f}")


# ============================================================
# 12. Score distribution
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
plt.show()


# ============================================================
# 13. Score distribution with KDE
# ============================================================

plt.figure(figsize=(7, 4))

# Density histograms.
plt.hist(normal_scores, bins=25, density=True, alpha=0.4, label="normal")
plt.hist(anomaly_scores, bins=25, density=True, alpha=0.4, label="anomaly")

# Smooth kernel-density curves.
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
plt.show()


# ============================================================
# End
# ============================================================

# This easy version intentionally omits:
#
# - top 5% reconstruction-error score
# - ignored image borders
# - classification_report
# - confusion matrix
# - defect-type summaries
# - ground-truth masks
# - pixel-level ROC-AUC
