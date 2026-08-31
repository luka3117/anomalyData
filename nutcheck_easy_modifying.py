#!/usr/bin/env python
# coding: utf-8

# In[58]:


#!/usr/bin/env python
# coding: utf-8

# ============================================================
# nutCheck_easy.py
# Minimal convolutional Autoencoder example for hazelnut images
#
# Expected folder structure:
#
#   current_folder/
#   ├── nutCheck_easy.py
#   └── hazelnutData/
#       ├── train/
#       │   └── good/
#       └── test/
#           ├── good/
#           ├── crack/
#           ├── cut/
#           ├── hole/
#           └── print/
#
# Run this file from current_folder:
#
#   python nutCheck_easy.py
#
# Required packages:
#   torch torchvision pillow matplotlib numpy scikit-learn
# ============================================================



# In[59]:


from pathlib import Path                  # Path handling for files/folders
import random                            # Randomly select example images

import matplotlib.pyplot as plt          # Plot images and graphs
import numpy as np                       # Numerical arrays and calculations

import torch                             # Main PyTorch package
import torch.nn as nn                    # Neural-network layers
import torch.optim as optim              # Optimization algorithms

from PIL import Image                    # Open and handle image files

from sklearn.metrics import roc_auc_score # Evaluate anomaly-detection performance

from torch.utils.data import DataLoader, Dataset
# Dataset    : defines how to get one observation
# DataLoader : supplies observations in batches

from torchvision import transforms       # Image preprocessing tools


# In[ ]:


# ============================================================
# 1. Basic settings
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# In[61]:


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device =", DEVICE)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("device =", DEVICE)
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20


# In[62]:


# ============================================================
# 2. Data folder
# ============================================================

# Run this Python file from the folder that contains hazelnutData/.

# hazelnutData/
# ├── train/
# │   └── good/      # 391 images
# └── test/
#     ├── good/      # 40 images
#     ├── crack/     # 18 images
#     ├── cut/       # 17 images
#     ├── hole/      # 18 images
#     └── print/     # 17 images

# test/
# ├── good/   -> normal / 正常
# ├── crack/  -> anomaly: crack / 異常：ひび割れ
# ├── cut/    -> anomaly: cut / 異常：切り傷・切断傷
# ├── hole/   -> anomaly: hole / 異常：穴・くぼみ
# └── print  -> anomaly: abnormal surface mark / 異常：表面の異常な模様・斑点


DATA_ROOT = Path.cwd() / "hazelnutData"

TRAIN_GOOD_DIR = DATA_ROOT / "train" / "good"

TEST_DIR = DATA_ROOT / "test"
TEST_GOOD_DIR = TEST_DIR / "good"

print(Path(*TEST_GOOD_DIR.parts[-3:]))
print(Path(*TEST_DIR.parts[-2:]))



# In[63]:


# ============================================================
# 3. Collect image paths
# ============================================================

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def image_files(folder):
    # Get all items directly inside the folder,
    # keep only image files with allowed extensions,
    # and return the paths in sorted order.
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
train_good_paths = image_files(TRAIN_GOOD_DIR)
test_good_paths = image_files(TEST_GOOD_DIR)




# In[64]:


# Find anomaly-type folders such as:
# crack/, cut/, hole/, print/
# "good" is excluded because it is normal data.
anomaly_dirs = sorted(
    folder for folder in TEST_DIR.iterdir()
    if folder.is_dir() and folder.name != "good"
)

# Combine all anomaly images into one list.
test_anomaly_paths = []

for folder in anomaly_dirs:
    for path in image_files(folder):
        test_anomaly_paths.append(path)


# Check how many images were collected.
print("train good  :", len(train_good_paths))
print("test good   :", len(test_good_paths))
print("test anomaly:", len(test_anomaly_paths))



# In[65]:


# ============================================================
# 4. Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


# In[66]:


class HazelnutDataset(Dataset):
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

# Training data: NORMAL images only
# Create label 0 for all training images.
train_dataset = HazelnutDataset(
    train_good_paths,
    [0] * len(train_good_paths),
)



# Test data: NORMAL + ANOMALY images
# Combine all test-image paths into one list.
test_paths = test_good_paths + test_anomaly_paths

# Create test labels:
# 0 = normal
# 1 = anomaly
test_labels = (
    [0] * len(test_good_paths)
    + [1] * len(test_anomaly_paths)
)

# Create the test Dataset object.
test_dataset = HazelnutDataset(
    test_paths,
    test_labels,
)



# In[67]:


# Create the training DataLoader.
# It sends BATCH_SIZE images at a time to the model.
# shuffle=True changes the image order in each epoch.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# Same training images, but no shuffle.
# This loader is used only to calculate training anomaly scores.
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


# In[ ]:


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

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)


# In[ ]:


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

        # Input image itself is the target.
        loss = criterion(reconstructed, images)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    epoch_loss = total_loss / len(train_dataset)
    loss_history.append(epoch_loss)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"loss = {epoch_loss:.6f}"
    )


# In[ ]:


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



# In[74]:


# ============================================================
# 8. Reconstruction
# ============================================================


def reconstruct(image_path):
    image = Image.open(image_path).convert("RGB")

    # Add batch dimension:
    # 1 x 128 x 128  ->  1 x 1 x 128 x 128
    x = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():
        x_hat = model(x)

    input_image = x[0, 0].cpu().numpy()
    reconstructed = x_hat[0, 0].cpu().numpy()

    # Pixel-wise reconstruction error
    error_map = (reconstructed - input_image) ** 2

    return input_image, reconstructed, error_map



def show_reconstruction(image_path):
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

    plt.tight_layout()
    plt.show()


print("normal example")
show_reconstruction(random.choice(test_good_paths))
show_reconstruction(test_good_paths[3])

print("anomaly example")
show_reconstruction(random.choice(test_anomaly_paths))
show_reconstruction(test_anomaly_paths[3])




# In[80]:


# len(test_anomaly_paths)


# In[81]:


# ============================================================
# 9. Simple anomaly score
# ============================================================

# EASY VERSION:
# Use the mean reconstruction error of all pixels.
#
# anomaly score = mean[(x_hat - x)^2]
#
# The full nutCheck.py uses a more practical score:
#   - ignore an 8-pixel border
#   - select the largest 5% pixel errors
#   - use their mean as the anomaly score
#
# That more advanced version is intentionally omitted here.


def calculate_scores(loader):
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(DEVICE)
            reconstructed = model(images)

            error = (reconstructed - images).pow(2)

            # One MSE anomaly score for each image.
            image_scores = error.mean(dim=(1, 2, 3))

            all_scores.extend(image_scores.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.asarray(all_scores), np.asarray(all_labels)


train_scores, _ = calculate_scores(train_score_loader)
test_scores, y_true = calculate_scores(test_loader)



# In[85]:


# print(train_scores)
print(len(test_scores))
print(len(train_scores))
# print(y_true)


# In[86]:


# ============================================================
# 10. Threshold
# ============================================================

# Choose the 95th percentile of NORMAL TRAINING scores.
threshold = np.percentile(train_scores, 95)

# 0 = normal, 1 = anomaly

y_pred = (test_scores >= threshold).astype(int)

print()
print(f"threshold          = {threshold:.6f}")
print(f"normal mean score  = {test_scores[y_true == 0].mean():.6f}")
print(f"anomaly mean score = {test_scores[y_true == 1].mean():.6f}")


# ============================================================
# 11. Very simple evaluation
# ============================================================

accuracy = (y_pred == y_true).mean()
roc_auc = roc_auc_score(y_true, test_scores)

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
# End
# ============================================================

# This EASY version intentionally omits:
#
# - detailed file/folder safety checks
# - top 5% reconstruction-error score
# - ignored 8-pixel image border
# - classification_report
# - confusion matrix
# - defect-type summaries
# - ground-truth masks
# - pixel-level ROC-AUC
#
# See the original nutCheck.py for those extensions.




# In[88]:


from scipy.stats import gaussian_kde

plt.figure(figsize=(7, 4))

# Density histograms
plt.hist(
    normal_scores,
    bins=25,
    density=True,
    alpha=0.4,
    label="normal",
)

plt.hist(
    anomaly_scores,
    bins=25,
    density=True,
    alpha=0.4,
    label="anomaly",
)

# Smooth density curves
x_min = min(normal_scores.min(), anomaly_scores.min())
x_max = max(normal_scores.max(), anomaly_scores.max())
x_grid = np.linspace(x_min, x_max, 300)

normal_kde = gaussian_kde(normal_scores)
anomaly_kde = gaussian_kde(anomaly_scores)

plt.plot(
    x_grid,
    normal_kde(x_grid),
    label="normal KDE",
)

plt.plot(
    x_grid,
    anomaly_kde(x_grid),
    label="anomaly KDE",
)

# Threshold
plt.axvline(
    threshold,
    linestyle="--",
    label="threshold",
)

plt.xlabel("Anomaly score")
plt.ylabel("Density")
plt.title("Distribution of anomaly scores")
plt.legend()
plt.tight_layout()
plt.show()

