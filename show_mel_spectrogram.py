# show_mel_spectrogram.py
# Display a Mel-spectrogram for one WAV file

import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np



wav_path = sys.argv[1]

sr = 16000
n_mels = 128
n_fft = 1024
hop_length = 512

# Load WAV
y, _ = librosa.load(wav_path, sr=sr, mono=True)

# Mel-spectrogram
mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
)

# Convert power to dB
logmel = librosa.power_to_db(mel, ref=np.max)

# Plot
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    logmel,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="mel",
)

plt.colorbar(format="%+2.0f dB")
plt.xlabel("Time")
plt.ylabel("Mel frequency")
plt.title("Mel-spectrogram")
plt.tight_layout()

from pathlib import Path

wav_path = Path(sys.argv[1])
output_path = wav_path.with_suffix(".png")

plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()