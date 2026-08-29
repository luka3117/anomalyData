# pure_notes_mel_spectrogram.py
# Generate pure sine notes from command-line arguments
# and display/save their Mel spectrogram.

import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
sr = 16000
duration = 3.0
n_mels = 128
n_fft = 1024
hop_length = 512


# ------------------------------------------------------------
# Read note names from command line
# Example:
#   python pure_notes_mel_spectrogram.py C4 E4 G4
# ------------------------------------------------------------
if len(sys.argv) < 2:
    notes = ["C4", "E4", "G4"]
else:
    notes = sys.argv[1:]


# ------------------------------------------------------------
# Time axis
# ------------------------------------------------------------
t = np.linspace(
    0,
    duration,
    int(sr * duration),
    endpoint=False,
)


# ------------------------------------------------------------
# Generate pure sine waves
# ------------------------------------------------------------
y = np.zeros_like(t)

print("Notes:")

for note in notes:
    freq = librosa.note_to_hz(note)

    print(f"  {note}: {freq:.2f} Hz")

    y += np.sin(2 * np.pi * freq * t)


# Avoid clipping
y /= len(notes)


# ------------------------------------------------------------
# Mel spectrogram
# ------------------------------------------------------------
mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
)

logmel = librosa.power_to_db(
    mel,
    ref=np.max,
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    logmel,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="mel",
)

plt.colorbar(format="%+2.0f dB")

plt.title(
    "Pure sine waves: " + " + ".join(notes)
)

plt.xlabel("Time")
plt.ylabel("Mel frequency")

plt.tight_layout()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
output_name = "mel_" + "_".join(notes) + ".png"

plt.savefig(
    output_name,
    dpi=200,
    bbox_inches="tight",
)

print("Saved:", output_name)

plt.show()