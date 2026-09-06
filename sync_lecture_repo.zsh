#!/bin/zsh

DEST="../anomaly_seminar"

mkdir -p "$DEST"

rsync -a hazelnutData/ "$DEST/hazelnutData/"
rsync -a soundData/ "$DEST/soundData/"

rsync -a \
  toySoundCheck_final.py \
  toySoundCheck_final.ipynb \
  nutcheck_easy_clean_jp.py \
  nutcheck_apply_selected.py \
  show_mel_spectrogram.py \
  pure_notes_mel_spectrogram.py \
  pyproject.toml \
  "$DEST/"

echo "Synced → $DEST"