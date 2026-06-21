#!/usr/bin/env bash
# Render a draft Isomap walkthrough for one dataset.
#   ./dev/manimexp/render_dataset.sh <datasetId> [N] [seed]
# Dumps the dataset's points (via the JS generators), renders at draft quality
# (480p/30), and writes a faststart clip + poster to assets/manim/isomap/drafts/.
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
DS="$1"; N="${2:-400}"; SEED="${3:-0}"
export PYTHONPATH=dev
MANIM=dev/manimexp/.venv/bin/manim
DRAFTS=assets/manim/isomap/drafts
mkdir -p dev/manimexp/isomap/points "$DRAFTS"

node dev/manimexp/isomap/gen_points.mjs "$DS" "$N" "$SEED" "dev/manimexp/isomap/points/${DS}_${N}.json"

MFI_DATASET="$DS" MFI_N="$N" "$MANIM" -ql --fps 30 --disable_caching \
  dev/manimexp/isomap/walkthrough.py IsomapWalkthrough

SRC=media/videos/walkthrough/480p30/IsomapWalkthrough.mp4
ffmpeg -y -loglevel error -i "$SRC" -c copy -movflags +faststart "${DRAFTS}/${DS}.mp4"
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${DRAFTS}/${DS}.mp4")
mid=$(awk -v d="$dur" 'BEGIN{printf "%.2f", d*0.5}')
ffmpeg -y -loglevel error -ss "$mid" -i "${DRAFTS}/${DS}.mp4" -frames:v 1 -q:v 2 "${DRAFTS}/${DS}.png"
echo "DONE ${DS} -> ${DRAFTS}/${DS}.mp4 (${dur}s)"
