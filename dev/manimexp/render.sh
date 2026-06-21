#!/usr/bin/env bash
# Render the Isomap walkthrough at 60fps/1080p, split into per-step section MP4s,
# and copy them plus poster frames to assets/manim/isomap/.
#
# Usage:
#   ./dev/manimexp/render.sh
#
# The full-quality render is slow (1000 points, 1080p, 60fps).
# For a fast preview use:
#   MFI_N=120 PYTHONPATH=dev dev/manimexp/.venv/bin/manim -ql --fps 30 --save_sections dev/manimexp/isomap/walkthrough.py IsomapWalkthrough
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
export PYTHONPATH=dev
PY=dev/manimexp/.venv/bin/manim
OUT=assets/manim/isomap
mkdir -p "$OUT"

$PY -qh --fps 60 --save_sections --disable_caching dev/manimexp/isomap/walkthrough.py IsomapWalkthrough

SECTIONS_DIR=$(find media/videos/walkthrough -type d -name sections | head -1)
i=1
for name in step-1-raw step-2-knn step-3-geodesic step-4-double-center step-5-eigendecomp step-6-embedding; do
  src=$(find "$SECTIONS_DIR" -name "*${name}*.mp4" | head -1)
  # Remux with +faststart (moov atom at the front) so the clips stream and play
  # progressively in the browser. Manim's raw output puts moov at the end, which
  # stops HTML5 video from starting over a simple HTTP server.
  ffmpeg -y -loglevel error -i "$src" -c copy -movflags +faststart "$OUT/step-${i}.mp4"
  # Poster: a frame halfway through the clip, past any opening fade.
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/step-${i}.mp4")
  mid=$(awk -v d="$dur" 'BEGIN{printf "%.2f", d*0.5}')
  ffmpeg -y -loglevel error -ss "$mid" -i "$OUT/step-${i}.mp4" -frames:v 1 -q:v 2 "$OUT/step-${i}.png"
  i=$((i + 1))
done
echo "Rendered ${OUT}/step-1..6.mp4 (faststart) and posters."
