#!/usr/bin/env bash
# Render the MDS / LLE / Laplacian / Kernel PCA walkthroughs at 1080p/60fps and
# copy each combined walkthrough video (remuxed with +faststart so it streams in
# the browser) plus a poster frame into assets/manim/<algo>/.
#
# The web player (js/manifold_isomap.js) plays the single combined walkthrough.mp4
# per algorithm and uses chapter start times, so only the combined video is
# needed (not the per-step section clips).
#
# Usage:  ./dev/manimexp/render_walkthroughs.sh            # all
#         ./dev/manimexp/render_walkthroughs.sh mds lle    # a subset
#
# Full quality is slow (1000 points, 1080p, 60fps).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
export PYTHONPATH=dev
PY=dev/manimexp/.venv/bin/manim
VID_DIR=media/videos/walkthrough/1080p60

# Remux the combined scene video with faststart and write a mid-clip poster.
emit() {
  local scene="$1" out_mp4="$2" out_png="$3"
  local src="$VID_DIR/${scene}.mp4"
  mkdir -p "$(dirname "$out_mp4")"
  ffmpeg -y -loglevel error -i "$src" -c copy -movflags +faststart "$out_mp4"
  local dur mid
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out_mp4")
  mid=$(awk -v d="$dur" 'BEGIN{printf "%.2f", d*0.06}')   # just past the opening fade
  ffmpeg -y -loglevel error -ss "$mid" -i "$out_mp4" -frames:v 1 -q:v 2 "$out_png"
}

render_one() {
  local file="$1" scene="$2"
  $PY -qh --fps 60 --save_sections --disable_caching "dev/manimexp/${file}" "$scene"
}

want() { [ "$#" -eq 0 ] && return 0; for a in "$@"; do [ "$a" = "$TARGET" ] && return 0; done; return 1; }

ALL_ARGS=("$@")
TARGET=isomap; if want "${ALL_ARGS[@]}"; then
  render_one isomap/walkthrough.py IsomapWalkthrough
  # Isomap predates the poster.png convention; the player uses step-1.png.
  emit IsomapWalkthrough assets/manim/isomap/walkthrough.mp4 assets/manim/isomap/step-1.png
fi
TARGET=pca; if want "${ALL_ARGS[@]}"; then
  render_one pca/walkthrough.py PCAWalkthrough
  # PCA predates the poster.png convention; the player uses walkthrough.png.
  emit PCAWalkthrough assets/manim/pca/walkthrough.mp4 assets/manim/pca/walkthrough.png
fi
TARGET=mds; if want "${ALL_ARGS[@]}"; then
  render_one mds/walkthrough.py MDSWalkthrough
  emit MDSWalkthrough assets/manim/mds/walkthrough.mp4 assets/manim/mds/poster.png
fi
TARGET=lle; if want "${ALL_ARGS[@]}"; then
  render_one lle/walkthrough.py LLEWalkthrough
  emit LLEWalkthrough assets/manim/lle/walkthrough.mp4 assets/manim/lle/poster.png
fi
TARGET=laplacian; if want "${ALL_ARGS[@]}"; then
  render_one laplacian/walkthrough.py LaplacianWalkthrough
  emit LaplacianWalkthrough assets/manim/laplacian/walkthrough.mp4 assets/manim/laplacian/poster.png
fi
TARGET=kpca; if want "${ALL_ARGS[@]}"; then
  for k in rbf polynomial linear; do
    MFI_KERNEL="$k" render_one kpca/walkthrough.py KPCAWalkthrough
    emit KPCAWalkthrough "assets/manim/kpca/walkthrough-${k}.mp4" "assets/manim/kpca/poster-${k}.png"
  done
fi
echo "Done. Wrote combined walkthroughs + posters under assets/manim/."
