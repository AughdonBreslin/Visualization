#!/usr/bin/env bash
# Render 480p/30fps preview clips for every algorithm across every non-swiss
# dataset, plus a poster frame each. n defaults to 1000 (override with MFI_N).
#
# Output paths match what js/manifold_isomap.js loads:
#   - isomap  -> assets/manim/isomap/drafts/<dataset>.{mp4,png}  (player's primary
#                path), and mirrored to assets/manim/preview480/isomap/ (fallback).
#   - others  -> assets/manim/preview480/<algo>/<dataset>.{mp4,png}
#
# Deliberately NOT set -e: a single failed render must not abort the whole batch.
# Each failure is logged and the loop continues.
#
# Usage:  ./manimexp/render_previews.sh            # all algos, all datasets
#         ./manimexp/render_previews.sh mds lle    # a subset of algo units
set -uo pipefail
cd "$(dirname "$0")/.."          # repo root
export PYTHONPATH=.
PY=manimexp/.venv/bin/manim
# Quality is overridable: defaults to 480p/30fps; pass MFI_QFLAG=-qh MFI_FPS=60
# MFI_VIDDIR=media/videos/walkthrough/1080p60 for the production 1080p sweep.
QFLAG="${MFI_QFLAG:--ql}"
FPS="${MFI_FPS:-30}"
# Per-worker media dir isolates each parallel render (own video/Tex trees), so
# concurrent renders of the same scene never collide. VID is derived from it.
MEDIADIR="${MFI_MEDIADIR:-media}"
VID="${MFI_VIDDIR:-$MEDIADIR/videos/walkthrough/480p30}"
# Skip a clip if its output already exists at this height (so a parallel relaunch
# does not redo finished work). Empty disables skipping.
SKIP_H="${MFI_SKIP_HEIGHT:-}"
N="${MFI_N:-1000}"
SEED="${MFI_SEED:-0}"

# Dataset list: override with MFI_DATASETS="s_curve helix ..." (space-separated).
if [ -n "${MFI_DATASETS:-}" ]; then
  read -ra DATASETS <<< "$MFI_DATASETS"
else
  DATASETS=(s_curve twin_peaks saddle cylinder severed_sphere helix \
            trefoil_knot toroidal_helix spiral_disk full_sphere hilbert clusters_3d)
fi

# Each unit: "name|scene_file|SceneClass|out_subdir|kernel"
UNITS=(
  "isomap|isomap/walkthrough.py|IsomapWalkthrough|isomap/drafts|"
  "pca|pca/walkthrough.py|PCAWalkthrough|preview480/pca|"
  "mds|mds/walkthrough.py|MDSWalkthrough|preview480/mds|"
  "lle|lle/walkthrough.py|LLEWalkthrough|preview480/lle|"
  "laplacian|laplacian/walkthrough.py|LaplacianWalkthrough|preview480/laplacian|"
  "kpca_rbf|kpca/walkthrough.py|KPCAWalkthrough|preview480/kpca_rbf|rbf"
  "kpca_polynomial|kpca/walkthrough.py|KPCAWalkthrough|preview480/kpca_polynomial|polynomial"
  "kpca_linear|kpca/walkthrough.py|KPCAWalkthrough|preview480/kpca_linear|linear"
)

selected() {  # unit_name -> 0 if it should run
  [ "${#WANT[@]}" -eq 0 ] && return 0
  for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}
WANT=("$@")

# Pre-generate point files for every dataset at this N (idempotent, deterministic).
echo "=== generating points (N=$N) ==="
for ds in "${DATASETS[@]}"; do
  node manimexp/isomap/gen_points.mjs "$ds" "$N" "$SEED" \
    "manimexp/isomap/points/${ds}_${N}.json" >/dev/null 2>&1 \
    && echo "  points: $ds" || echo "  POINTS FAILED: $ds"
done

emit() {  # SceneClass  out_mp4_base(no ext)
  local scene="$1" base="$2" src="$VID/$1.mp4"
  [ -f "$src" ] || { echo "   MISSING manim output $src"; return 1; }
  mkdir -p "$(dirname "$base")"
  ffmpeg -y -loglevel error -i "$src" -c copy -movflags +faststart "${base}.mp4" || return 1
  local dur mid
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${base}.mp4")
  mid=$(awk -v d="$dur" 'BEGIN{printf "%.2f", d*0.5}')
  ffmpeg -y -loglevel error -ss "$mid" -i "${base}.mp4" -frames:v 1 -q:v 2 "${base}.png"
}

total=0; ok=0; fail=0
for unit in "${UNITS[@]}"; do
  IFS='|' read -r name file scene outdir kernel <<< "$unit"
  selected "$name" || continue
  echo ""
  echo "########## UNIT $name  ($scene)  ##########"
  for ds in "${DATASETS[@]}"; do
    total=$((total+1))
    base="assets/manim/${outdir}/${ds}"
    if [ -n "$SKIP_H" ] && [ -f "${base}.mp4" ] && \
       [ "$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "${base}.mp4" 2>/dev/null)" = "$SKIP_H" ]; then
      ok=$((ok+1)); echo "[$(date +%H:%M:%S)] $name / $ds  SKIP (${SKIP_H}p exists)"; continue
    fi
    echo "[$(date +%H:%M:%S)] $name / $ds ..."
    # Use `env` so the expansion-produced MFI_KERNEL=... is honored as an
    # assignment (bash would otherwise treat it as a command word and fail).
    if env MFI_N="$N" MFI_DATASET="$ds" ${kernel:+MFI_KERNEL="$kernel"} \
        "$PY" "$QFLAG" --fps "$FPS" --media_dir "$MEDIADIR" --disable_caching "manimexp/$file" "$scene" >/dev/null 2>&1; then
      if emit "$scene" "assets/manim/${outdir}/${ds}"; then
        ok=$((ok+1)); echo "   ok -> assets/manim/${outdir}/${ds}.mp4"
        # Mirror isomap drafts into preview480/isomap for the fallback path.
        if [ "$name" = "isomap" ]; then
          mkdir -p assets/manim/preview480/isomap
          cp -f "assets/manim/${outdir}/${ds}.mp4" "assets/manim/preview480/isomap/${ds}.mp4"
          cp -f "assets/manim/${outdir}/${ds}.png" "assets/manim/preview480/isomap/${ds}.png"
        fi
      else
        fail=$((fail+1)); echo "   EMIT FAILED $name/$ds"
      fi
    else
      fail=$((fail+1)); echo "   RENDER FAILED $name/$ds"
    fi
  done
done

echo ""
echo "=== previews done: $ok ok, $fail failed, $total total ($(date +%H:%M:%S)) ==="
