"""Render one square favicon per (algorithm x dataset) 2D embedding.

Reuses the exact embedding functions from audit_embeddings.py, then saves each
embedding as a small turbo-colored scatter on the site's dark surface. A
manifest.json lists every file written so the page can pick one at random.

Run:  PYTHONPATH=. manimexp/.venv/bin/python manimexp/make_favicons.py
Out:  assets/favicons/<algo>__<dataset>.png  and  assets/favicons/manifest.json
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manimexp.audit_embeddings import ALGOS, DATASETS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "favicons")
BG = "#131313"   # site background, so the favicon reads as on-brand
PX = 128         # browsers downscale; 128 keeps the cloud crisp at 16-32px


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = []
    for aname, fn in ALGOS:
        for ds in DATASETS:
            try:
                Y, t = fn(ds)
                Y = np.asarray(Y, dtype=float)
                if not np.isfinite(Y).all():
                    raise ValueError("non-finite embedding")
            except Exception as e:  # noqa: BLE001
                print(f"skip {aname}/{ds}: {e}")
                continue

            fig = plt.figure(figsize=(1, 1), dpi=PX)
            ax = fig.add_axes([0, 0, 1, 1])
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            ax.scatter(Y[:, 0], Y[:, 1], c=t, cmap="turbo", s=6, linewidths=0)
            ax.set_aspect("equal", adjustable="datalim")
            ax.margins(0.08)
            ax.axis("off")

            name = f"{aname}__{ds}.png"
            fig.savefig(os.path.join(OUT_DIR, name), facecolor=BG)
            plt.close(fig)
            names.append(name)
            print("wrote", name)

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(names, f)
    print(f"{len(names)} favicons -> {OUT_DIR}")


if __name__ == "__main__":
    main()
