"""Headless embedding audit: compute every (algorithm x dataset) 2D embedding
using the exact same data.py helpers the manim walkthroughs use, and save a
thumbnail grid colored by the manifold parameter t. No video rendering.

Run:  PYTHONPATH=. manimexp/.venv/bin/python manimexp/audit_embeddings.py
Out:  /tmp/embedding_audit_<part>.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manimexp.isomap.data import (
    build_dataset, load_dataset_points, double_center_squared, top2_eig,
    knn_graph, heat_affinity, graph_laplacian, bottom2_eig,
    lle_weights, lle_matrix, kernel_matrix, center_kernel, top2_kernel_embed,
)

N = 1000
SEED = 0
K = 8
SIGMA = 3.0

DATASETS = [
    "swiss_roll", "s_curve", "twin_peaks", "saddle", "cylinder",
    "severed_sphere", "helix", "trefoil_knot", "toroidal_helix",
    "spiral_disk", "full_sphere", "hilbert", "clusters_3d",
]


def emb_isomap(ds):
    d = build_dataset(n=N, k=K, seed=SEED, dataset=ds)
    return d["embedding"], d["t"]


def emb_mds(ds):
    d = load_dataset_points(ds, N, seed=SEED)
    pts, t = d["points"], d["t"]
    order = np.argsort(t); pts, t = pts[order], t[order]
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    B = double_center_squared(D)
    eigvals, eigvecs = top2_eig(B)
    return eigvecs * np.sqrt(np.maximum(eigvals, 0.0)), t


def emb_pca(ds):
    d = load_dataset_points(ds, N, seed=SEED)
    pts, t = d["points"], d["t"]
    Xc = pts - pts.mean(axis=0)
    C = (Xc.T @ Xc) / max(1, pts.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    return Xc @ evecs[:, order][:, :2], t


def emb_lle(ds):
    d = load_dataset_points(ds, N, seed=SEED)
    pts, t = d["points"], d["t"]
    W = lle_weights(pts, k=K, reg=1e-3)
    M = lle_matrix(W)
    vecs, _ = bottom2_eig(M)
    return vecs, t


def emb_laplacian(ds):
    d = load_dataset_points(ds, N, seed=SEED)
    pts, t = d["points"], d["t"]
    adj, edges = knn_graph(pts, k=K)
    W = heat_affinity(pts, edges, sigma=SIGMA)
    L, _ = graph_laplacian(W)
    vecs, _ = bottom2_eig(L)
    return vecs, t


def emb_kpca(ds, kernel):
    d = load_dataset_points(ds, N, seed=SEED)
    pts, t = d["points"], d["t"]
    Km = kernel_matrix(pts, kernel, gamma=1.0, degree=3, constant=1.0)
    Kc = center_kernel(Km)
    Y, _, _ = top2_kernel_embed(Kc)
    return Y, t


ALGOS = [
    ("isomap", emb_isomap),
    ("pca", emb_pca),
    ("mds", emb_mds),
    ("lle", emb_lle),
    ("laplacian", emb_laplacian),
    ("kpca_rbf", lambda ds: emb_kpca(ds, "rbf")),
    ("kpca_poly", lambda ds: emb_kpca(ds, "polynomial")),
    ("kpca_linear", lambda ds: emb_kpca(ds, "linear")),
]


def main():
    ncols = len(DATASETS)
    for aname, fn in ALGOS:
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.6, 3.0))
        fig.patch.set_facecolor("black")
        for c, ds in enumerate(DATASETS):
            ax = axes[c]
            ax.set_facecolor("black")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#333")
            try:
                Y, t = fn(ds)
                Y = np.asarray(Y, dtype=float)
                ax.scatter(Y[:, 0], Y[:, 1], c=t, cmap="turbo", s=3, linewidths=0)
                ax.set_aspect("equal", adjustable="datalim")
            except Exception as e:  # noqa: BLE001
                ax.text(0.5, 0.5, "ERR", color="red", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                print(f"  ERR {aname}/{ds}: {e}")
            ax.set_title(ds, color="white", fontsize=9, pad=4)
        fig.suptitle(aname, color="#7fd", fontsize=13, y=0.99)
        plt.tight_layout(pad=0.5, rect=(0, 0, 1, 0.93))
        out = f"/tmp/audit_{aname}.png"
        fig.savefig(out, dpi=100, facecolor="black")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
