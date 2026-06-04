"""Isomap math for the explainer, mirroring js/manifold/linalg.js.

Pure numpy/scipy. Deterministic given a seed so the rendered numbers are stable
and unit-testable.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def swiss_roll(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    height = 21.0 * rng.random(n)
    x = t * np.cos(t)
    z = t * np.sin(t)
    points = np.stack([x, height, z], axis=1)
    points = points - points.mean(axis=0)
    points = points / np.abs(points).max() * 3.0
    return {"points": points, "t": t}


def knn_graph(points, k=8):
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    adj = np.zeros((n, n))
    for i in range(n):
        nn = np.argsort(dist[i])[:k]
        for j in nn:
            w = dist[i, j]
            adj[i, j] = w
            adj[j, i] = w
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] > 0]
    return adj, edges


def geodesic_distances(adj):
    g = csr_matrix(adj)
    D, predecessors = shortest_path(g, method="D", directed=False, return_predecessors=True)
    return D, predecessors


def path_between(predecessors, src, tgt):
    path = [tgt]
    cur = tgt
    while cur != src and cur >= 0:
        cur = predecessors[src, cur]
        if cur < 0:
            break
        path.append(cur)
    path.reverse()
    return path


def double_center_squared(D):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ (D ** 2) @ J


def top2_eig(B):
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:2]
    return vals[order], vecs[:, order]


def build_dataset(n=1000, k=8, seed=0):
    roll = swiss_roll(n=n, seed=seed)
    points = roll["points"]
    adj, edges = knn_graph(points, k=k)
    D, predecessors = geodesic_distances(adj)
    finite = np.where(np.isfinite(D), D, -1)
    src = int(np.unravel_index(np.argmax(finite), finite.shape)[0])
    tgt = int(np.unravel_index(np.argmax(finite), finite.shape)[1])
    path = path_between(predecessors, src, tgt)
    Dfix = np.where(np.isfinite(D), D, finite.max())
    B = double_center_squared(Dfix)
    eigvals, eigvecs = top2_eig(B)
    embedding = eigvecs * np.sqrt(np.maximum(eigvals, 0.0))
    return {
        "points": points, "t": roll["t"], "adj": adj, "edges": edges,
        "D": Dfix, "src": src, "tgt": tgt, "path": path,
        "B": B, "eigvals": eigvals, "eigvecs": eigvecs, "embedding": embedding,
        "excerpt_D": np.round(Dfix[:4, :4], 2).tolist(),
        "excerpt_B": np.round(B[:4, :4], 2).tolist(),
    }
