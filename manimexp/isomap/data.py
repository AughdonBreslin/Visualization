"""Isomap math for the explainer, mirroring js/manifold/linalg.js.

Pure numpy/scipy. Deterministic given a seed so the rendered numbers are stable
and unit-testable.
"""
import heapq
import json
import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def _normalize(points):
    """Center at the origin and scale so the largest coordinate magnitude is 3.

    Matches the swiss-roll normalization so the camera framing works for any
    dataset fed into the walkthrough.
    """
    points = points - points.mean(axis=0)
    scale = np.abs(points).max()
    if scale > 1e-9:
        points = points / scale * 3.0
    return points


def swiss_roll(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    height = 21.0 * rng.random(n)
    x = t * np.cos(t)
    z = t * np.sin(t)
    points = np.stack([x, height, z], axis=1)
    points = _normalize(points)
    return {"points": points, "t": t}


def load_dataset_points(dataset, n, seed=0):
    """Return {points, t} for a named dataset.

    For 'swiss_roll' (or None) the points are generated in-process. For any other
    dataset the points are read from points/<dataset>_<n>.json, produced by
    gen_points.mjs from the exact web-sandbox generators, then normalized.
    """
    if dataset in (None, "", "swiss_roll"):
        return swiss_roll(n=n, seed=seed)
    path = os.path.join(os.path.dirname(__file__), "points", f"{dataset}_{n}.json")
    with open(path) as f:
        d = json.load(f)
    points = _normalize(np.asarray(d["points"], dtype=float))
    t = np.asarray(d["t"], dtype=float)
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


def neighbor_edges(points, adj, center):
    """Return list of (j, weight) for every neighbor j of center in adj.

    A neighbor is any node j != center where adj[center, j] > 0.
    The list is sorted ascending by weight.

    Parameters
    ----------
    points : ignored (reserved for future use; pass None if not needed)
    adj    : NxN numpy array of edge weights (0 means no edge)
    center : int, the query node index
    """
    n = adj.shape[0]
    edges = [
        (j, float(adj[center, j]))
        for j in range(n)
        if j != center and adj[center, j] > 0
    ]
    edges.sort(key=lambda x: x[1])
    return edges


def dijkstra_order(adj, src):
    """Run Dijkstra from src on a dense weight matrix and record settle order.

    Parameters
    ----------
    adj : NxN numpy array; adj[i, j] > 0 is the edge weight, 0 means no edge.
    src : int, the source node.

    Returns
    -------
    order : list of node indices in the order they are finalized (settled),
            order[0] == src.
    dist  : 1-D numpy array of final shortest distances from src.
    """
    n = adj.shape[0]
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    settled = [False] * n
    order = []
    # heap entries: (distance, node)
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if settled[u]:
            continue
        settled[u] = True
        order.append(u)
        for v in range(n):
            if v == u or adj[u, v] == 0:
                continue
            nd = d + adj[u, v]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return order, dist


def power_iteration_trace(M, iters=10, seed=0):
    """Trace power iteration for the dominant eigenvector of symmetric matrix M.

    Parameters
    ----------
    M     : symmetric NxN numpy array
    iters : number of iterations to run (result has iters+1 entries)
    seed  : integer seed for the initial random vector

    Returns
    -------
    vectors  : list of iters+1 unit-norm numpy arrays (the estimate after each
               multiply-and-normalize step, starting from the seeded vector).
    rayleigh : list of iters+1 Rayleigh quotients v^T M v (each v is unit norm).
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M.shape[0])
    v = v / np.linalg.norm(v)
    vectors = [v.copy()]
    rayleigh = [float(v @ M @ v)]
    for _ in range(iters):
        v = M @ v
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        vectors.append(v.copy())
        rayleigh.append(float(v @ M @ v))
    return vectors, rayleigh


def sample_along_path(path, m=4):
    """Return m node indices spaced as evenly as possible along path.

    Always includes path[0] and path[-1]. If len(path) <= m, returns all
    elements (with first and last guaranteed at the boundaries).

    Parameters
    ----------
    path : list of node indices
    m    : desired number of samples

    Returns
    -------
    list of m node indices (or len(path) if path is shorter than m)
    """
    L = len(path)
    if L <= m:
        return list(path)
    indices = [round(i * (L - 1) / (m - 1)) for i in range(m)]
    return [path[i] for i in indices]


def farthest_point_sample(D, m, start):
    """Return m well-separated node indices via farthest-point sampling.

    Begins at `start` and repeatedly adds the node whose distance to the
    current set (its nearest already-chosen node) is largest. This yields a
    spread-out subset whose double-centered Gram matrix is non-degenerate,
    unlike points sampled along a single path (which are nearly collinear and
    give a rank-1 Gram matrix).

    Parameters
    ----------
    D     : NxN distance matrix.
    m     : number of points to select.
    start : index of the first point.
    """
    idx = [int(start)]
    while len(idx) < m:
        dmin = D[idx].min(axis=0).astype(float).copy()
        dmin[idx] = -1.0
        idx.append(int(np.argmax(dmin)))
    return idx


def build_dataset(n=1000, k=8, seed=0, dataset=None):
    roll = load_dataset_points(dataset, n, seed=seed)
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

    # Center: node closest to the centroid of all points (interior, typical node).
    centroid = points.mean(axis=0)
    center = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))

    center_edges = neighbor_edges(points, adj, center)

    dijk_order, _ = dijkstra_order(adj, src)

    power_vecs, power_rayleigh = power_iteration_trace(B, iters=10, seed=0)

    # A spread-out 4-point sample (not along one path) so the 4x4 Gram matrix
    # shown in steps 4 and 5 is non-degenerate: it then has a real second
    # eigenvalue and a power iteration that visibly climbs over several steps.
    sample_idx = farthest_point_sample(Dfix, 4, src)

    D_sub = Dfix[np.ix_(sample_idx, sample_idx)]
    D_sample = np.round(D_sub, 2).tolist()

    # Power iteration is traced on the rounded matrix that is actually drawn,
    # so the v^T B v values shown match the visible numbers.
    B_sub_disp = np.round(double_center_squared(D_sub), 2)
    B_sample = B_sub_disp.tolist()
    sample_power_vectors, sample_power_rayleigh = power_iteration_trace(
        B_sub_disp, iters=6, seed=1
    )
    sample_eigvals = np.sort(np.linalg.eigvalsh(B_sub_disp))[::-1]

    return {
        "points": points, "t": roll["t"], "adj": adj, "edges": edges,
        "D": Dfix, "src": src, "tgt": tgt, "path": path,
        "B": B, "eigvals": eigvals, "eigvecs": eigvecs, "embedding": embedding,
        "excerpt_D": np.round(Dfix[:4, :4], 2).tolist(),
        "excerpt_B": np.round(B[:4, :4], 2).tolist(),
        "center": center,
        "center_edges": center_edges,
        "dijkstra_order": dijk_order,
        "power_vectors": power_vecs,
        "power_rayleigh": power_rayleigh,
        "sample_idx": sample_idx,
        "D_sample": D_sample,
        "B_sample": B_sample,
        "sample_power_vectors": sample_power_vectors,
        "sample_power_rayleigh": sample_power_rayleigh,
        "sample_eigvals": sample_eigvals,
    }
