"""Sharp frontier bounds for the stationary same-label probability."""
from collections import Counter, defaultdict
from itertools import combinations, product
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def compress(values):
    index = {}
    return np.array([index.setdefault(x, len(index)) for x in values])


def state(graph, labels, depth):
    if depth < 1 or set(graph) != set(range(len(labels))) or graph.is_directed() or graph.is_multigraph():
        raise ValueError("Expected positive depth and a simple undirected graph on 0,...,n-1")
    rows = [list(graph[v]) for v in range(len(labels))]
    if any(not row or v in row for v, row in enumerate(rows)):
        raise ValueError("Remove loops and isolated vertices before analysis")
    colors = compress(labels)
    signals = [np.eye(max(colors) + 1)[colors]]
    for step in range(depth):
        signatures = [(int(colors[v]), tuple(sorted(Counter(colors[row]).items())))
                      for v, row in enumerate(rows)]
        refined = compress(signatures)
        signals.append(np.array([signals[-1][row].mean(axis=0) for row in rows]))
        if step + 1 < depth:
            colors = refined
    return colors, signals, signatures


def moments(graph, labels, horizon):
    colors = compress(labels)
    onehot = np.eye(max(colors) + 1)[colors]
    current = onehot
    rows = [list(graph[v]) for v in range(len(colors))]
    degrees = np.array([len(row) for row in rows])
    values = []
    for t in range(horizon + 1):
        values.append(float(degrees @ np.sum(onehot * current, axis=1) / degrees.sum()))
        if t < horizon:
            current = np.array([current[row].mean(axis=0) for row in rows])
    return values


def blocks(graph, colors):
    parts = defaultdict(list)
    for u, v in sorted(graph.edges()):
        if (colors[u], u) > (colors[v], v):
            u, v = v, u
        parts[(colors[u], colors[v])].append((u, v))
    return parts


def endpoints(edges, same, q):
    """Unnormalized edge-sum optima, two witnesses, stub bounds, largest solver gap."""
    a = [u for u, v in edges]
    b = [v for u, v in edges]
    if same:
        left = right = np.sort(q[a + b], axis=0)
        scale = 0.5
    else:
        left, right = np.sort(q[a], axis=0), np.sort(q[b], axis=0)
        scale = 1.0
    outer = [float(scale * np.sum(left * right[::-1])), float(scale * np.sum(left * right))]
    observed = float(np.sum(q[a] * q[b]))
    if outer[1] - outer[0] < 1e-12:
        return [observed] * 2, [edges, edges], outer, 0.
    active = sorted(set(a + b))
    pairs = list(combinations(active, 2) if same else product(sorted(set(a)), sorted(set(b))))
    u, v = np.array(pairs).T
    weights = np.sum(q[u] * q[v], axis=1)
    # Additive vertex potentials are constant on a fixed degree sequence.
    # Scaling keeps the solver's absolute tolerances small in probability units.
    weights = (weights - weights.mean()) * 10000
    indices = {node: i for i, node in enumerate(active)}
    row = [indices[x] for x in np.r_[u, v]]
    col = np.tile(np.arange(len(pairs)), 2)
    inc = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(active), len(pairs))).tocsc()
    counts = Counter(a + b)
    degree = np.array([counts[x] for x in active])
    values, witnesses, gaps = [], [], []
    for sign in (1, -1):
        result = milp(sign * weights, integrality=np.ones(len(pairs)),
                      bounds=Bounds(0, 1), constraints=LinearConstraint(inc, degree, degree),
                      options={"mip_rel_gap": 0.0})
        if not result.success or result.status != 0:
            raise RuntimeError(result.message)
        mask = np.rint(result.x).astype(int)
        if not np.array_equal(inc @ mask, degree) or np.max(np.abs(mask - result.x)) > 1e-6:
            raise ArithmeticError("Invalid endpoint witness")
        witnesses.append([pair for pair, selected in zip(pairs, mask) if selected])
        values.append(float(np.sum(q[u[mask == 1]] * q[v[mask == 1]])))
        gaps.append(abs(float(result.fun - result.mip_dual_bound)) / 10000)
    return values, witnesses, outer, max(gaps)


def solve(graph, labels, depth):
    colors, signals, _ = state(graph, labels, depth)
    values, outer = np.zeros(2), np.zeros(2)
    witnesses = [[], []]
    gap = 0.
    for (a, b), edges in blocks(graph, colors).items():
        extreme, selected, relaxed, block_gap = endpoints(edges, a == b, signals[-1])
        values += extreme
        outer += relaxed
        gap += block_gap
        for target, part in zip(witnesses, selected):
            target.extend(part)
    m = graph.number_of_edges()
    return values / m, witnesses, outer / m, gap / m

