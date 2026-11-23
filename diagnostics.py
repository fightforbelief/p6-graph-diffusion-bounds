"""Post-primary diagnostics for graph rigidity, one switch and a common lag."""
from itertools import combinations, product
from collections import Counter
from pathlib import Path
import json
import hashlib
import networkx as nx
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from diffusion import state, blocks, moments
from datasets import dataset


def mobility(graph, parts):
    """Maximize deleted original edges by minimizing overlap with the observed graph."""
    changed = 0
    witness = []
    for (a, b), edges in parts.items():
        left, right = zip(*edges)
        nodes = sorted(set(left + right))
        pairs = list(combinations(nodes, 2) if a == b else product(sorted(set(left)), sorted(set(right))))
        if len(pairs) == len(edges):
            witness.extend(edges)
            continue
        idx = {v: i for i, v in enumerate(nodes)}
        u, v = np.array(pairs).T
        rows = [idx[i] for i in np.r_[u, v]]
        incidence = coo_matrix((np.ones(len(rows)), (rows, np.tile(np.arange(len(pairs)), 2))),
                               shape=(len(nodes), len(pairs))).tocsc()
        deg = Counter(left + right)
        bvec = [deg[i] for i in nodes]
        overlap = np.array([int(graph.has_edge(i, j)) for i, j in pairs])
        result = milp(overlap, integrality=np.ones(len(pairs)), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(incidence, bvec, bvec), options={"mip_rel_gap": 0})
        if result.status or abs(result.fun - result.mip_dual_bound) > 1e-6:
            raise RuntimeError(result.message)
        mask = np.rint(result.x).astype(int)
        if not np.array_equal(incidence @ mask, bvec):
            raise ArithmeticError("Invalid mobility witness")
        witness.extend(edge for edge, bit in zip(pairs, mask) if bit)
        changed += len(edges) - int(overlap @ mask)
    return changed, witness


def analyze(row):
    graph, labels = dataset(row["dataset"])
    colors, signals, _ = state(graph, labels, row["depth"])
    q = signals[-1]
    parts = blocks(graph, colors)
    m = graph.number_of_edges()
    extremes = [0., 0.]
    best = [None, None]
    for (a, b), edges in parts.items():
        for (u, v), (s, t) in combinations(edges, 2):
            if len({u, v, s, t}) < 4:
                continue
            for x, y in ([(s, t), (t, s)] if a == b else [(s, t)]):
                if graph.has_edge(u, y) or graph.has_edge(x, v):
                    continue
                delta = float((q[u] - q[x]) @ (q[y] - q[v]) / m)
                for side, improve in enumerate((delta < extremes[0], delta > extremes[1])):
                    if improve:
                        extremes[side] = delta
                        best[side] = (u, v, x, y)
    for delta, swap in zip(extremes, best):
        if swap:
            u, v, x, y = swap
            other = graph.copy()
            other.remove_edges_from([(u, v), (x, y)])
            other.add_edges_from([(u, y), (x, v)])
            if abs(moments(other, labels, row["lag"])[-1] - row["observed"][-1] - delta) > 1e-10:
                raise ArithmeticError("Switch witness mismatch")
    changed, farthest = mobility(graph, parts)
    fifth = [moments(graph, labels, 5)[-1]]
    for edges in row["witnesses"] + [farthest]:
        other = nx.Graph()
        other.add_nodes_from(graph)
        other.add_edges_from(edges)
        assert all(sorted(colors[list(other[v])]) == sorted(colors[list(graph[v])]) for v in graph)
        fifth.append(moments(other, labels, 5)[-1])
    return dict(dataset=row["dataset"], depth=row["depth"], max_deletions=changed,
                farthest=farthest, one_switch_deltas=extremes, switches=best, lag5_examples=fifth)


if __name__ == "__main__":
    source = Path("results/case_studies.json")
    results = [analyze(row) for row in json.loads(source.read_text())["runs"]]
    output = dict(primary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), runs=results)
    Path("results/diagnostics.json").write_text(json.dumps(output, separators=(",", ":")) + "\n")

