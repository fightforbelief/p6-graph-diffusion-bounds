"""Reproduce the case studies, with independent checks on every returned graph."""
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
import networkx as nx
import numpy as np
import scipy
from datasets import dataset, NAMES, fetch
from diffusion import blocks, moments, solve, state


def sampled(graph, labels, depth, seed):
    """Symmetric same-block proposals; retain rejections, sample at fixed attempt counts."""
    rng = random.Random(seed)
    colors, signals, _ = state(graph, labels, depth)
    groups = blocks(graph, colors)
    selectable = [key for key, group in groups.items() if len(group) >= 2]
    current = graph.copy()
    m = graph.number_of_edges()
    accepted = 0
    draws = []
    for sweep in range(52):
        for _ in range(m if selectable else 0):
            key = rng.choice(selectable)
            group = groups[key]
            i, j = rng.sample(range(len(group)), 2)
            a, b = group[i]
            c, d = group[j]
            if key[0] == key[1] and rng.randrange(2):
                c, d = d, c
            if len({a, b, c, d}) != 4 or current.has_edge(a, d) or current.has_edge(c, b):
                continue
            current.remove_edges_from([(a, b), (c, d)])
            current.add_edges_from([(a, d), (c, b)])
            group[i], group[j] = (a, d), (c, b)
            accepted += 1
        if sweep >= 20:
            edges = np.array(list(current.edges()))
            draws.append(float(np.sum(signals[-1][edges[:, 0]] * signals[-1][edges[:, 1]]) / m))
    if abs(moments(current, labels, 2 * depth + 1)[-1] - draws[-1]) > 1e-10:
        raise ArithmeticError("Sampled score disagreement")
    return {"seed": seed, "accepted": accepted, "draws": draws}


def study(name, depth):
    graph, labels = dataset(name)
    start = time.perf_counter()
    bounds, witnesses, outer, gap = solve(graph, labels, depth)
    elapsed = time.perf_counter() - start
    lag = 2 * depth + 1
    observed = moments(graph, labels, lag)
    colors, signals, _ = state(graph, labels, depth)
    def profile(g):
        return [sorted(colors[list(g[v])]) for v in range(len(g))]
    reference = profile(graph)
    residual = 0.
    for value, edges in zip(bounds, witnesses):
        other = nx.Graph()
        other.add_nodes_from(graph)
        other.add_edges_from(edges)
        if profile(other) != reference or other.number_of_edges() != graph.number_of_edges():
            raise ArithmeticError("Witness violates a neighborhood constraint")
        direct = moments(other, labels, lag)
        residual = max(residual, abs(direct[-1] - value),
                       float(np.max(np.abs(np.array(direct[:-1]) - observed[:-1]))))
    if residual > 1e-9 or gap > 1e-8:
        raise ArithmeticError((residual, gap))
    samples = [sampled(graph, labels, depth, s) for s in (41, 73, 101)]
    draws = [x for chain in samples for x in chain["draws"]]
    if min(draws + [observed[-1]]) < bounds[0] - 1e-9 or max(draws + [observed[-1]]) > bounds[1] + 1e-9:
        raise ArithmeticError("Inner range outside endpoints")
    if outer[0] > bounds[0] + 1e-9 or outer[1] < bounds[1] - 1e-9:
        raise ArithmeticError("Outer relaxation excludes an endpoint")
    degrees = np.array([graph.degree(v) for v in range(len(graph))])
    mass = degrees @ signals[0] / degrees.sum()
    return dict(dataset=name, depth=depth, nodes=len(graph), edges=graph.number_of_edges(),
                classes=len(mass), block_colors=len(set(colors)), lag=lag, observed=observed,
                independent_label_agreement=float(mass @ mass), bounds=bounds.tolist(), outer=outer.tolist(),
                sample_range=[min(draws), max(draws)], chains=samples, solve_seconds=elapsed,
                max_check_error=residual, solver_gap_bound=gap, witnesses=witnesses)


def main():
    fetch()
    out = Path("results")
    out.mkdir(exist_ok=True)
    runs = []
    for name in NAMES:
        for depth in (1, 2, 3):
            row = study(name, depth)
            runs.append(row)
            print(name, depth, row["bounds"], flush=True)
    files = ["datasets.py", "diffusion.py", "experiment.py", "test_diffusion.py", "data_sha256.json"]
    report = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                  scipy=scipy.__version__, networkx=nx.__version__,
                  hashes={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}, runs=runs)
    target = out / "case_studies.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    tmp.replace(target)


if __name__ == "__main__":
    main()

