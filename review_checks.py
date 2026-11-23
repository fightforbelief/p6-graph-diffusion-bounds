"""Reproduce the post-primary checks on connectivity, isomorphism and the control."""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

import networkx as nx

from connected_bound import connected_maximum
from datasets import dataset, fetch
from diffusion import blocks, state
from exact_check import prepare


def legal_switches(graph, colors):
    """Yield the two possible reconnections in a within-color block, one across colors."""
    for (a, b), edges in blocks(graph, colors).items():
        for (u, v), (s, t) in combinations(edges, 2):
            if len({u, v, s, t}) != 4:
                continue
            for x, y in ([(s, t), (t, s)] if a == b else [(s, t)]):
                if not graph.has_edge(u, y) and not graph.has_edge(x, v):
                    yield u, v, x, y


def switch(graph, u, v, x, y):
    graph.remove_edges_from([(u, v), (x, y)])
    graph.add_edges_from([(u, y), (x, v)])


def orbit_check(graph, labels):
    colors, _, _ = state(graph, labels, 3)
    nx.set_node_attributes(graph, dict(enumerate(labels)), "label")
    results = []
    for move in legal_switches(graph, colors):
        u, v, x, y = move
        other = graph.copy()
        switch(other, *move)
        # Usually a single transposition is enough. Verify its entire edge image.
        permutation = None
        for a, b in ((u, x), (v, y)):
            renaming = {a: b, b: a}
            if labels[a] == labels[b] and all(other.has_edge(renaming.get(i, i), renaming.get(j, j))
                                               for i, j in graph.edges):
                permutation = renaming
                break
        if permutation is None:
            matcher = nx.isomorphism.GraphMatcher(
                graph, other, node_match=nx.isomorphism.categorical_node_match("label", None))
            if matcher.is_isomorphic():
                permutation = {i: j for i, j in matcher.mapping.items() if i != j}
        results.append({"switch": move, "renaming": permutation})
    return {"switches": results, "one_orbit": all(r["renaming"] is not None for r in results)}


def agreement(graph, q):
    """Exact three-step agreement from the fixed one-step class probabilities."""
    return sum(sum(a*b for a, b in zip(q[i], q[j])) for i, j in graph.edges) / graph.number_of_edges()


def join_components(graph, colors, q, maximize):
    """Join components with the smallest loss from the selected endpoint."""
    moves = []
    while not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        membership = {v: i for i, part in enumerate(components) for v in part}
        bridges = {frozenset(edge) for edge in nx.bridges(graph)}
        best = None
        for u, v, x, y in legal_switches(graph, colors):
            if membership[u] == membership[x]:
                continue
            if frozenset((u, v)) in bridges and frozenset((x, y)) in bridges:
                continue
            delta = sum((a-b)*(c-d) for a, b, c, d in zip(q[u], q[x], q[y], q[v]))
            candidate = (delta if maximize else -delta, u, v, x, y)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break
        move = best[1:]
        switch(graph, *move)
        if nx.number_connected_components(graph) != len(components) - 1:
            raise ArithmeticError("Component count did not fall by one")
        moves.append(move)
    return moves


def connectivity(graph, labels, case):
    colors, _, _ = state(graph, labels, 1)
    q, _ = prepare(graph, labels, 1)
    quotas = [Counter(labels[u] for u in graph[v]) for v in range(len(graph))]
    records = []
    for side, edges in enumerate(case["witnesses"]):
        other = nx.Graph()
        other.add_nodes_from(graph)
        other.add_edges_from(edges)
        initial = agreement(other, q)
        if abs(float(initial) - case["bounds"][side]) > 1e-10:
            raise ArithmeticError("Stored endpoint differs from its witness")
        sizes = sorted(map(len, nx.connected_components(other)), reverse=True)
        removed = sum(not other.has_edge(u, v) for u, v in graph.edges)
        moves = join_components(other, colors, q, maximize=bool(side))
        if quotas != [Counter(labels[u] for u in other[v]) for v in range(len(graph))]:
            raise ArithmeticError("Connected witness violates a vertex quota")
        records.append({"original_component_sizes": sizes, "original_edges_removed": removed,
                        "unrestricted_score": str(initial), "switches_from_witness": moves,
                        "connected": nx.is_connected(other), "final_score": str(agreement(other, q))})
    return records


def eight_vertex_control():
    left, right = range(4), range(4, 8)
    choices = [list(combinations(right, degree)) for degree in (3, 3, 1, 1)]
    histogram = Counter()
    representatives = {}
    for stars in product(*choices):
        if any(sum(v in star for star in stars) != 2 for v in right):
            continue
        graph = nx.complete_graph(left)
        graph.add_nodes_from(right)
        graph.add_edges_from((u, v) for u, star in enumerate(stars) for v in star)
        if not nx.is_connected(graph):
            raise ArithmeticError("Disconnected control")
        q = [Fraction(v in left) for v in graph]
        values = []
        for lag in range(1, 5):
            q = [sum(q[u] for u in graph[v]) / graph.degree(v) for v in graph]
            if lag >= 3:
                mass = sum(graph.degree(v) * (q[v] if v in left else 1-q[v]) for v in graph)
                values.append(mass / (2*graph.number_of_edges()))
        key = tuple(values)
        histogram[key] += 1
        representatives.setdefault(key, stars)
    return {"realizations": sum(histogram.values()), "groups": [
        {"h3": str(a), "h4": str(b), "count": count, "cross_stars": representatives[a, b]}
        for (a, b), count in sorted(histogram.items())]}


def main():
    fetch()
    primary = Path("results/case_studies.json")
    cases = json.loads(primary.read_text())
    orbits, witnesses = {}, {}
    for row in cases["runs"]:
        name = row["dataset"]
        if row["depth"] == 1:
            graph, labels = dataset(name)
            witnesses[name] = connectivity(graph, labels, row)
            orbits[name] = orbit_check(graph, labels)
    graph, labels = dataset("cornell")
    cornell = connected_maximum(graph, labels)
    feasible = Fraction(witnesses["cornell"][1]["final_score"])
    if not witnesses["cornell"][1]["connected"] or float(feasible) > cornell["upper"] + 1e-9:
        raise ArithmeticError("Invalid connected-maximum bracket")
    if cornell["status"] == "optimal":
        optimum = nx.Graph()
        optimum.add_nodes_from(graph)
        optimum.add_edges_from(cornell["edges"])
        q, _ = prepare(graph, labels, 1)
        score = agreement(optimum, q)
        if (not nx.is_connected(optimum) or
                any(Counter(labels[u] for u in graph[v]) != Counter(labels[u] for u in optimum[v])
                    for v in graph) or abs(float(score) - cornell["upper"]) > 1e-9):
            raise ArithmeticError("Invalid maximizing connected graph")
        cornell["witness_score"] = str(score)
    files = (primary, Path("data_sha256.json"), Path(__file__), Path("connected_bound.py"))
    output = {"sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in files},
              "eight_vertex_control": eight_vertex_control(), "depth_three_orbits": orbits,
              "depth_one_connectivity": witnesses, "cornell_connected_maximum": cornell}
    target = Path("results/review.json")
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    temporary.replace(target)
    print("Eight-vertex realizations:", output["eight_vertex_control"]["realizations"])
    print("One-orbit checks:", {k: v["one_orbit"] for k, v in orbits.items()})
    print("Connected Cornell maximum:", cornell["status"], cornell["upper"], flush=True)


if __name__ == "__main__":
    main()
