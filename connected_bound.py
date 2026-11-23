"""Upper bound on connected three-step label agreement using connectivity cuts."""
from collections import Counter
from itertools import combinations, product

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack

from diffusion import blocks, state


def connected_maximum(graph, labels, max_rounds=150):
    """Return the best upper bound and a witness when a maximizing graph is connected.

    The input graph must be connected and have at least three vertices of degree > 1.
    All cuts are necessary for connectedness; an unfinished solve still bounds the maximum.
    """
    if not nx.is_connected(graph) or sum(d > 1 for _, d in graph.degree) < 3:
        raise ValueError("Expected a connected graph with at least three non-leaves")
    colors, signals, _ = state(graph, labels, 1)
    pairs, rows, quotas = [], [], []
    core = {v for v, d in graph.degree if d > 1}
    for (a, b), edges in blocks(graph, colors).items():
        left, right = zip(*edges)
        counts = Counter(left + right)
        nodes = sorted(counts)
        row = {v: len(quotas) + i for i, v in enumerate(nodes)}
        quotas.extend(counts[v] for v in nodes)
        candidates = (combinations(nodes, 2) if a == b else
                      product(sorted(set(left)), sorted(set(right))))
        for u, v in candidates:
            # A leaf--leaf edge would be an isolated two-vertex component.
            if u in core or v in core:
                pairs.append((u, v))
                rows.extend((row[u], row[v]))
    columns = np.arange(len(pairs))
    incidence = coo_matrix((np.ones(len(rows)), (rows, np.repeat(columns, 2))),
                           shape=(len(quotas), len(pairs))).tocsc()
    u, v = np.array(pairs).T
    weights = np.sum(signals[-1][u] * signals[-1][v], axis=1)
    # Removing leaves from any connected completion leaves a connected core.
    # Each of its vertices therefore has at least one neighbor in the core.
    cuts = []
    for node in sorted(core):
        mask = [a in core and b in core and node in (a, b) for a, b in pairs]
        cuts.append(csr_matrix(np.asarray(mask, dtype=int)[None, :]))
    seen = set()
    m = graph.number_of_edges()
    upper = 1.0
    for round_number in range(1, max_rounds + 1):
        matrix = vstack([incidence, *cuts], format="csc")
        constraints = LinearConstraint(matrix, np.r_[quotas, np.ones(len(cuts))],
                                       np.r_[quotas, np.full(len(cuts), np.inf)])
        fit = milp(-10000 * weights, integrality=np.ones(len(pairs)), bounds=Bounds(0, 1),
                   constraints=constraints, options={"mip_rel_gap": 0})
        if fit.status != 0 or abs(fit.fun - fit.mip_dual_bound) > 1e-5:
            raise RuntimeError(fit.message)
        mask = np.rint(fit.x).astype(int)
        if np.max(np.abs(fit.x - mask)) > 1e-6 or not np.array_equal(incidence @ mask, quotas):
            raise ArithmeticError("Infeasible binary solution")
        upper = min(upper, -fit.mip_dual_bound / (10000 * m))
        witness = nx.Graph()
        witness.add_nodes_from(graph)
        witness.add_edges_from(pair for pair, bit in zip(pairs, mask) if bit)
        components = list(nx.connected_components(witness))
        if len(components) == 1:
            return {"status": "optimal", "rounds": round_number, "upper": float(upper),
                    "component_cuts": len(seen), "edges": sorted(witness.edges)}
        if round_number == max_rounds:
            break
        for component in components:
            side = frozenset(component if 0 not in component else set(graph) - component)
            if side not in seen:
                seen.add(side)
                crossing = np.array([(a in side) != (b in side) for a, b in pairs], dtype=int)
                cuts.append(csr_matrix(crossing[None, :]))
    return {"status": "upper_bound", "rounds": max_rounds, "upper": float(upper),
            "component_cuts": len(seen)}
