"""Continuous edge relaxation of the depth-one, three-step agreement problem."""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from datasets import dataset, fetch
from exact_check import prepare


def candidate_edges(edges, same_color):
    left, right = zip(*edges)
    quota = Counter(left + right)
    pairs = (combinations(sorted(quota), 2) if same_color else
             product(sorted(set(left)), sorted(set(right))))
    return quota, list(pairs)


def fractional_endpoints(graph, labels, save_prices=False):
    q, parts = prepare(graph, labels, 1)
    signals = np.asarray(q, dtype=float)
    totals = np.zeros(2)
    prices = []
    for (a, b), edges in sorted(parts.items()):
        quota, pairs = candidate_edges(edges, a == b)
        nodes = sorted(quota)
        index = {v: i for i, v in enumerate(nodes)}
        u, v = np.array(pairs).T
        rows = [index[i] for i in np.r_[u, v]]
        columns = np.tile(np.arange(len(pairs)), 2)
        inc = coo_matrix((np.ones(len(rows)), (rows, columns)),
                         shape=(len(nodes), len(pairs))).tocsc()
        weights = np.sum(signals[u] * signals[v], axis=1)
        degrees = np.array([quota[i] for i in nodes])
        for side, sign in enumerate((1, -1)):
            fit = linprog(sign * weights, A_eq=inc, b_eq=degrees, bounds=(0., 1.), method="highs")
            if not fit.success:
                raise RuntimeError(fit.message)
            if (np.max(np.abs(inc @ fit.x - degrees)) > 1e-7 or
                    np.min(fit.x) < -1e-7 or np.max(fit.x) > 1 + 1e-7):
                raise ArithmeticError("Infeasible LP solution")
            totals[side] += sign * fit.fun
            if save_prices and sign == -1:
                # For maximization, reverse the sign of the minimizing solver's marginals.
                prices.append({"colors": [a, b], "prices": {
                    str(node): format(-marginal, ".8f")
                    for node, marginal in zip(nodes, fit.eqlin.marginals)}})
    return (totals / graph.number_of_edges()).tolist(), prices


def certified_upper(graph, labels, records):
    """Evaluate a valid LP upper bound for any supplied rational vertex prices."""
    q, parts = prepare(graph, labels, 1)
    prices_by_block = {tuple(row["colors"]): row["prices"] for row in records}
    if set(prices_by_block) != set(parts) or len(records) != len(parts):
        raise ValueError("Certificate block mismatch")
    total = Fraction(0)
    for (a, b), edges in parts.items():
        quota, pairs = candidate_edges(edges, a == b)
        prices = {int(i): Fraction(y) for i, y in prices_by_block[a, b].items()}
        if set(prices) != set(quota):
            raise ValueError("Certificate vertex mismatch")
        total += sum(quota[i] * prices[i] for i in quota)
        for u, v in pairs:
            weight = sum(x * y for x, y in zip(q[u], q[v]))
            total += max(0, weight - prices[u] - prices[v])
    return total / graph.number_of_edges()


def main():
    fetch()
    primary = Path("results/case_studies.json")
    rows = []
    for row in json.loads(primary.read_text())["runs"]:
        if row["depth"] != 1:
            continue
        name = row["dataset"]
        graph, labels = dataset(name)
        bounds, prices = fractional_endpoints(graph, labels, save_prices=(name == "wisconsin"))
        lo, hi = row["bounds"]
        if bounds[0] > lo + 1e-9 or bounds[1] < hi - 1e-9:
            raise ArithmeticError("LP bounds exclude a simple-graph endpoint")
        entry = {"dataset": name, "lp": bounds, "simple": row["bounds"],
                 "stub": row["outer"], "independent": row["independent_label_agreement"]}
        if prices:
            upper = certified_upper(graph, labels, prices)
            volume = 2 * graph.number_of_edges()
            masses = Counter()
            for v, d in graph.degree:
                masses[labels[v]] += d
            reference = sum(Fraction(d, volume)**2 for d in masses.values())
            if abs(float(upper) - bounds[1]) > 1e-6 or upper >= reference:
                raise ArithmeticError("Wisconsin upper certificate failed")
            entry["certificate"] = {"prices_by_block": prices, "upper": str(upper),
                                    "independent": str(reference)}
        rows.append(entry)
        print(f"{name}: {bounds[0]:.10f} to {bounds[1]:.10f}")
    sources = [primary, Path(__file__), Path("data_sha256.json"), Path("exact_check.py")]
    result = {"scipy": scipy.__version__, "sha256": {
        p.name: sha256(p.read_bytes()).hexdigest() for p in sources}, "runs": rows}
    target = Path("results/fractional_bounds.json")
    temp = target.with_suffix(".tmp")
    temp.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    temp.replace(target)


if __name__ == "__main__":
    main()
