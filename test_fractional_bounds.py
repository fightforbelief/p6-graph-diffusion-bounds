"""Recompute the LP benchmark and independently check its Wisconsin sign certificate."""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import unittest

import numpy as np

from datasets import dataset, NAMES
from fractional_bounds import fractional_endpoints


class TestFractionalBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(Path("results/fractional_bounds.json").read_text())
        cls.rows = {row["dataset"]: row for row in cls.report["runs"]}

    def test_five_relaxations(self):
        self.assertEqual(len(self.report["runs"]), len(NAMES))
        self.assertEqual(set(self.rows), set(NAMES))
        for name, digest in self.report["sha256"].items():
            path = Path("results", name) if name == "case_studies.json" else Path(name)
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), digest, name)
        for name, row in self.rows.items():
            bounds, _ = fractional_endpoints(*dataset(name))
            np.testing.assert_allclose(bounds, row["lp"], rtol=0, atol=1e-10)
            lo, hi = row["simple"]
            self.assertLessEqual(bounds[0], lo + 1e-9)
            self.assertGreaterEqual(bounds[1], hi - 1e-9)
            self.assertFalse(bounds[0] <= row["independent"] <= bounds[1])
        positive_gaps = {name for name, row in self.rows.items()
                         if row["lp"][1] - row["simple"][1] > 1e-9}
        self.assertEqual(positive_gaps, {"wisconsin", "cornell"})

    def test_wisconsin_certificate_from_graph(self):
        graph, labels = dataset("wisconsin")
        cert = self.rows["wisconsin"]["certificate"]
        classes = sorted(set(labels))
        quotas = [Counter(labels[u] for u in graph[v]) for v in range(len(labels))]
        probs = [[Fraction(quotas[v][c], graph.degree(v)) for c in classes] for v in range(len(labels))]
        # Independently enumerate candidate pairs and evaluate the rational dual inequality.
        prices = {tuple(row["colors"]): {int(v): Fraction(y) for v, y in row["prices"].items()}
                  for row in cert["prices_by_block"]}
        upper = Fraction(0)
        for v, own in enumerate(labels):
            for other, degree in quotas[v].items():
                upper += degree * prices[tuple(sorted((own, other)))][v]
        for u, v in combinations(range(len(labels)), 2):
            if not quotas[u][labels[v]] or not quotas[v][labels[u]]:
                continue
            y = prices[tuple(sorted((labels[u], labels[v])))]
            weight = sum(a*b for a, b in zip(probs[u], probs[v]))
            upper += max(0, weight-y[u]-y[v])
        upper /= graph.number_of_edges()
        volume = 2 * graph.number_of_edges()
        reference = sum(Fraction(sum(graph.degree(v) for v in graph if labels[v] == c), volume)**2
                        for c in classes)
        self.assertEqual(reference, Fraction(cert["independent"]))
        self.assertEqual(upper, Fraction(cert["upper"]))
        self.assertLess(upper, reference)
        self.assertAlmostEqual(float(upper), self.rows["wisconsin"]["lp"][1], places=6)


if __name__ == "__main__":
    unittest.main()
