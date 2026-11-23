"""Validate the review records against the graph inputs and the original results."""
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import unittest

import networkx as nx
import numpy as np

from datasets import dataset
from diffusion import moments, solve, state
from review_checks import eight_vertex_control, switch


def edge_pair(a, b):
    return tuple(sorted((a, b)))


def switch_key(u, v, x, y):
    return (frozenset((edge_pair(u, v), edge_pair(x, y))),
            frozenset((edge_pair(u, y), edge_pair(x, v))))


class TestReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(Path("results/review.json").read_text())
        records = json.loads(Path("results/case_studies.json").read_text())
        cls.primary = {(r["dataset"], r["depth"]): r for r in records["runs"]}

    def test_provenance_and_primary_bounds(self):
        for name, digest in self.review["sha256"].items():
            path = Path("results") / name if name == "case_studies.json" else Path(name)
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), digest, name)
        for (name, depth), row in self.primary.items():
            graph, labels = dataset(name)
            bounds, _, outer, _ = solve(graph, labels, depth)
            np.testing.assert_allclose(bounds, row["bounds"], rtol=0, atol=1e-10)
            np.testing.assert_allclose(outer, row["outer"], rtol=0, atol=1e-10)

    def test_full_eight_vertex_control(self):
        observed = eight_vertex_control()
        self.assertEqual(json.loads(json.dumps(observed)), self.review["eight_vertex_control"],
                         "Control enumeration must reproduce stored groups")
        groups = observed["groups"]
        self.assertEqual(observed["realizations"], 28)
        self.assertEqual({g["h3"] for g in groups}, {"61/112"})
        self.assertEqual({g["h4"]: g["count"] for g in groups}, {"119/192": 24, "839/1344": 4})
        self.assertEqual(Fraction(groups[1]["h4"]) - Fraction(groups[0]["h4"]), Fraction(1, 224))

    def test_all_switches_and_isomorphism_witnesses(self):
        for name, record in self.review["depth_three_orbits"].items():
            graph, labels = dataset(name)
            colors, _, _ = state(graph, labels, 3)
            # Search all edge pairs independently of the block-based generator.
            possible = set()
            for (u, v), (a, b) in combinations(graph.edges, 2):
                if len({u, v, a, b}) < 4:
                    continue
                for x, y in ((a, b), (b, a)):
                    if colors[u] == colors[x] and colors[v] == colors[y]:
                        if not graph.has_edge(u, y) and not graph.has_edge(x, v):
                            possible.add(switch_key(u, v, x, y))
            found = record["switches"]
            self.assertEqual({switch_key(*r["switch"]) for r in found}, possible, name)
            self.assertEqual(len(found), len(possible), name)
            self.assertTrue(record["one_orbit"])
            for witness in found:
                target = graph.copy()
                switch(target, *witness["switch"])
                permutation = {v: v for v in graph}
                permutation.update({int(k): v for k, v in witness["renaming"].items()})
                self.assertEqual(set(permutation.values()), set(graph))
                self.assertTrue(all(labels[v] == labels[permutation[v]] for v in graph))
                image = {edge_pair(permutation[u], permutation[v]) for u, v in graph.edges}
                self.assertEqual(image, {edge_pair(u, v) for u, v in target.edges})

    def test_connected_scores_and_fixed_lag(self):
        for name, records in self.review["depth_one_connectivity"].items():
            graph, labels = dataset(name)
            case = self.primary[name, 1]
            reference = [sorted(labels[u] for u in graph[v]) for v in graph]
            for i, record in enumerate(records):
                other = nx.Graph()
                other.add_nodes_from(graph)
                other.add_edges_from(case["witnesses"][i])
                for move in record["switches_from_witness"]:
                    switch(other, *move)
                self.assertTrue(nx.is_connected(other))
                self.assertEqual(reference, [sorted(labels[u] for u in other[v]) for v in graph])
                self.assertAlmostEqual(moments(other, labels, 3)[-1], float(Fraction(record["final_score"])))
                if (name, i) != ("cornell", 1):
                    self.assertEqual(record["unrestricted_score"], record["final_score"])
        graph, labels = dataset("cornell")
        solution = self.review["cornell_connected_maximum"]
        self.assertEqual(solution["status"], "optimal")
        other = nx.Graph()
        other.add_nodes_from(graph)
        other.add_edges_from(solution["edges"])
        self.assertTrue(nx.is_connected(other))
        self.assertTrue(all(sorted(labels[u] for u in graph[v]) == sorted(labels[u] for u in other[v])
                            for v in graph))
        self.assertAlmostEqual(moments(other, labels, 3)[-1], solution["upper"], places=9)
        self.assertAlmostEqual(float(Fraction(solution["witness_score"])), solution["upper"], places=9)

        graph, labels = dataset("wisconsin")
        case = self.primary["wisconsin", 1]
        self.assertLess(case["bounds"][1], case["independent_label_agreement"])
        self.assertLess(case["independent_label_agreement"], case["outer"][1])
        endpoints = []
        for edges in case["witnesses"]:
            other = nx.Graph()
            other.add_nodes_from(graph)
            other.add_edges_from(edges)
            endpoints.append(moments(other, labels, 5)[-1])
        self.assertGreater(max(endpoints) - min(endpoints), 0.15747)
        rational = json.loads(Path("results/rational.json").read_text())["runs"]
        row = next(r for r in rational if (r["dataset"], r["depth"]) == ("wisconsin", 2))
        self.assertEqual(Fraction(row["width"]), Fraction(1370527, 7233624000))


if __name__ == "__main__":
    unittest.main()
