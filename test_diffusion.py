"""Independent enumeration and all-depth sharpness regression tests."""
from collections import defaultdict
from itertools import combinations
import unittest
import networkx as nx
import numpy as np
from diffusion import moments, solve, state


class TestDiffusion(unittest.TestCase):
    def test_enumeration(self):
        # Group *all* labeled simple isolate-free graphs on five vertices by
        # raw, uncompressed vertex signatures, independently of solver blocks.
        n = 5
        labels = [0, 0, 0, 1, 1]
        pairs = list(combinations(range(n), 2))
        for depth in (1, 2):
            groups = defaultdict(list)
            for bits in range(1 << len(pairs)):
                graph = nx.Graph()
                graph.add_nodes_from(range(n))
                graph.add_edges_from(e for j, e in enumerate(pairs) if bits & (1 << j))
                if min(dict(graph.degree()).values()) == 0:
                    continue
                sig = labels[:]
                for _ in range(depth):
                    sig = [(sig[v], tuple(sorted(sig[w] for w in graph[v]))) for v in range(n)]
                groups[tuple(sig)].append((graph, moments(graph, labels, 2 * depth + 1)))
            total = sum(len(family) for family in groups.values())
            self.assertEqual(total, 768)
            for family in groups.values():
                reference = family[0][0]
                direct = np.array([seq for _, seq in family])
                self.assertLess(np.ptp(direct[:, :-1], axis=0).max(), 1e-12)
                actual, _, _, _ = solve(reference, labels, depth)
                np.testing.assert_allclose(actual, [min(direct[:, -1]), max(direct[:, -1])], atol=1e-12)
            print(f"enumeration depth={depth}: {total} graphs, {len(groups)} distinct fibers")

    def test_all_depth_sharpness(self):
        for r in range(1, 9):
            length = 8 * r + 8
            g = nx.cycle_graph(2 * length)
            labels = [int(v in (0, length)) for v in range(2 * length)]
            h = g.copy()
            u, v = r, r + 1
            h.remove_edges_from([(u, v), (u + length, v + length)])
            h.add_edges_from([(u, u + length), (v, v + length)])
            self.assertTrue(nx.is_connected(h))
            self.assertTrue(all(d == 2 for _, d in h.degree()))
            colors, _, _ = state(g, labels, r)
            for x in g:
                self.assertEqual(sorted(colors[list(g[x])]), sorted(colors[list(h[x])]))
            # Both directions are counted for each of the two complementary classes.
            diff = np.array(moments(h, labels, 2*r+1)) - moments(g, labels, 2*r+1)
            np.testing.assert_allclose(diff[:-1], 0, atol=1e-13)
            self.assertAlmostEqual(diff[-1], 4 / (2 * length * 2**(2*r+1)), places=13)


if __name__ == "__main__":
    unittest.main()

