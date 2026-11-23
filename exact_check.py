"""Independent rational enumeration of block optima after two or three rounds."""
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import json
import hashlib
from datasets import dataset


def prepare(graph, labels, r):
    names = sorted(set(labels))
    q = [tuple(Fraction(int(label == name)) for name in names) for label in labels]
    colors = list(labels)
    for step in range(r):
        q = [tuple(sum(q[u][i] for u in graph[v]) / graph.degree(v) for i in range(len(names)))
             for v in range(len(labels))]
        if step + 1 < r:
            raw = [(colors[v], tuple(sorted(colors[u] for u in graph[v]))) for v in range(len(labels))]
            palette = {value: i for i, value in enumerate(sorted(set(raw)))}
            colors = [palette[value] for value in raw]
    blocks = {}
    for u, v in graph.edges():
        if (colors[u], u) > (colors[v], v):
            u, v = v, u
        blocks.setdefault((colors[u], colors[v]), []).append((u, v))
    return q, blocks


def exact_block(edges, q, same):
    left, right = zip(*edges)
    dot = lambda a, b: sum(i * j for i, j in zip(a, b))
    observed = sum(dot(q[u], q[v]) for u, v in edges)
    # On a cross block, each constant side fixes that coordinate's edge sum.
    for i in range(len(q[0])):
        side1 = left + right if same else left
        side2 = side1 if same else right
        if len({q[u][i] for u in side1}) > 1 and len({q[u][i] for u in side2}) > 1:
            break
    else:
        return observed, observed
    nodes = sorted(set(left + right))
    n = len(nodes)
    left_set = set(left)
    allowed = [[j for j in range(n) if i != j and (same or ((u in left_set) != (nodes[j] in left_set)))]
               for i, u in enumerate(nodes)]
    weight = [[dot(q[u], q[v]) for v in nodes] for u in nodes]
    counts = Counter(left + right)

    @lru_cache(maxsize=None)
    def visit(degrees):
        active = [i for i, count in enumerate(degrees) if count]
        if not active:
            return Fraction(0), Fraction(0)
        choices = {i: [j for j in allowed[i] if degrees[j]] for i in active}
        if any(len(choices[i]) < degrees[i] for i in active):
            return None
        # A saturated vertex has only one possible star.
        from math import comb
        i = min(active, key=lambda v: comb(len(choices[v]), degrees[v]))
        low, high = None, None
        for star in combinations(choices[i], degrees[i]):
            rest = list(degrees)
            rest[i] = 0
            for j in star:
                rest[j] -= 1
            tail = visit(tuple(rest))
            if tail is None:
                continue
            score = sum(weight[i][j] for j in star)
            a, b = score + tail[0], score + tail[1]
            low = a if low is None else min(low, a)
            high = b if high is None else max(high, b)
        return None if low is None else (low, high)
    return visit(tuple(counts[u] for u in nodes))


def check():
    source = Path("results/case_studies.json")
    report = json.loads(source.read_text())
    rows = []
    for case in report["runs"]:
        if case["depth"] < 2:
            continue
        graph, labels = dataset(case["dataset"])
        q, parts = prepare(graph, labels, case["depth"])
        values = [Fraction(0), Fraction(0)]
        for (a, b), edges in parts.items():
            extremes = exact_block(edges, q, a == b)
            for i in range(2):
                values[i] += extremes[i] / graph.number_of_edges()
        if max(abs(float(v) - t) for v, t in zip(values, case["bounds"])) > 1e-12:
            raise ArithmeticError((case["dataset"], values))
        rows.append(dict(dataset=case["dataset"], depth=case["depth"], endpoints=list(map(str, values)),
                         width=str(values[1] - values[0])))
        print(case["dataset"], case["depth"], "width", rows[-1]["width"], flush=True)
    payload = dict(primary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), runs=rows)
    Path("results/rational.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    check()

