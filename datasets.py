"""Fetch inputs at documented sources and prepare unweighted undirected graphs."""
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlopen
from hashlib import sha256
import json
import networkx as nx

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data"
WEB = "https://raw.githubusercontent.com/bingzhewei/geom-gcn/1124af17444d7fd09686504ec46caa3f39f4f632/new_data"
NAMES = ("karate", "football", "cornell", "texas", "wisconsin")
FILES = {"football.zip": "https://websites.umich.edu/~mejn/netdata/football.zip"}
for school in NAMES[2:]:
    for suffix in ("graph_edges", "node_feature_label"):
        FILES[f"{school}_{suffix}.txt"] = f"{WEB}/{school}/out1_{suffix}.txt"


def fetch():
    CACHE.mkdir(exist_ok=True)
    checksums = {}
    manifest = HERE / "data_sha256.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    for name, url in FILES.items():
        path = CACHE / name
        if not path.exists():
            with urlopen(url, timeout=90) as response:
                content = response.read()
            if name in previous and sha256(content).hexdigest() != previous[name]:
                raise ValueError(f"Unexpected download for {name}")
            path.write_bytes(content)
        checksums[name] = sha256(path.read_bytes()).hexdigest()
    if previous and previous != checksums:
        raise ValueError("Input checksums changed")
    if not previous:
        manifest.write_text(json.dumps(checksums, indent=2) + "\n")


def dataset(name):
    if name == "karate":
        graph = nx.karate_club_graph()
        labels = nx.get_node_attributes(graph, "club")
    elif name == "football":
        with ZipFile(CACHE / "football.zip") as archive:
            graph = nx.parse_gml(archive.read("football.gml").decode().splitlines(), label=None)
        labels = nx.get_node_attributes(graph, "value")
    elif name in NAMES[2:]:
        nodes = (CACHE / f"{name}_node_feature_label.txt").read_text().splitlines()[1:]
        labels = {int(row.split()[0]): int(row.split()[-1]) for row in nodes}
        graph = nx.Graph()
        graph.add_nodes_from(labels)
        for row in (CACHE / f"{name}_graph_edges.txt").read_text().splitlines()[1:]:
            a, b = map(int, row.split())
            if a != b:
                graph.add_edge(a, b)
    else:
        raise ValueError(name)
    keep = sorted(v for v, degree in graph.degree() if degree)
    relabel = {v: i for i, v in enumerate(keep)}
    return nx.relabel_nodes(graph.subgraph(keep).copy(), relabel), [labels[v] for v in keep]


if __name__ == "__main__":
    fetch()

