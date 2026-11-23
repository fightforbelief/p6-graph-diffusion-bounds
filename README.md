# How many steps of label diffusion are fixed by local neighborhood counts?

Yunjie Guo and Hanqing Yang. Independent Researcher, Suzhou, China. Correspondence to Hanqing Yang (hanqing.yang189@gmail.com).

[Read the manuscript](paper/manuscript.pdf) or [inspect the LaTeX source](paper/manuscript.tex).

For a stationary random walk on a simple undirected graph, fixing each named vertex's depth-r color-refinement signature fixes the entire joint endpoint-label distribution through time 2r. A single switch on a connected cycle makes the next time differ. At that time the extrema of any linear contrast reduce to weighted degree-constrained subgraph problems within color blocks. The numerical implementation computes the trace, the probability of matching endpoint labels.

In five labeled networks the full lag-three width spans 0.68 to 12.91 percentage points under one round of constraints. All five retain the sign of their contrast with independent stationary endpoints. The paper distinguishes finite sampling, sharp attainable endpoints, and maximum edge replacement. It studies stationary label diffusion on known labels. The code symmetrizes hyperlinks and omits self-loops, isolates and weights.

## Reproduction

Run from this project root. The recorded computations ran on the Windows research host with Python 3.12 and the project-local environment. Data retrieval needs network access once. The WebKB inputs are pinned to an upstream commit, and acquired files are checked against [the input SHA-256 manifest](data_sha256.json). Raw downloads are cached under the ignored data directory. Karate is provided by the pinned NetworkX release.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe datasets.py
.\.venv\Scripts\python.exe -m unittest -v test_diffusion
.\.venv\Scripts\python.exe experiment.py
.\.venv\Scripts\python.exe diagnostics.py
.\.venv\Scripts\python.exe exact_check.py
.\.venv\Scripts\python.exe paper/tables.py
~~~

For the manuscript, run Tectonic on paper/manuscript.tex. The vector illustration is included. Its optional regeneration script uses matplotlib with a noninteractive backend. A rerun regenerates results in place and wall times will vary. Different optimal graphs can realize the same endpoint. The exact enumeration script targets depths two and three of the five supplied networks and has exponential worst-case complexity.

## Computation records

* [Primary protocol](PROTOCOL.md). The five networks, depths, statistic, switch budgets and validation design were specified before the case studies.
* [Primary results](results/case_studies.json). For all 15 instances this records the observed moments, two extremizing edge lists, relaxed and sampled ranges, chain seeds and draws, optimization times, numerical checks, environment and code checksums. Every edge list uses the normalized vertex order in datasets.py. Timings measure endpoint computation only.
* [Supplementary diagnostics](results/diagnostics.json). After the primary run we measured maximum replacement of original edges, the entire one-switch range from the observed graph, and a witnessed lower bound on the depth-one lag-five spread. The four lag-five examples are the original graph, the two frontier extremizers and the maximum-replacement witness, in that order.
* [Exact fractions](results/rational.json). A separate rational-arithmetic recursion gives the two endpoints at depths two and three, including every zero-width case. Each supplementary output identifies the primary result file by checksum.
* [Tests](test_diffusion.py). All 768 isolate-free simple graphs on five named vertices under a fixed 3:2 label split, grouped independently at depths one and two, plus exact predicted increments on connected-cycle examples through depth eight.

## Inputs

Geom-GCN WebKB uses Cornell, Texas and Wisconsin webpage categories. Football uses the 2000 US college conference affiliations. Karate uses the observed split as encoded by NetworkX. See the manuscript for the original study references and datasets.py for the exact input paths. [Geom-GCN release](https://github.com/bingzhewei/geom-gcn/tree/1124af17444d7fd09686504ec46caa3f39f4f632/new_data) · [Newman's network data](https://websites.umich.edu/~mejn/netdata/) · [NetworkX karate club data](https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html).

