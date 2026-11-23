# A sharp time horizon for label diffusion from local neighborhood counts

Yunjie Guo and Hanqing Yang · Independent Researcher, Suzhou, China

Correspondence: Hanqing Yang (hanqing.yang189@gmail.com)

[Manuscript (PDF)](paper/manuscript.pdf) · [LaTeX source](paper/manuscript.tex)

Fix the depth-*r* color-refinement signature of every named vertex of a simple undirected graph. For a stationary random walk, this determines the joint distribution of the starting and ending labels through step **2r**. At step **2r + 1**, a single switch on a connected cycle can change that distribution. The paper gives attainable bounds on any linear contrast at this first possible step of disagreement. The numerical implementation studies the probability that the two labels agree.

## Main case and control

| Wisconsin WebKB | Result |
| --- | --- |
| Three-step agreement with one round fixed | **16.08%–28.98%**, with both endpoints achieved on connected graphs. The independent-label reference is **29.9222%**. The stub-pairing upper bound is **30.6841%**. The fractional LP already certifies the deficit at **28.9950%**. The largest attainable score is **28.9823%**. |
| At most one switch from the observed graph | The change ranges from **−0.259 to +0.178 percentage points**. |
| Five-step agreement with one versus two rounds fixed | With one round the witnessed width is **at least 15.747 percentage points**. With two rounds the full width is **0.01895 points**. These widths cover the primary family of simple graphs. |

In the five empirical graphs, depth three fixes the entire labeled graph up to label-preserving isomorphism. Every lag of label diffusion therefore agrees, including lag seven. Named edges can still change. A separate eight-vertex connected control has 28 realizations with a common three-step agreement of 61/112. Four-step agreement differs by 1/224 between two groups, which proves that these groups include nonisomorphic labeled networks.

## Reproduction

Run from the project root. The reference run used Windows, Python 3.12.10, and the versions in [requirements.txt](requirements.txt). Acquisition verifies cached files against [their SHA-256 checksums](data_sha256.json). The WebKB inputs are pinned to an upstream commit. NetworkX supplies the karate graph.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe datasets.py
.\.venv\Scripts\python.exe -m unittest -v test_diffusion
.\.venv\Scripts\python.exe experiment.py
.\.venv\Scripts\python.exe diagnostics.py
.\.venv\Scripts\python.exe exact_check.py
.\.venv\Scripts\python.exe review_checks.py
.\.venv\Scripts\python.exe fractional_bounds.py
.\.venv\Scripts\python.exe -m unittest -v test_review test_fractional_bounds
.\.venv\Scripts\python.exe paper/tables.py
~~~

To check the archived release directly, run the data acquisition step and then

~~~powershell
.\.venv\Scripts\python.exe -m unittest -v test_diffusion test_review test_fractional_bounds
~~~

The tests enumerate the 768 isolate-free five-vertex graphs under a fixed 3:2 label split, check the cycle construction at depths 1–8, reoptimize all 15 empirical endpoint pairs, and independently check the review witnesses. The eight-vertex enumeration uses exact fractions. An independent search over all edge pairs verifies the completeness of each depth-three switch list and checks the saved label-preserving permutations. The LP tests rerun both endpoints on all five graphs and independently evaluate the Wisconsin upper certificate with exact fractions.

Build the paper from the root using Tectonic on paper/manuscript.tex. The vector illustration is included. Its optional regeneration script uses matplotlib with a noninteractive backend. Regenerating the case studies replaces local result files. Wall times and the choice among tied optimal witnesses can vary.

## Code and results

| Analysis | Entry point | Archived output |
| --- | --- | --- |
| Primary bounds, endpoint graphs, stub bounds and 96 constrained-switch draws per instance | [experiment.py](experiment.py), [primary protocol](PROTOCOL.md) | [case_studies.json](results/case_studies.json) |
| Maximum replacement of original edges, one-switch effects and five-step witness values | [diagnostics.py](diagnostics.py) | [diagnostics.json](results/diagnostics.json) |
| Independent rational endpoint enumeration at depths two and three | [exact_check.py](exact_check.py) | [rational.json](results/rational.json) |
| Fractional-edge LP endpoints, independent-label reference, and a rational Wisconsin upper certificate | [fractional_bounds.py](fractional_bounds.py) | [fractional_bounds.json](results/fractional_bounds.json) |
| Eight-vertex control, depth-three isomorphisms and connected witnesses | [review_checks.py](review_checks.py) | [review.json](results/review.json) |
| Connected Cornell maximum with component cuts | [connected_bound.py](connected_bound.py), called by the review program | [review.json](results/review.json), under **cornell_connected_maximum** |

The fractional-edge comparison is a post-primary analysis. It preserves all per-vertex label quotas and allows each candidate edge to have any weight in [0,1]. The LP resolves the sign relative to independent stationary labels on all five graphs. Only the Cornell and Wisconsin upper endpoints differ numerically from the simple-graph optima. The Wisconsin record includes rational vertex prices, indexed by block and vertex, and the resulting upper bound. These prices can be checked against the input graph without invoking an optimizer.

The connectivity and isomorphism review was performed after the primary study. Its result file contains the counts and representative cross-neighbor sets for the control, one permutation for each depth-three switch, and the moves that turn saved depth-one witnesses into connected graphs. To replay a switch (u,v,x,y), replace (u,v) and (x,y) by (u,y) and (x,v). Unlisted vertices in a permutation retain their names.

Cornell is the only case where requiring connectedness changes a three-step endpoint. The initial review bracketed the connected maximum between **22.0317%** (a connected witness) and **22.3900%** (the unrestricted maximum). Excluding leaf–leaf edges and enforcing a neighbor in the non-leaf core for every non-leaf strengthened the component-cut model. It closed the gap after 105 solves. The connected maximizing graph has exact witness score **790411/3580225**, or **22.0771%**, and is stored with the numerical solver bound. The minimum and both endpoints on the other four graphs have connected witnesses attaining the unrestricted values.

## Inputs and scope

The data comprise an observed club split, college football conferences and three campuses from one collection of labeled webpages. The code merges reciprocal edges, drops loops and isolates, and discards edge weights. The manuscript concerns stationary diffusion on observed labels. [Geom-GCN WebKB inputs](https://github.com/bingzhewei/geom-gcn/tree/1124af17444d7fd09686504ec46caa3f39f4f632/new_data) · [Newman's network data](https://websites.umich.edu/~mejn/netdata/) · [NetworkX karate data](https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html).
