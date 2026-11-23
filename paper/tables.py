"""Rebuild manuscript tables from the checked Windows outputs (standard library)."""
from pathlib import Path
import json

paper = Path(__file__).resolve().parent
root = paper.parent
runs = json.loads((root / "results/case_studies.json").read_text())["runs"]
diagnostics = json.loads((root / "results/diagnostics.json").read_text())["runs"]
diagnostics = {(x["dataset"], x["depth"]): x for x in diagnostics}
span = lambda pair: 100 * (max(pair) - min(pair))
primary, depth_rows = [], []
for row in runs:
    name, r = row["dataset"], row["depth"]
    diagnostic = diagnostics[name, r]
    if r == 1:
        entries = [name.title(), f'{row["nodes"]}/{row["edges"]}', str(row["classes"])]
        values = [100*row["observed"][-1], 100*row["bounds"][0], 100*row["bounds"][1],
                  span(row["sample_range"]), span(row["outer"]), 100*row["independent_label_agreement"]]
        primary.append(" & ".join(entries + [f"{x:.2f}" for x in values]) + r" \\")
        second = next(x for x in runs if x["dataset"] == name and x["depth"] == 2)
        entries = [name.title(), f'{span(diagnostic["lag5_examples"]):.3f}', f'{span(second["bounds"]):.3f}']
        entries += [str(diagnostics[name, t]["max_deletions"]) for t in (1, 2, 3)]
        entries += [f'{100*x:+.3f}' for x in diagnostic["one_switch_deltas"]]
        depth_rows.append(" & ".join(entries) + r" \\")
(paper / "primary.tex").write_text(r"\newcommand{\primaryrows}{" + "\n" + "\n".join(primary) + "}\n")
(paper / "depth.tex").write_text(r"\newcommand{\depthrows}{" + "\n" + "\n".join(depth_rows) + "}\n")

