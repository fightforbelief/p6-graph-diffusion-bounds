"""Rebuild both manuscript tables from the archived case studies and diagnostics."""
from pathlib import Path
import json

paper = Path(__file__).resolve().parent
results = paper.parent / "results"
primary = json.loads((results / "case_studies.json").read_text())["runs"]
cases = {(r["dataset"], r["depth"]): r for r in primary}
supplement = json.loads((results / "diagnostics.json").read_text())["runs"]
diagnostics = {(r["dataset"], r["depth"]): r for r in supplement}


def span(values):
    return 100 * (max(values) - min(values))


def interval(pair):
    return f"{100*pair[0]:.2f}--{100*pair[1]:.2f}"


def save_rows(file, macro, rows):
    text = r"\newcommand{" + macro + "}{\n" + "\n".join(rows) + "}\n"
    (paper / file).write_text(text)


three, five = [], []
for row in primary:
    name = row["dataset"]
    if row["depth"] != 1:
        continue
    dataset = name.title()
    entries = [dataset, f'{row["nodes"]}/{row["edges"]}', f'{100*row["observed"][-1]:.2f}',
               interval(row["bounds"]), interval(row["outer"]),
               f'{100*row["independent_label_agreement"]:.2f}', f'{span(row["sample_range"]):.2f}']
    three.append(" & ".join(entries) + r" \\")
    diagnostic = diagnostics[name, 1]
    entries = [dataset, f'{span(diagnostic["lag5_examples"]):.3f}',
               f'{span(cases[name, 2]["bounds"]):.3f}']
    entries.extend(str(diagnostics[name, r]["max_deletions"]) for r in (1, 2, 3))
    entries.extend(f'{100*x:+.3f}' for x in diagnostic["one_switch_deltas"])
    five.append(" & ".join(entries) + r" \\")
save_rows("primary.tex", r"\primaryrows", three)
save_rows("depth.tex", r"\depthrows", five)
