"""Draw the depth-one connected counterexample using named cycle vertices."""
from pathlib import Path
import math
import matplotlib.pyplot as plt

orders = [list(range(12)), [0, 1, 7, 6, 5, 4, 3, 2, 8, 9, 10, 11]]
pairs = [{(1, 2), (7, 8)}, {(1, 7), (2, 8)}]
titles = ["Before the switch", "After the switch"]
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.35))
for ax, order, changed, title in zip(axes, orders, pairs, titles):
    positions = {v: (math.sin(j * math.pi / 6), math.cos(j * math.pi / 6))
                 for j, v in enumerate(order)}
    for u, v in zip(order, order[1:] + order[:1]):
        edited = tuple(sorted((u, v))) in changed
        ax.plot(*zip(positions[u], positions[v]), color="#b45029" if edited else "#788185",
                linewidth=2.6 if edited else 1.1, zorder=1)
    for v, (x, y) in positions.items():
        marked = v in (0, 6)
        ax.scatter(x, y, s=142, c="#233b50" if marked else "white",
                   edgecolor="#233b50", linewidth=0.7, zorder=2)
        ax.text(x, y, str(v), color="white" if marked else "#233b50",
                fontsize=7.2, ha="center", va="center", zorder=3)
    ax.set(xlim=(-1.18, 1.18), ylim=(-1.18, 1.18), aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
fig.subplots_adjust(left=.04, right=.96, top=.87, bottom=.01, wspace=.23)
fig.savefig(Path(__file__).with_name("cycle.pdf"), bbox_inches="tight", pad_inches=.02)

