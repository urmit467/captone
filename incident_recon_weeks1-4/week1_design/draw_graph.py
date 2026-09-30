
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, networkx as nx

here = os.path.dirname(os.path.abspath(__file__))
g = json.load(open(os.path.join(here, "path_graph.json")))
G = nx.DiGraph(); G.add_edges_from(g["edges"])
pos = {"P1": (0, 1), "P2": (1, 1), "P3": (2, 1), "P4": (3.2, 1.6), "P5": (4.4, 0.3), "P6": (5.4, 1.0), "P7": (5.4, 2.2)}
fig, ax = plt.subplots(figsize=(10, 5))
hid = g["hidden_path"]
cols = ["#e74c3c" if n == hid else "#3498db" for n in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=cols, node_size=2600, ax=ax)
nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", ax=ax)
styles = ["dashed" if hid in e else "solid" for e in G.edges]
nx.draw_networkx_edges(G, pos, style=styles, arrows=True, arrowsize=22, node_size=2600, ax=ax, edge_color="#555")
for n, d in g["paths"].items():
    x, y = pos[n]
    label = f"{d['camera']} | {d['length_m']} m" if d["camera"] else f"NO CAMERA | {d['length_m']} m"
    ax.text(x, y - 0.32, label, ha="center", fontsize=9, color="#c0392b" if not d["camera"] else "#2c3e50")
ax.set_title("MVP path network (P4 = unobservable, incident location)")
ax.axis("off"); plt.tight_layout()
plt.savefig(os.path.join(here, "path_graph.png"), dpi=150)
print("saved path_graph.png")
