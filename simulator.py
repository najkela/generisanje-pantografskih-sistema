import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

data_json = {
    "nodes": {
        "0": [0, 0],
        "1": [15, 0],
        "2": [-38, -7.8],
        "3": [-10.8, -42.1],
        "4": [21.3, -51.5],
        "5": [-54.7, -43.2],
        "6": [-25.4, -90.6],
        "7": [-30.0, -135.0]
    },
    "edges": [
        (0, 1), (1, 3), (2, 3), (1, 4), (3, 4),
        (2, 5), (3, 5), (4, 6), (5, 6), (5, 7), (6, 7)
    ],
    "fixed": [0, 2],
    "crank": 1
}

nodes = {int(k): np.array(v, dtype=float) for k, v in data_json["nodes"].items()}
edges = data_json["edges"]
fixed = data_json["fixed"]
crank_idx = data_json["crank"]

lengths = {}
for edge in edges:
    p1, p2 = nodes[edge[0]], nodes[edge[1]]
    lengths[edge] = np.linalg.norm(p1 - p2)

solving_order = [
    {'target': 3, 'parents': (1, 2), 'len': (lengths[(1,3)], lengths[(2,3)])},
    {'target': 4, 'parents': (1, 3), 'len': (lengths[(1,4)], lengths[(3,4)])},
    {'target': 5, 'parents': (2, 3), 'len': (lengths[(2,5)], lengths[(3,5)])},
    {'target': 6, 'parents': (4, 5), 'len': (lengths[(4,6)], lengths[(5,6)])},
    {'target': 7, 'parents': (5, 6), 'len': (lengths[(5,7)], lengths[(6,7)])}
]

def smart_intersect(p1, p2, r1, r2, target_idx):
    d = np.linalg.norm(p2 - p1)
    if d > r1 + r2 or d < abs(r1 - r2): return nodes[target_idx]

    l = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, r1**2 - l**2))
    base = p1 + l * (p2 - p1) / d
    v_norm = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / d

    s1, s2 = base + h * v_norm, base - h * v_norm
    chosen = s1 if np.linalg.norm(s1 - nodes[target_idx]) < np.linalg.norm(s2 - nodes[target_idx]) else s2
    nodes[target_idx] = chosen
    return chosen

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-120, 60); ax.set_ylim(-130, 50); ax.set_aspect('equal')
lines = [ax.plot([], [], 'o-', lw=2, color="blue")[0] for _ in edges]

def update(angle):
    r_crank = lengths[(0, 1)]
    nodes[crank_idx] = nodes[0] + np.array([r_crank * np.cos(angle), r_crank * np.sin(angle)])

    for task in solving_order:
        p1, p2 = nodes[task['parents'][0]], nodes[task['parents'][1]]
        smart_intersect(p1, p2, task['len'][0], task['len'][1], task['target'])

    for i, (n1, n2) in enumerate(edges):
        lines[i].set_data([nodes[n1][0], nodes[n2][0]], [nodes[n1][1], nodes[n2][1]])

    return lines

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 200), blit=True, interval=20)
plt.title("Пантографски симулатор")
plt.show()
