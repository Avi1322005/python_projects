import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------
# FIXED 10x10 MAZE
# ----------------------------
MAZE = np.array([
    [0,0,0,1,0,0,0,0,0,0],
    [1,1,0,1,0,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,1,0,1,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,1,1,1,1,0,1,1,1,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,1,1,0,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,1,0,0,0],
])

ROWS, COLS = MAZE.shape
START = (0, 0)
GOAL  = (9, 9)

MOVES = [(-1,0),(1,0),(0,-1),(0,1)]

# ----------------------------
# STATE SPACE
# ----------------------------
def in_bounds(r,c):
    return 0 <= r < ROWS and 0 <= c < COLS

def passable(r,c):
    return MAZE[r,c] == 0

# All states
states = [(r,c) for r in range(ROWS) for c in range(COLS) if passable(r,c)]

# Transition function (adjacency list)
adj = {}
for s in states:
    r,c = s
    neighbors = []
    for dr,dc in MOVES:
        nr,nc = r+dr, c+dc
        if in_bounds(nr,nc) and passable(nr,nc):
            neighbors.append((nr,nc))
    adj[s] = neighbors

print("Total states (free cells):", len(states))

print("\nSample state transitions:")
for s in states[:10]:
    print(s, "->", adj[s])

# ----------------------------
# BUILD GRAPH
# ----------------------------
G = nx.Graph()
for s in states:
    G.add_node(s)
    for t in adj[s]:
        G.add_edge(s,t)

# ----------------------------
# VISUALIZE MAZE
# ----------------------------
plt.figure(figsize=(6,6))
plt.title("Fixed 10x10 Maze")
plt.imshow(MAZE, origin="upper")
plt.scatter(START[1], START[0], s=120)
plt.scatter(GOAL[1], GOAL[0], s=120)
plt.xticks([]); plt.yticks([])
plt.show()

# ----------------------------
# VISUALIZE STATE SPACE GRAPH
# ----------------------------
plt.figure(figsize=(8,8))
plt.title("State Space Graph")

# Position nodes by grid coordinates
pos = { (r,c):(c,-r) for (r,c) in G.nodes() }

nx.draw_networkx_nodes(G,pos,node_size=250)
nx.draw_networkx_edges(G,pos,alpha=0.6)
nx.draw_networkx_labels(G,pos,font_size=7)

plt.axis("off")
plt.show()
