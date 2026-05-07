import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
def generate_random_maze(rows, cols, wall_probability=0.28):
    maze = []

    for r in range(rows):
        row = []
        for c in range(cols):
            if random.random() < wall_probability:
                row.append('#')
            else:
                row.append('.')
        maze.append(row)

    maze[0][0] = 'S'
    maze[rows - 1][cols - 1] = 'G'

    # Random special positions
    special = []

    while len(special) < 3:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)

        if maze[r][c] == '.':
            special.append((r, c))

    (r1, c1), (r2, c2), (r3, c3) = special

    maze[r1][c1] = '1'
    maze[r2][c2] = '2'
    maze[r3][c3] = 'E'

    return ["".join(row) for row in maze]
maze = generate_random_maze(10, 12)
ROWS, COLS = len(maze), len(maze[0])

def find_pos(ch):
    for r in range(ROWS):
        for c in range(COLS):
            if maze[r][c] == ch:
                return (r, c)
    return None

start = find_pos('S')
goal = find_pos('G')
enemy = find_pos('E')
key1 = find_pos('1')   # because K1 written as K and 1
key2 = find_pos('2')

def is_valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and maze[r][c] != '#'

def neighbors(pos):
    r, c = pos
    moves = [(1,0), (-1,0), (0,1), (0,-1)]
    result = []
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if is_valid(nr, nc):
            result.append((nr, nc))
    return result

def reconstruct(parent, end):
    path = []
    while end is not None:
        path.append(end)
        end = parent.get(end)
    return path[::-1]

def bfs(start, goal):
    q = deque([start])
    parent = {start: None}
    visited = []

    while q:
        curr = q.popleft()
        visited.append(curr)

        if curr == goal:
            return reconstruct(parent, goal), visited

        for nxt in neighbors(curr):
            if nxt not in parent:
                parent[nxt] = curr
                q.append(nxt)

    return [], visited

def dfs(start, goal):
    stack = [start]
    parent = {start: None}
    visited = []

    while stack:
        curr = stack.pop()
        visited.append(curr)

        if curr == goal:
            return reconstruct(parent, goal), visited

        for nxt in neighbors(curr):
            if nxt not in parent:
                parent[nxt] = curr
                stack.append(nxt)

    return [], visited

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal):
    pq = []
    heapq.heappush(pq, (0, start))
    parent = {start: None}
    cost = {start: 0}
    visited = []

    while pq:
        _, curr = heapq.heappop(pq)
        visited.append(curr)

        if curr == goal:
            return reconstruct(parent, goal), visited

        for nxt in neighbors(curr):
            new_cost = cost[curr] + 1

            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(pq, (priority, nxt))
                parent[nxt] = curr

    return [], visited

def ao_star():
    # AO* style task planning:
    # Goal can be reached through either Key1 or Key2.
    # Choose cheaper option.
    path1a, visit1a = astar(start, key1)
    path1b, visit1b = astar(key1, goal)

    path2a, visit2a = astar(start, key2)
    path2b, visit2b = astar(key2, goal)

    full1 = path1a + path1b[1:] if path1a and path1b else []
    full2 = path2a + path2b[1:] if path2a and path2b else []

    visited = visit1a + visit1b + visit2a + visit2b

    if not full1:
        return full2, visited
    if not full2:
        return full1, visited

    return (full1 if len(full1) <= len(full2) else full2), visited


def visualize(path, visited, title):

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background grid
    grid = np.zeros((ROWS, COLS))

    for r in range(ROWS):
        for c in range(COLS):

            if maze[r][c] == '#':
                grid[r][c] = 1

    # Draw maze
    ax.imshow(grid, cmap="binary")

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=1)

    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    # Animate visited nodes
    for i, (r, c) in enumerate(visited):

        ax.scatter(
            c,
            r,
            s=180,
            color="deepskyblue",
            marker="s",
            edgecolors="black"
        )

        plt.pause(0.02)

    # Animate final path
    for r, c in path:

        ax.scatter(
            c,
            r,
            s=220,
            color="gold",
            marker="o",
            edgecolors="black"
        )

        plt.pause(0.05)

    # Start node
    sr, sc = start
    ax.scatter(
        sc,
        sr,
        s=400,
        color="lime",
        marker="*",
        edgecolors="black",
        label="Start"
    )

    # Goal node
    gr, gc = goal
    ax.scatter(
        gc,
        gr,
        s=400,
        color="red",
        marker="X",
        edgecolors="black",
        label="Goal"
    )

    # Key nodes
    if key1:
        kr, kc = key1
        ax.scatter(
            kc,
            kr,
            s=250,
            color="orange",
            marker="P",
            edgecolors="black",
            label="Key1"
        )

    if key2:
        kr, kc = key2
        ax.scatter(
            kc,
            kr,
            s=250,
            color="violet",
            marker="P",
            edgecolors="black",
            label="Key2"
        )

    # Enemy
    if enemy:
        er, ec = enemy
        ax.scatter(
            ec,
            er,
            s=300,
            color="black",
            marker="D",
            edgecolors="white",
            label="Enemy"
        )

    ax.set_title(
        f"{title} Visualization",
        fontsize=18,
        fontweight='bold'
    )

    ax.legend(loc="upper right")

    plt.show()
    
def run_algorithm(name, func):
    start_time = time.time()
    path, visited = func()
    end_time = time.time()

    return {
        "Algorithm": name,
        "Path Length": len(path),
        "Visited Nodes": len(visited),
        "Time": round(end_time - start_time, 6),
        "Path": path,
        "Visited": visited
    }

results = []

results.append(run_algorithm("BFS", lambda: bfs(start, goal)))
results.append(run_algorithm("DFS", lambda: dfs(start, goal)))
results.append(run_algorithm("A*", lambda: astar(start, goal)))
results.append(run_algorithm("AO*", ao_star))

for res in results:
    visualize(res["Path"], res["Visited"], res["Algorithm"])

print("\nComparison Table:")
print("-" * 60)
print(f"{'Algorithm':<12}{'Path Length':<15}{'Visited':<12}{'Time'}")
print("-" * 60)

for res in results:
    print(f"{res['Algorithm']:<12}{res['Path Length']:<15}{res['Visited Nodes']:<12}{res['Time']}")

valid_results = [r for r in results if r["Path Length"] > 0]

best = min(valid_results, key=lambda x: (x["Path Length"], x["Visited Nodes"], x["Time"]))

print("\nBest Algorithm:")
print(best["Algorithm"])
print("Reason: Shortest path with fewer visited nodes and less time.")