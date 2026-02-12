import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# 30x30 Random Grid World
# ----------------------------
ROWS, COLS = 30, 30
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R

def generate_grid(rows=30, cols=30, wall_prob=0.22, num_goals=6):
    """Generates a random grid with walls, one start S, and multiple goals G."""
    grid = [["." for _ in range(cols)] for _ in range(rows)]

    # Place walls
    for r in range(rows):
        for c in range(cols):
            if random.random() < wall_prob:
                grid[r][c] = "#"

    # Place start
    while True:
        sr, sc = random.randrange(rows), random.randrange(cols)
        if grid[sr][sc] == ".":
            grid[sr][sc] = "S"
            start = (sr, sc)
            break

    # Place goals
    goals = set()
    while len(goals) < num_goals:
        gr, gc = random.randrange(rows), random.randrange(cols)
        if grid[gr][gc] == ".":
            grid[gr][gc] = "G"
            goals.add((gr, gc))

    return grid, start, goals

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def passable(grid, r, c):
    return grid[r][c] != "#"

def bfs(grid, start):
    """
    BFS returns dist, parent pointers for shortest paths, AND exploration order
    so we can animate the search wavefront.
    """
    q = deque([start])
    dist = {start: 0}
    parent = {start: None}
    exploration_order = []

    while q:
        r, c = q.popleft()
        cur = (r, c)
        exploration_order.append(cur)

        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if in_bounds(nr, nc) and passable(grid, nr, nc) and nxt not in dist:
                dist[nxt] = dist[cur] + 1
                parent[nxt] = cur
                q.append(nxt)

    return dist, parent, exploration_order

def reconstruct_path(goal, parent):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def grid_to_base_image(grid):
    """
    base image values:
      0 = free
      1 = wall
    """
    base = np.zeros((ROWS, COLS), dtype=int)
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == "#":
                base[r, c] = 1
    return base

def make_random_reachable_instance(max_tries=200, wall_prob=0.22, num_goals=6):
    """
    Keeps generating random grids until at least one goal is reachable.
    Returns (grid, start, goals, chosen_goal, path, exploration_order).
    """
    for _ in range(max_tries):
        grid, start, goals = generate_grid(ROWS, COLS, wall_prob=wall_prob, num_goals=num_goals)
        dist, parent, exploration_order = bfs(grid, start)
        reachable = [g for g in goals if g in dist]
        if reachable:
            chosen_goal = random.choice(reachable)
            path = reconstruct_path(chosen_goal, parent)
            return grid, start, goals, chosen_goal, path, exploration_order

    raise RuntimeError("Couldn't generate a reachable goal. Try lowering wall_prob (e.g., 0.15).")

def animate(grid, start, goals, goal, path, exploration, interval_ms=40):
    base = grid_to_base_image(grid)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f"BFS Exploration → Shortest Path\nstart {start} → goal {goal} | steps={len(path)-1}")
    ax.imshow(base, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])

    # Goals (stars)
    ax.scatter([g[1] for g in goals], [g[0] for g in goals], marker="*", s=160, label="Goals (G)")

    # Explored nodes (animated scatter)
    explored_scatter = ax.scatter([], [], s=15, alpha=0.4, label="Explored")

    # Planned path line (draw after exploration completes)
    path_line, = ax.plot([], [], linewidth=2, label="Shortest path")

    # Agent dot (animated)
    agent = ax.scatter([start[1]], [start[0]], s=140, marker="o", label="Agent")

    ax.legend(loc="upper right")

    total_frames = len(exploration) + len(path)

    def update(i):
        if i < len(exploration):
            # Exploration phase
            explored_points = exploration[: i + 1]
            xs = [p[1] for p in explored_points]
            ys = [p[0] for p in explored_points]
            explored_scatter.set_offsets(np.column_stack((xs, ys)))

            # Keep path hidden until exploration done
            path_line.set_data([], [])
        else:
            # Path phase
            if i == len(exploration):
                # When switching to path phase, draw the whole planned path
                xs = [p[1] for p in path]
                ys = [p[0] for p in path]
                path_line.set_data(xs, ys)

            idx = i - len(exploration)
            if idx < len(path):
                r, c = path[idx]
                agent.set_offsets([[c, r]])

        return explored_scatter, path_line, agent

    anim = FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=True, repeat=False)
    plt.show()

# ----------------------------
# Run: RANDOM every time
# ----------------------------
if __name__ == "__main__":
    # Randomness: no fixed seed => different animation each run
    grid, start, goals, goal, path, exploration = make_random_reachable_instance(
        max_tries=200,
        wall_prob=0.22,   # increase for harder mazes, decrease for easier
        num_goals=6
    )

    print("Start:", start)
    print("All goals:", goals)
    print("Chosen reachable goal:", goal)
    print("Steps:", len(path) - 1)
    print("Explored nodes:", len(exploration))

    animate(grid, start, goals, goal, path, exploration, interval_ms=40)
