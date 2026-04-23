import heapq

# Represent state as tuple of 3 tuples (pegs)
# Example: ((3,2,1), (), ())  -> all disks on first peg

def heuristic(state, goal):
    # count disks not in correct position
    return sum(1 for peg_s, peg_g in zip(state, goal) if peg_s != peg_g)

def get_neighbors(state):
    neighbors = []
    pegs = list(state)

    for i in range(3):  # from peg
        if not pegs[i]:
            continue
        disk = pegs[i][-1]

        for j in range(3):  # to peg
            if i != j:
                if not pegs[j] or pegs[j][-1] > disk:
                    new_pegs = [list(p) for p in pegs]
                    new_pegs[j].append(new_pegs[i].pop())
                    neighbors.append(tuple(tuple(p) for p in new_pegs))

    return neighbors

def a_star(start, goal):
    pq = []
    heapq.heappush(pq, (0, start))

    g_cost = {start: 0}
    parent = {start: None}

    while pq:
        _, current = heapq.heappop(pq)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for neighbor in get_neighbors(current):
            new_cost = g_cost[current] + 1

            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                f_cost = new_cost + heuristic(neighbor, goal)
                heapq.heappush(pq, (f_cost, neighbor))
                parent[neighbor] = current

    return None

# --------- Run ---------
n = 3  # number of disks

start = (tuple(range(n, 0, -1)), (), ())
goal = ((), (), tuple(range(n, 0, -1)))

path = a_star(start, goal)

# Print steps
for step in path:
    print(step)
