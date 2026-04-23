class AONode:
    def __init__(self, name, heuristic):
        self.name = name
        self.heuristic = heuristic
        self.children = []   # list of AND/OR options
        self.solution = None

# ----------------------------
# Build Graph
# ----------------------------
# Graph structure:
# A -> [(B AND C), (D)]
# B -> [(E), (F)]
# C, D, E, F are terminal

A = AONode('A', 10)
B = AONode('B', 6)
C = AONode('C', 2)
D = AONode('D', 4)
E = AONode('E', 0)
F = AONode('F', 0)

# Define AND-OR relationships
A.children = [[B, C], [D]]   # OR between groups
B.children = [[E], [F]]      # OR
C.children = []
D.children = []
E.children = []
F.children = []

# ----------------------------
# AO* Algorithm
# ----------------------------
def ao_star(node):
    # If leaf node
    if not node.children:
        return node.heuristic

    min_cost = float('inf')
    best_group = None

    # Check each AND group
    for group in node.children:
        cost = 0
        for child in group:
            cost += ao_star(child)

        if cost < min_cost:
            min_cost = cost
            best_group = group

    node.solution = best_group
    node.heuristic = min_cost

    return node.heuristic

# ----------------------------
# Run AO*
# ----------------------------
cost = ao_star(A)

# ----------------------------
# Print Solution Path
# ----------------------------
def print_solution(node):
    if not node.solution:
        print(node.name, end=" ")
        return

    print(node.name, "-> ", end="")
    for child in node.solution:
        print_solution(child)

print("Optimal Cost:", cost)
print("Solution Path:")
print_solution(A)
