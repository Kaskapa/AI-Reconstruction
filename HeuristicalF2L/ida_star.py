import json
from tqdm import tqdm
import copy
from rubiks_cube import Cube

# Load heuristic databases
with open('C:/Users/krist/Documents/GitHub/AI-Reconstruction/heuristic_f2l_one_pair_safe.json', 'r') as f:
    corner_pdb = json.load(f)

with open('C:/Users/krist/Documents/GitHub/AI-Reconstruction/heuristic_f2l_one_pair_edge_safe.json', 'r') as f:
    edge_pdb = json.load(f)

# Define actions
actions = ["L", "R", "U", "D", "F", "B", "L'", "R'", "U'", "D'", "F'", "B'", "L2", "R2", "U2", "D2", "F2", "B2"]

# Function to get combined heuristic value
def get_combined_heuristic(corner_state, edge_state):
    corner_heuristic = corner_pdb.get(str(corner_state), float('inf'))
    edge_heuristic = edge_pdb.get(str(edge_state), float('inf'))
    return max(corner_heuristic, edge_heuristic)


# Define goal state for corners and edges (example)
goal_corner_state = ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 3, 2), (0, 0, 0))
goal_edge_state = ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (2, 3), (0, 0), (0, 0), (0, 2), (0, 3), (0, 1), (0, 4))

def is_goal_state(corner_state, edge_state):
    return corner_state == goal_corner_state and edge_state == goal_edge_state

def get_corner_and_edge_state(cube):
    return get_corner_state(cube), get_edge_state(cube)

def get_edge_state(cube):
    edges = cube.get_edges()
    state = [[0, 0] for _ in range(12)]
    for i, edge in enumerate(edges):
        if set(edge) in [{0, 2}, {0, 3}, {2, 3}, {0, 4}, {0, 1}]:
            state[i] = list(edge)
    return tuple(tuple(edge) for edge in state)

def get_corner_state(cube):
    corners = cube.get_corners()
    state = [[0,0,0] for _ in range(8)]
    for i, corner in enumerate(corners):
        if set(corner) == {0, 2, 3}:
            state[i] = list(corner)  # Preserve the order of colors
            break
    return tuple(tuple(corner) for corner in state)  # Make it hashable

def ida_star(cube):
    def search(path, g, threshold):
        node = path[-1]
        corner_state, edge_state = get_corner_and_edge_state(node)
        f = g + get_combined_heuristic(corner_state, edge_state)
        if f > threshold:
            return f
        if is_goal_state(corner_state, edge_state):
            return True
        min_threshold = float('inf')
        for move in actions:
            new_cube = copy.deepcopy(node)
            new_cube.do_moves(move)
            if new_cube not in path:  # Prevent cycles
                path.append(new_cube)
                t = search(path, g + 1, threshold)
                if t == True:
                    return True
                if t < min_threshold:
                    min_threshold = t
                path.pop()
        return min_threshold

    start = cube
    threshold = get_combined_heuristic(*get_corner_and_edge_state(start))
    path = [start]
    while True:
        t = search(path, 0, threshold)
        if t == True:
            return path
        if t == float('inf'):
            return None
        threshold = t

# Example usage:
cube = Cube(0)
cube.do_algorithm("z2 y' R U R' U2' R U R' U' R U R y")
cube.previous_moves = []
solution_path = ida_star(cube)

if solution_path:
    moves = []
    for state in solution_path[1:]:
        moves.append(state.previous_moves)  # Assuming each state knows the last move applied
    print("Solution found:", moves)
else:
    print("No solution found within the given threshold.")
