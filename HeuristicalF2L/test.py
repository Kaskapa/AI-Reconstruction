import copy
import json
from tqdm import tqdm
from rubiks_cube import Cube

actions = ["L", "R", "U", "D", "F", "B", "L'", "R'", "U'", "D'", "F'", "B'"]

def build_heuristic_db(cube, actions, max_moves=10, heuristic=None):
    if heuristic is None:
        heuristic = {}

    def get_f2l_state(cube):
        # corners = cube.get_corners()
        # state = [[0,0,0] for _ in range(8)]
        # for i, corner in enumerate(corners):
        #     if set(corner) == {0, 2, 3}:
        #         state[i] = list(corner)  # Preserve the order of colors
        #         break
        # return tuple(tuple(corner) for corner in state)  # Make it hashable
        edges = cube.get_edges()
        state = [[0, 0] for _ in range(12)]
        for i, edge in enumerate(edges):
            if set(edge) in [{0, 2}, {0, 3}, {2, 3}, {0, 4}, {0, 1}]:
                state[i] = list(edge)
        return tuple(tuple(edge) for edge in state)


    que = [(cube, 0, [], str(get_f2l_state(cube)))]
    node_count = sum([len(actions) ** (x + 1) for x in range(max_moves + 1)])

    with tqdm(total=node_count, desc='Heuristic DB') as pbar:
        while True:
            if not que:
                break
            current_cube, depth, prev_actions, previous_state = que.pop(0)  # Using pop(0) for breadth-first search

            if depth > max_moves:
                pbar.update(1)
                continue

            if depth == max_moves:
                pbar.update(1)
                continue

            for action in actions:
                new_cube = copy.deepcopy(current_cube)
                new_cube.do_moves(action)
                new_actions = prev_actions + [action]

                f2l_state = get_f2l_state(new_cube)
                state_str = str(f2l_state)

                if state_str == previous_state:
                    pbar.update(1)
                    continue

                if state_str not in heuristic or heuristic[state_str] > depth:
                    heuristic[state_str] = depth

                if state_str in heuristic and heuristic[state_str] < depth:
                    pbar.update(1)
                    continue

                que.append((new_cube, depth + 1, new_actions, state_str))
                pbar.update(1)

    return heuristic

MAX_MOVES = 6
HEURISTIC_FILE = "heuristic_f2l_one_pair_edge.json"

cube = Cube(0)
cube.do_algorithm("z2")
h_db = build_heuristic_db(cube, actions, max_moves=MAX_MOVES)

with open(HEURISTIC_FILE, 'w', encoding='utf-8') as f:
    json.dump(h_db, f, ensure_ascii=False, indent=4)

print(f"Generated {len(h_db)} heuristic states.")