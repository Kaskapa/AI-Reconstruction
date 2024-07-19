from rubiks_cube import Cube
from tqdm import tqdm
import json
import copy

# cube = Cube(0)
# cube.do_algorithm("z2")

# edges = cube.get_edges()
# corners = cube.get_corners()

# for edge in range(len(edges)):
#     if (2 not in edges[edge] or 3 not in edges[edge]) and 0 not in edges[edge]:
#         edges[edge] = 0

# for corner in range(len(corners)):
#     if 2 not in corners[corner] or 3 not in corners[corner] or 0 not in corners[corner]:
#         corners[corner] = 0

# print(edges)
# print(corners)

# for face in cube.cube:
#     for row in face:
#         print(row)
#     print()

actions = ["L", "R", "U", "D", "F", "B", "L'", "R'", "U'", "D'", "F'", "B'"]

def build_heuristic_db(cube, actions, max_moves=10, heuristic = None):
    if heuristic is None:
        corners = cube.get_corners()
        for corner in range(len(corners)):
            if 2 not in corners[corner] or 3 not in corners[corner] or 0 not in corners[corner]:
                corners[corner] = 0

        cornsersStr = str(corners)

        heuristic = {cornsersStr: 0}

    prev_actions = []

    que = [(cube, 0, prev_actions)]
    node_count = sum([len(actions) ** (x + 1) for x in range(max_moves + 1)])
    with tqdm(total=node_count, desc='Heuristic DB') as pbar:
        while True:
            if not que:
                break
            cube, depth, prev_actions = que.pop()
            if depth > max_moves:
                pbar.update(1)
                continue
            for action in actions:
                doableCube = copy.deepcopy(cube)
                doableCube.do_moves(action)
                prev_actions.append(action)

                if len(prev_actions) > 1 and prev_actions[-1][0] == prev_actions[-2][0]:
                    pbar.update(1)
                    continue

                corners = doableCube.get_corners()
                for corner in range(len(corners)):
                    if 2 not in corners[corner] or 3 not in corners[corner] or 0 not in corners[corner]:
                        corners[corner] = 0

                cornsersStr = str(corners)
                if cornsersStr not in heuristic or heuristic[cornsersStr] > depth + 1:
                    heuristic[cornsersStr] = depth + 1

                que.append((copy.deepcopy(doableCube), depth+1, prev_actions.copy()))
                pbar.update(1)
    return heuristic

MAX_MOVES = 5
NEW_HEURISTIC = False
HEURISTIC_FILE = "heuristic_easy_f2l.json"

cube = Cube(0)
cube.do_algorithm("z2")

h_db = build_heuristic_db(cube, actions, max_moves = MAX_MOVES, heuristic=None)

with open(HEURISTIC_FILE, 'w', encoding='utf-8') as f:
        json.dump(
            h_db,
            f,
            ensure_ascii=False,
            indent=4
        )