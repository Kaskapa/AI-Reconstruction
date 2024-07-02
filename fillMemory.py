import csv
from rubiks_cube import Cube
from collections import deque

ACTIONS = ["L", "L'", "L2", "R", "R'", "R2", "U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2", "x", "x'", "y", "y'", "z", "z'", "x2", "y2", "z2", "M", "M'", "M2", "E", "E'", "E2", "S", "S'", "S2", "l", "l'", "l2", "r", "r'", "r2", "u", "u'", "u2", "d", "d'", "d2", "f", "f'", "f2", "b", "b'", "b2"]

def fillMemory():
    with open("f2lPreTrain.csv", "r") as file:
        csvreader = csv.reader(file)
        expertSolutions = []
        for row in csvreader:
            expertSolutions.append(row)

    memory = deque(maxlen=100_000)

    actionArr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    total_reward = 0

    for solution in expertSolutions:

        cube = Cube(0)

        solutionArr = solution[0].split(";")
        for i in range(len(solutionArr) - 1 ):
            cube.do_algorithm(solutionArr[i])

        ROTATIONS = ["x", "x'", "y", "y'", "z", "z'", "x2", "y2", "z2"]

        index = 0

        while not cube.is_white_cross_solved() and index < len(ROTATIONS):
            cube = Cube(0)
            cube.do_moves(ROTATIONS[index])
            for i in range(len(solutionArr) - 1 ):
                cube.do_algorithm(solutionArr[i])
            index += 1

        f2lArr = solutionArr[-1].split(" ")
        move_count = 0

        for move in f2lArr:

            action = move

            for i in range(len(ACTIONS)):
                if action == ACTIONS[i]:
                    actionArr[i] = 1

            insertedPairs = cube.get_white_inserted_pairs()
            stateOld = [
                # #pair 1 paired
                # cube.is_pair_paired(0),

                # #pair 2 paired
                # cube.is_pair_paired(1),

                # #pair 3 paired
                # cube.is_pair_paired(2),

                # #pair 4 paired
                # cube.is_pair_paired(3),

                # #pair 5 paired
                # cube.is_pair_paired(4),

                # #pair 6 paired
                # cube.is_pair_paired(5),

                # #pair 7 paired
                # cube.is_pair_paired(6),

                # #pair 8 paired
                # cube.is_pair_paired(7),

                # #pair 9 paired
                # cube.is_pair_paired(8),

                # #pair 10 paired
                # cube.is_pair_paired(9),

                # #pair 11 paired
                # cube.is_pair_paired(10),

                # #pair 12 paired
                # cube.is_pair_paired(11),

                # #pair 13 paired
                # cube.is_pair_paired(12),

                # #pair 14 paired
                # cube.is_pair_paired(13),

                # #pair 15 paired
                # cube.is_pair_paired(14),

                # #pair 16 paired
                # cube.is_pair_paired(15),

                # #pair 17 paired
                # cube.is_pair_paired(16),

                # #pair 18 paired
                # cube.is_pair_paired(17),

                # #pair 19 paired
                # cube.is_pair_paired(18),

                # #pair 20 paired
                # cube.is_pair_paired(19),

                # #pair 21 paired
                # cube.is_pair_paired(20),

                # #pair 22 paired
                # cube.is_pair_paired(21),

                # #pair 23 paired
                # cube.is_pair_paired(22),

                # #pair 24 paired
                # cube.is_pair_paired(23),

                # #pair 1 is inserted
                # 0 in insertedPairs,

                # #pair 2 is inserted
                # 1 in insertedPairs,

                # #pair 3 is inserted
                # 2 in insertedPairs,

                # #pair 4 is inserted
                # 3 in insertedPairs,

                # #pair 5 is inserted
                # 4 in insertedPairs,

                # #pair 6 is inserted
                # 5 in insertedPairs,

                # #pair 7 is inserted
                # 6 in insertedPairs,

                # #pair 8 is inserted
                # 7 in insertedPairs,

                # #pair 9 is inserted
                # 8 in insertedPairs,

                # #pair 10 is inserted
                # 9 in insertedPairs,

                # #pair 11 is inserted
                # 10 in insertedPairs,

                # #pair 12 is inserted
                # 11 in insertedPairs,

                # #pair 13 is inserted
                # 12 in insertedPairs,

                # #pair 14 is inserted
                # 13 in insertedPairs,

                # #pair 15 is inserted
                # 14 in insertedPairs,

                # #pair 16 is inserted
                # 15 in insertedPairs,

                # #pair 17 is inserted
                # 16 in insertedPairs,

                # #pair 18 is inserted
                # 17 in insertedPairs,

                # #pair 19 is inserted
                # 18 in insertedPairs,

                # #pair 20 is inserted
                # 19 in insertedPairs,

                # #pair 21 is inserted
                # 20 in insertedPairs,

                # #pair 22 is inserted
                # 21 in insertedPairs,

                # #pair 23 is inserted
                # 22 in insertedPairs,

                # #pair 24 is inserted
                # 23 in insertedPairs
            ]

            flattenedCube = []

            for face in cube.cube:
                for row in face:
                    for val in row:
                        flattenedCube.append(val)

            stateOld += flattenedCube

            cube.do_moves(action)

            reward = 0
            pairsInserted = cube.get_white_inserted_pairs()
            done = False
            move_count += 1
            if len(pairsInserted) > 0 and cube.is_white_cross_solved():
                for i in range(len(pairsInserted)):
                    reward += 10
            if (len(pairsInserted) > 0 and cube.is_white_cross_solved()) or move_count > 100:
                done = True
                if len(pairsInserted) == 0 or not cube.is_white_cross_solved():
                    reward -= 2
                else:
                    plusReward = 100-move_count*0.5

                    reward += plusReward

                move_count = 0

            insertedPairs = cube.get_white_inserted_pairs()
            next_state = [
                # #pair 1 paired
                # cube.is_pair_paired(0),

                # #pair 2 paired
                # cube.is_pair_paired(1),

                # #pair 3 paired
                # cube.is_pair_paired(2),

                # #pair 4 paired
                # cube.is_pair_paired(3),

                # #pair 5 paired
                # cube.is_pair_paired(4),

                # #pair 6 paired
                # cube.is_pair_paired(5),

                # #pair 7 paired
                # cube.is_pair_paired(6),

                # #pair 8 paired
                # cube.is_pair_paired(7),

                # #pair 9 paired
                # cube.is_pair_paired(8),

                # #pair 10 paired
                # cube.is_pair_paired(9),

                # #pair 11 paired
                # cube.is_pair_paired(10),

                # #pair 12 paired
                # cube.is_pair_paired(11),

                # #pair 13 paired
                # cube.is_pair_paired(12),

                # #pair 14 paired
                # cube.is_pair_paired(13),

                # #pair 15 paired
                # cube.is_pair_paired(14),

                # #pair 16 paired
                # cube.is_pair_paired(15),

                # #pair 17 paired
                # cube.is_pair_paired(16),

                # #pair 18 paired
                # cube.is_pair_paired(17),

                # #pair 19 paired
                # cube.is_pair_paired(18),

                # #pair 20 paired
                # cube.is_pair_paired(19),

                # #pair 21 paired
                # cube.is_pair_paired(20),

                # #pair 22 paired
                # cube.is_pair_paired(21),

                # #pair 23 paired
                # cube.is_pair_paired(22),

                # #pair 24 paired
                # cube.is_pair_paired(23),

                # #pair 1 is inserted
                # 0 in insertedPairs,

                # #pair 2 is inserted
                # 1 in insertedPairs,

                # #pair 3 is inserted
                # 2 in insertedPairs,

                # #pair 4 is inserted
                # 3 in insertedPairs,

                # #pair 5 is inserted
                # 4 in insertedPairs,

                # #pair 6 is inserted
                # 5 in insertedPairs,

                # #pair 7 is inserted
                # 6 in insertedPairs,

                # #pair 8 is inserted
                # 7 in insertedPairs,

                # #pair 9 is inserted
                # 8 in insertedPairs,

                # #pair 10 is inserted
                # 9 in insertedPairs,

                # #pair 11 is inserted
                # 10 in insertedPairs,

                # #pair 12 is inserted
                # 11 in insertedPairs,

                # #pair 13 is inserted
                # 12 in insertedPairs,

                # #pair 14 is inserted
                # 13 in insertedPairs,

                # #pair 15 is inserted
                # 14 in insertedPairs,

                # #pair 16 is inserted
                # 15 in insertedPairs,

                # #pair 17 is inserted
                # 16 in insertedPairs,

                # #pair 18 is inserted
                # 17 in insertedPairs,

                # #pair 19 is inserted
                # 18 in insertedPairs,

                # #pair 20 is inserted
                # 19 in insertedPairs,

                # #pair 21 is inserted
                # 20 in insertedPairs,

                # #pair 22 is inserted
                # 21 in insertedPairs,

                # #pair 23 is inserted
                # 22 in insertedPairs,

                # #pair 24 is inserted
                # 23 in insertedPairs
            ]

            flattenedCube = []

            for face in cube.cube:
                for row in face:
                    for val in row:
                        flattenedCube.append(val)

            next_state += flattenedCube

            total_reward += reward

            memory.append((stateOld, actionArr, reward, next_state, done))
    print(total_reward)

    return memory
