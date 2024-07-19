import csv
import random
from rubiks_cube import Cube

notations = ["L", "L'", "L2", "R", "R'", "R2", "U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"];

def readOutputFile(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        lines = [row for row in reader]
        return lines

allLines = readOutputFile("output.csv")

def writeOutputFile(file_path, lines):
    new_lines = []
    for line in lines:
        cube = Cube(0)
        cube.do_algorithm(line[0])
        cube.do_algorithm(line[1])

        moves = []

        index = 0

        while(len(cube.get_white_inserted_pairs()) > 0 and cube.is_white_cross_solved()) or index < 2:
            move  = random.choice(notations)
            cube.do_moves(move)
            moves.append(move)
            index += 1

        new_lines.append([line[0], line[1], " ".join(moves)])

    with open(file_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_lines)
    return new_lines

newLines = writeOutputFile("output1.csv", allLines)

print(newLines)