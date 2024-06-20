import csv
from rubiks_cube import Cube
from givemecrossUsable import CrossSolver

def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        stateStr = []
        for row in csvreader:
            row_str = row[0]
            # Convert the scramble to a list of integers
            stateStr.append(row_str.split(";")[0])

    return stateStr

import os

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
file_path = os.path.join(script_dir, "File(30).csv")

states = preprocess_data(file_path)

cube = Cube(0)

stateArr = states[0].split(" ")

for i in stateArr:
    cube.do_moves(action=i)

solver = CrossSolver(states[0])

solver.solve()

print(solver.allSol)