import csv
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

for state in states:

    solver = CrossSolver(state)
    solver.solve()

    solution = solver.allSol[0].replace("\n", " ")

    print(state + " " + solution)

    with open('dataset.csv', mode='a') as file:
        file.write(state + " " + solution + "\n")