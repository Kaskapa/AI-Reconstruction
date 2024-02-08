import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Step 1: Data Preprocessing
def preprocess_data(file_path):
    file = open(file_path)
    csvreader = csv.reader(file)
    scramble = []
    stateStr = []
    for row in csvreader:
        scramble.append(str(row).split(";")[0][2:])
        stateStr.append(str(row).split(";")[1][:-2])

    state = []

    for row in stateStr:
        cube = []
        # Split the row by "|"
        faces = row.split("|")
        # Iterate through each face in the row and print it
        for face in faces:
            rows = face.split(" ")
            facesInt = []
            for row in rows:
                elements = [*row]
                desired_array = [int(numeric_string) for numeric_string in elements]
                facesInt.append(desired_array)
            cube.append(facesInt)
        state.append(cube)                    

    file.close()

    return scramble, state

# Step 2: Data Loading
class RubiksCubeDataset(Dataset):
    def __init__(self, states, scrambles):
        self.states = states
        self.scrambles = scrambles

        print(self.states)
        print(self.scrambles)

    def __len__(self):
        print(len(self.states))
        return len(self.states)

    def __getitem__(self, idx):
        if(idx >= 1000):
            print("opaa indeks out of bounds, sorry you are out of luck")
        else:
            print(self.states[1], self.scrambles[1])
            return self.states[idx], self.scrambles[idx]


# Example list of scrambles
scrambles = [
    "R U R' U' R' F R2 U' R' U' R U R' F'",
    "U D R2 B2 L2 U R' D' R2 F2 U' B2 D2 F2 L2 R2 U'"
]

# Tokenize each scramble by splitting on spaces
tokenized_scrambles = [scramble.split() for scramble in scrambles]

# Convert each tokenized sequence to a tensor
print(torch.tensor(tokenized_scrambles[0][1]))
tensorized_scrambles = [torch.tensor([move for move in scramble]) for scramble in tokenized_scrambles]

# Pad the sequences to a maximum length of 25 moves
padded_scrambles = pad_sequence(tensorized_scrambles, batch_first=True, padding_value=0)

# Print the padded scrambles
print(padded_scrambles)

