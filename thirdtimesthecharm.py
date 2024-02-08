import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# Step 1: Data Preprocessing
def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        scramble = []
        stateStr = []
        for row in csvreader:
            row_str = row[0]
            scramble.append(row_str.split(";")[0])
            stateStr.append(row_str.split(";")[1])

        state = []

        for row in stateStr:
            cube = []
            faces = row.split("|")
            for face in faces:
                rows = face.split(" ")
                facesInt = []
                for row in rows:
                    elements = [*row]
                    desired_array = [int(numeric_string) for numeric_string in elements]
                    facesInt.append(desired_array)
                cube.append(facesInt)
            state.append(cube)

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


# Step 3: Model Definition
class RubiksCubeModel(nn.Module):
    def __init__(self):
        super(RubiksCubeModel, self).__init__()

    def forward(self, x):
        return x

# Step 4: Model Training
def train_model(model, train_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for states, scrambles in train_loader:
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, scrambles)
            loss.backward()
            optimizer.step()

# Step 5: Model Testing
def test_model(model, test_loader):
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for states, scrambles in test_loader:
            outputs = model(states)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += scrambles.size(0)
            total_correct += (predicted == scrambles).sum().item()

        accuracy = total_correct / total_samples
        print(f"Test Accuracy: {accuracy * 100}%")

# Step 6: Model Usage
def get_scramble(model, cube_state):
    with torch.no_grad():
        output = model(cube_state)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()
