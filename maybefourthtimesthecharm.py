import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader
import csv
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# R: 1, R':2, L:3, L':4, U:5, U':6, D:7, D':8, F:9, F':10, B:11, B':12, R2:13, L2:14, U2:15, D2:16, F2:17, B2:18

# Preprocess the data
def preprocess_data(file_path):
    # Define the mapping from moves to integers
    move_to_int = {"R": 1, "R'": 2, "L": 3, "L'": 4, "U": 5, "U'": 6, "D": 7, "D'": 8, "F": 9, "F'": 10, "B": 11, "B'": 12, "R2": 13, "L2": 14, "U2": 15, "D2": 16, "F2": 17, "B2": 18}

    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        scramble = []
        stateStr = []
        for row in csvreader:
            row_str = row[0]
            # Convert the scramble to a list of integers
            scramble.append([move_to_int[move] for move in row_str.split(";")[0].split()])
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

# Define the RNN model
class RubiksCubeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RubiksCubeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state

        out, _ = self.rnn(x, h0)  # RNN output and last hidden state
        out = out[:, -1, :]  # Only take the output from the last time step
        out = self.fc(out)  # Final output

        return out

# Preprocess the data
scrambles, states = preprocess_data('dataset.csv')

# Convert the data to PyTorch tensors
scramble_tensor = [torch.tensor(s) for s in scrambles]  # Convert each scramble to a tensor
state_tensor = [torch.tensor(s) for s in states]  # Convert each state to a tensor

# Pad the scrambles and states
scramble_tensor = pad_sequence(scramble_tensor, batch_first=True)
state_tensor = pad_sequence(state_tensor, batch_first=True)

# Create a TensorDataset from the tensors
dataset = TensorDataset(state_tensor, scramble_tensor)

# Split the dataset into training and testing sets
train_len = int(len(dataset) * 0.8)
test_len = len(dataset) - train_len
train_data, test_data = random_split(dataset, [train_len, test_len])

# Create DataLoader objects for the training and testing sets
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Create the model
model = RubiksCubeRNN(input_size=54, hidden_size=100, output_size=20, num_layers=2)  # Adjust these parameters as needed

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Check if CUDA is available and set PyTorch to use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Train the model
for epoch in range(10):  # Number of epochs
    for i, (states, scrambles) in enumerate(train_loader):
        # Move the inputs and targets to the same device as the model
        states = states.to(device)
        scrambles = scrambles.to(device)

        # Forward pass
        outputs = model(states)
        loss = criterion(outputs, scrambles)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/10], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print('Finished Training')

# Test the model
# You need to write your own testing code here