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
    move_to_int = {"R": 1, "R'": 2, "L": 3, "L'": 4, "U": 5, "U'": 6, "D": 7, "D'": 8, "F": 9, "F'": 10, "B": 11, "B'": 12, "R2": 13, "L2": 14, "U2": 15, "D2": 16, "F2": 17, "B2": 18, "": 0}

    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        scramble = []
        stateStr = []
        for row in csvreader:
            row_str = row[0]
            # Convert the scramble to a list of integers
            scramble.append([move_to_int[move] for move in row_str.split(";")[0].split()])

            while(len(scramble[len(scramble)-1]) < 25):
                scramble[len(scramble)-1].append(0)

            stateStr.append(row_str.split(";")[1])

        state = []

        for row in stateStr:
            cube = []
            faces = row.split("|")
            for face in faces:
                rows = face.split(" ")
                for row in rows:
                    cube.append(int(row))

            state.append(cube)

    return scramble, state

# Load the data

# Preprocess the data

scrambles, states = preprocess_data('dataset.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the RNN model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Convert the 2D arrays into PyTorch tensors
def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

# Create a DataLoader from the tensors
def create_data_loader(scrambles, states, batch_size):
    scrambles_tensor = to_tensor(scrambles)
    states_tensor = to_tensor(states)
    dataset = TensorDataset(scrambles_tensor, states_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the training loop
def train(model, data_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for i, (scrambles, states) in enumerate(data_loader):
            scrambles = scrambles.to(device)
            states = states.to(device)

            outputs = model(scrambles)
            loss = criterion(outputs, states)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Load the data
scrambles, states = preprocess_data('dataset.csv')

# Instantiate the model and optimizer
input_size = len(scrambles[0]) # number of features
hidden_size = 128
output_size = len(states[0]) # number of output classes
n_layers = 2
model = LSTMModel(input_size, 2*hidden_size, output_size, 2*n_layers)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Call the training loop function
batch_size = 64
num_epochs = 10

data_loader = create_data_loader(scrambles, states, batch_size)
train(model, data_loader, criterion, optimizer, num_epochs)


#265 114 326|654 426 341|155 236 314|432 545 123|135 451 636|235 661 422
cube_state = [265, 114, 326, 654, 426, 341, 155, 236, 314, 432, 545, 123, 135, 451, 636, 235, 661, 422, 0, 0, 0, 0, 0 ,0 ,0]

model.eval()

# Convert the cube state to a tensor and add an extra dimension for batch size
cube_state = torch.tensor(cube_state, dtype=torch.float32).unsqueeze(0).to(device)

# Pass the cube state to the model
output = model(cube_state)

def output_to_scramble(output):
    # Define a mapping from output values to moves
    move_mapping = {1:"R", 2:"R'", 3:"L", 4:"L'", 5:"U", 6:"U'", 7:"D", 8:"D'", 9:"F", 10:"F'", 11:"B", 12:"B'", 13:"R2", 14:"L2", 15:"U2", 16:"D2", 17:"F2", 18:"B2", 0:""}

    # Convert the output tensor to a list of integers
    output = output.detach().numpy().tolist()

    print(output)

    for i in range(len(output[0])):
        output[0][i] = round(output[0][i] * 100)
        output[0][i] = int(output[0][i])
        if output[0][i] < 0:
            output[0][i] *= -1

    print(output)

    scramble = "";

    for i in range(len(output[0])):
        scramble += move_mapping[output[0][i]] + " "

    return scramble

# Postprocess the output to get the scramble
# This will depend on how your model's output corresponds to scrambles
scramble = output_to_scramble(output)

print('Scramble:', scramble)