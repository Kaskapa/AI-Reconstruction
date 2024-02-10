import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader
import csv
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd


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

# List to hold all datasets
all_datasets = []

# Loop over all dataset files
for i in range(3, 23):  # 3 to 22 inclusive

    # Preprocess the dataset
    scrambles, states = preprocess_data(f'data/dataset({i}).csv')

    # Add the preprocessed data to the list
    all_datasets.append((scrambles, states))

# Concatenate all datasets into a single one
full_scrambles = np.concatenate([data[0] for data in all_datasets])
full_states = np.concatenate([data[1] for data in all_datasets])

# Now you can split full_dataset into training and testing sets
train_scrambles, test_scrambles, train_states, test_states = train_test_split(full_scrambles, full_states, test_size=0.2, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the RNN model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = self.dropout(out[:, -1, :])
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

# Instantiate the model and optimizer
input_size = len(scrambles[0]) # number of features
hidden_size = 128
output_size = len(states[0]) # number of output classes
n_layers = 2
model = LSTMModel(input_size, hidden_size, output_size, n_layers, 0.5)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Call the training loop function
batch_size = 32
num_epochs = 10

train_data_loader = create_data_loader(train_scrambles, train_states, batch_size)
test_data_loader = create_data_loader(test_scrambles, test_states, batch_size)

# Train the model on the training data
train(model, train_data_loader, criterion, optimizer, num_epochs)

def test(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for scrambles, states in data_loader:
            scrambles = scrambles.to(device)
            states = states.to(device)
            outputs = model(scrambles)
            loss = criterion(outputs, states)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Call the testing loop function
test_loss = test(model, test_data_loader, criterion)
print('Test Loss:', test_loss)


def evaluate(model, data_loader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for scrambles, states in data_loader:
            scrambles = scrambles.to(device)
            states = states.to(device)
            outputs = model(scrambles)
            _, predicted = torch.max(outputs.data, 1)
            states = torch.argmax(states, dim=1)  # Convert states to class labels
            total_predictions += states.size(0)
            correct_predictions += (predicted == states).sum().item()
    return correct_predictions / total_predictions

# Call the evaluate function to calculate the accuracy
accuracy = evaluate(model, test_data_loader)
print('Accuracy:', accuracy)

# #265 114 326|654 426 341|155 236 314|432 545 123|135 451 636|235 661 422
# cube_state = [265, 114, 326, 654, 426, 341, 155, 236, 314, 432, 545, 123, 135, 451, 636, 235, 661, 422, 0, 0, 0, 0, 0 ,0 ,0]
