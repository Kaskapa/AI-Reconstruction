import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader
import csv
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd

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

def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

# Create a DataLoader from the tensors
def create_data_loader(scrambles, states, batch_size):
    scrambles_tensor = to_tensor(scrambles)
    states_tensor = to_tensor(states)
    dataset = TensorDataset(scrambles_tensor, states_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def convert_to_string(arr):
    # Convert all elements to string
    arr = list(map(str, arr))

    # Join every three elements with '|'
    return '|'.join([' '.join(arr[i:i+3]) for i in range(0, len(arr), 3)])

def convert_to_3d(state):
    threedArr = []

    faceArr = state.split('|')
    for face in faceArr:
        rowArr = face.split(' ')
        threeDRowArr = []
        for row in rowArr:
            cellarr = [int(ch)-1 for ch in row]
            threeDRowArr.append(cellarr)
        threedArr.append(threeDRowArr)

    return threedArr

class Cube:
    def __init__(self):
        self.cube = [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
            [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        ]

    def right_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = self.cube[5][i][2]  # Yellow
            temp[1][i] = self.cube[2][i][2]  # Green
            temp[2][i] = self.cube[0][2 - i][2]  # White
            temp[3][i] = self.cube[4][2 - i][0]  # Blue
        for i in range(3):
            self.cube[2][i][2] = temp[0][i]
            self.cube[0][i][2] = temp[1][i]
            self.cube[4][i][0] = temp[2][i]
            self.cube[5][i][2] = temp[3][i]

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = self.cube[3][i][j]
        for i in range(3):
            for j in range(3):
                self.cube[3][i][j] = temp[2 - j][i]

    def left_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = self.cube[5][2-i][0]
            temp[1][i] = self.cube[2][i][0]
            temp[2][i] = self.cube[0][i][0]
            temp[3][i] = self.cube[4][2- i][2]
        for i in range(3):
            self.cube[2][i][0] = temp[2][i]
            self.cube[0][i][0] = temp[3][i]
            self.cube[4][i][2] = temp[0][i]
            self.cube[5][i][0] = temp[1][i]

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = self.cube[1][i][j]
        for i in range(3):
            for j in range(3):
                self.cube[1][i][j] = temp[2-j][i]
    def up_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = self.cube[1][0][i]
            temp[1][i] = self.cube[2][0][i]
            temp[2][i] = self.cube[3][0][i]
            temp[3][i] = self.cube[4][0][i]
        for i in range(3):
            self.cube[1][0][i] = temp[1][i]
            self.cube[2][0][i] = temp[2][i]
            self.cube[3][0][i] = temp[3][i]
            self.cube[4][0][i] = temp[0][i]

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = self.cube[0][i][j]
        for i in range(3):
            for j in range(3):
                self.cube[0][i][j] = temp[2-j][i]

    def down_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = self.cube[1][2][i]
            temp[1][i] = self.cube[2][2][i]
            temp[2][i] = self.cube[3][2][i]
            temp[3][i] = self.cube[4][2][i]
        for i in range(3):
            self.cube[1][2][i] = temp[3][i]
            self.cube[2][2][i] = temp[0][i]
            self.cube[3][2][i] = temp[1][i]
            self.cube[4][2][i] = temp[2][i]

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = self.cube[5][i][j]
        for i in range(3):
            for j in range(3):
                self.cube[5][i][j] = temp[2-j][i]

    def front_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = self.cube[1][i][2]
            temp[1][i] = self.cube[0][2][i]
            temp[2][i] = self.cube[3][i][0]
            temp[3][i] = self.cube[5][0][i]
        for i in range(3):
            self.cube[1][i][2] = temp[3][i]
            self.cube[0][2][i] = temp[0][2-i]
            self.cube[3][i][0] = temp[1][i]
            self.cube[5][0][i] = temp[2][2-i]

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = self.cube[2][i][j]
        for i in range(3):
            for j in range(3):
                self.cube[2][i][j] = temp[2-j][i]

    def back_turn(self):
        temp = [[0]*3 for _ in range(4)]
        for i in range(3):
            temp[0][i] = self.cube[1][i][0]
            temp[1][i] = self.cube[0][0][i]
            temp[2][i] = self.cube[3][i][2]
            temp[3][i] = self.cube[5][2][i]
        for i in range(3):
            self.cube[1][i][0] = temp[1][2-i]
            self.cube[0][0][i] = temp[2][i]
            self.cube[3][i][2] = temp[3][2-i]
            self.cube[5][2][i] = temp[0][i]

        temp = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                temp[i][j] = self.cube[4][i][j]
        for i in range(3):
            for j in range(3):
                self.cube[4][i][j] = temp[2-j][i]

# R: 1, R':2, L:3, L':4, U:5, U':6, D:7, D':8, F:9, F':10, B:11, B':12, R2:13, L2:14, U2:15, D2:16, F2:17, B2:18, "":0
def turnCube(scramble, cube):
    for move in scramble:
        if move == 1:
            cube.right_turn()
        elif move == 2:
            cube.right_turn()
            cube.right_turn()
            cube.right_turn()
        elif move == 3:
            cube.left_turn()
        elif move == 4:
            cube.left_turn()
            cube.left_turn()
            cube.left_turn()
        elif move == 5:
            cube.up_turn()
        elif move == 6:
            cube.up_turn()
            cube.up_turn()
            cube.up_turn()
        elif move == 7:
            cube.down_turn()
        elif move == 8:
            cube.down_turn()
            cube.down_turn()
            cube.down_turn()
        elif move == 9:
            cube.front_turn()
        elif move == 10:
            cube.front_turn()
            cube.front_turn()
            cube.front_turn()
        elif move == 11:
            cube.back_turn()
        elif move == 12:
            cube.back_turn()
            cube.back_turn()
            cube.back_turn()
        elif move == 13:
            cube.right_turn()
            cube.right_turn()
        elif move == 14:
            cube.left_turn()
            cube.left_turn()
        elif move == 15:
            cube.up_turn()
            cube.up_turn()
        elif move == 16:
            cube.down_turn()
            cube.down_turn()
        elif move == 17:
            cube.front_turn()
            cube.front_turn()
        elif move == 18:
            cube.back_turn()
            cube.back_turn()
    return cube.cube

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.LSTM(input_size=25, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=25, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass the input through the encoder
        _, (hidden, cell) = self.encoder(x)

        # Prepare the decoder input as a sequence of zeros
        decoder_input = torch.zeros_like(x)

        # Pass the decoder input and the encoder hidden state through the decoder
        output, _ = self.decoder(decoder_input, (hidden, cell))

        # Pass the decoder outputs through the fully connected layer
        output = self.fc(output)

        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def trainLoop():
    for i in range(3, 23):
        if(i == 3):
            # Define the hyperparameters
            input_size = 25
            hidden_size = 64
            output_size = 18
            n_layers = 1
            learning_rate = 0.001
            num_epochs = 10
            batch_size = 64

            # Create the model
            model = Seq2Seq(input_size, hidden_size, output_size, n_layers)

            # Define the loss and the optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Load the data
        scrambles, states = preprocess_data(f'data/dataset({i}).csv')

        # Split the data into training and validation sets
        train_scrambles, temp_scrambles, train_states, temp_states = train_test_split(scrambles, states, test_size=0.3, random_state=42)
        valid_scrambles, test_scrambles, valid_states, test_states = train_test_split(temp_scrambles, temp_states, test_size=1/3, random_state=42)

        # Create data loaders for the training, validation, and testing sets
        train_data_loader = create_data_loader(train_scrambles, train_states, batch_size)
        valid_data_loader = create_data_loader(valid_scrambles, valid_states, batch_size)
        test_data_loader = create_data_loader(test_scrambles, test_states, batch_size)


        train(num_epochs, model, train_data_loader, criterion, optimizer);
        test(model, test_data_loader, criterion)

def train(num_epochs, model, data_loader, criterion, optimizer):
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

def test(model, data_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_values = 0

    with torch.no_grad():  # Disable gradient calculation
        for i, (scrambles, states) in enumerate(data_loader):
            scrambles = scrambles.to(device)
            states = states.to(device)

            outputs = model(scrambles)
            loss = criterion(outputs, states)
            total_loss += loss.item()

            # Convert outputs probabilities to binary predictions
            predictions = (torch.sigmoid(outputs) > 0.5).float()

            # Compare predictions to actual values
            total_correct += (predictions == states).sum().item()
            total_values += torch.numel(states)

    avg_loss = total_loss / len(data_loader)  # Calculate average loss
    accuracy = total_correct / total_values  # Calculate accuracy
    print(f'Test Loss: {avg_loss}, Accuracy: {accuracy}')

trainLoop();