import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader
import csv
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd


# R: 1, R':2, L:3, L':4, U:5, U':6, D:7, D':8, F:9, F':10, B:11, B':12, R2:13, L2:14, U2:15, D2:16, F2:17, B2:18, "":0

#D' F2 L U' R' D' F R2 D2 B' L' F' U' D2 R2 B2 L2 F U' D
# 8 17 14 6 2 8 17 1 13 16 4 10 6 16 13 18 14 6 8

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

# Define the RNN model class
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
    best_val_loss = float("inf")
    epochs_no_improve = 0
    n_epochs_stop = 5  # Number of epochs to wait before stopping
    for epoch in range(num_epochs):
        total_loss = 0
        total_val_loss = 0
        for i, (scrambles, states) in enumerate(data_loader):
            scrambles = scrambles.to(device)
            states = states.to(device)

            outputs = model(scrambles)
            loss = criterion(outputs, states)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for scrambles, states in valid_data_loader:
                scrambles = scrambles.to(device)
                states = states.to(device)
                outputs = model(scrambles)
                val_loss = criterion(outputs, states)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(valid_data_loader)
        model.avg_val_loss = avg_val_loss
        # Check if the validation loss has improved
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # If the validation loss hasn't improved for n_epochs_stop epochs, stop training
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            break
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    model.load_state_dict(torch.load('best_model.pt'))

# Load the data

# Train the model on the training data
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_val_loss = float("inf")
for i in range(3, 4):  # 3 to 22 inclusive
    # Preprocess the dataset
    scrambles, states = preprocess_data(f'data/dataset({i}).csv')

    if(i == 3):
        # Instantiate the model and optimizer
        input_size = len(scrambles[0]) # number of features
        hidden_size = 256
        output_size = len(states[0]) # number of output classes
        n_layers = 3
        model = Seq2Seq(input_size, hidden_size, output_size, n_layers)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Call the training loop function
        batch_size = 32
        num_epochs = 10
    else:
        model.load_state_dict(torch.load('best_model.pt'))
    # Split the data into training, validation, and testing sets
    train_scrambles, temp_scrambles, train_states, temp_states = train_test_split(scrambles, states, test_size=0.3, random_state=42)
    valid_scrambles, test_scrambles, valid_states, test_states = train_test_split(temp_scrambles, temp_states, test_size=1/3, random_state=42)

    # Create data loaders for the training, validation, and testing sets
    train_data_loader = create_data_loader(train_scrambles, train_states, batch_size)
    valid_data_loader = create_data_loader(valid_scrambles, valid_states, batch_size)
    test_data_loader = create_data_loader(test_scrambles, test_states, batch_size)

    # Train the model on the training data
    train(model, train_data_loader, criterion, optimizer, num_epochs)

    # Evaluate the model on the testing data
    test_loss = test(model, test_data_loader, criterion)
    print('Test Loss:', test_loss)

    # Calculate the accuracy
    accuracy = evaluate(model, test_data_loader)
    print('Accuracy:', accuracy)

    if model.avg_val_loss < best_val_loss:
        torch.save(model.state_dict(), 'best_model.pt')
        best_val_loss = model.avg_val_loss

# #265 114 326|654 426 341|155 236 314|432 545 123|135 451 636|235 661 422
# cube_state = [265, 114, 326, 654, 426, 341, 155, 236, 314, 432, 545, 123, 135, 451, 636, 235, 661, 422, 0, 0, 0, 0, 0 ,0 ,0]

def predict_state(model, scramble):
    model.eval()  # Set the model to evaluation mode

    # Convert the scramble to the format expected by your model
    # This will depend on how you've preprocessed your scrambles
    # Here I'm assuming you need to convert it to a PyTorch tensor
    scramble_tensor = torch.tensor(scramble).float().to(device)

    # Add an extra dimension to match the input size expected by your model
    scramble_tensor = scramble_tensor.unsqueeze(0)

    # Get the model's prediction
    prediction = model(scramble_tensor)

    # Convert the prediction to the format of your scramble states
    # This will depend on how your model's outputs relate to your states
    # Here I'm assuming each position in the output is a probability distribution over possible scramble states
    predicted_state = prediction.argmax(dim=2).tolist()

    return predicted_state

# Use the function
scramble = [8, 17, 14, 6, 2, 8, 17, 1, 13, 16, 4, 10, 6, 16, 13, 18, 14, 6, 8, 0, 0, 0, 0, 0, 0]  # Replace this with your actual scramble
predicted_state = predict_state(model, scramble)
print('Predicted state:', predicted_state)