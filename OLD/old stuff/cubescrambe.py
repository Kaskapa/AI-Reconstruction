import csv
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Step 1: Data Loading and Preprocessing

def load_dataset(file_path):
    scrambles = []
    cube_states = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Split the row into scramble and cube state parts
            scramble, cube_state = row[0].split(';')
            
            # Append to lists
            scrambles.append(scramble.split())  # Split scramble into individual moves
            cube_states.append([list(map(int, face.split())) for face in cube_state.split('|')])  # Convert cube state strings to lists of integers
    
    return scrambles, cube_states

# Step 2: Define the Dataset Class

class RubiksCubeDataset(Dataset):
    def __init__(self, scrambles, cube_states):
        self.scrambles = scrambles
        self.cube_states = cube_states

    def __len__(self):
        return len(self.scrambles)

    def __getitem__(self, idx):
        return self.scrambles[idx], self.cube_states[idx]

# Step 3: Define Neural Network Model

class ScramblePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(ScramblePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        try:
            embedded = self.embedding(x.long())  # Convert input to long tensor
        except IndexError:
            print("Index out of range in input tensor. Adjusting indices...")
            x = torch.clamp(x, min=0, max=self.embedding.num_embeddings - 1)
            embedded = self.embedding(x.long())
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])  # Get the last output of the sequence
        return output

# Step 4: Training the Model

def collate_fn(batch):
    move_to_idx = {'U': 0, 'U2': 1, "U'": 2, 'D': 3, 'D2': 4, "D'": 5, 'F': 6, 'F2': 7, "F'": 8, 
                   'B': 9, 'B2': 10, "B'": 11, 'L': 12, 'L2': 13, "L'": 14, 'R': 15, 'R2': 16, "R'": 17}
    
    batch_moves = []
    batch_cube_states = []
    for scrambles, cube_states in batch:
        # Preprocess moves
        moves = [move_to_idx[move] for move in scrambles]
        batch_moves.append(torch.tensor(moves))
        batch_cube_states.append(torch.tensor(cube_states, dtype=torch.float32))
    
    # Pad sequences
    padded_moves = pad_sequence(batch_moves, batch_first=True)
    padded_cube_states = pad_sequence(batch_cube_states, batch_first=True)
    
    return padded_moves, padded_cube_states

# Modify the train_model function to handle consistent batch sizes
def train_model(model, dataloader, num_epochs=10, learning_rate=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0
        for scrambles, cube_states in dataloader:
            optimizer.zero_grad()

            outputs = model(cube_states)
            outputs = outputs.view(-1, outputs.size(-1))

            # Flatten cube_states to match the shape of outputs
            flat_cube_states = cube_states.view(-1, cube_states.size(-1)).long()

            loss = criterion(outputs, flat_cube_states)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(scrambles)  # Multiply by batch size
            total_samples += len(scrambles)
            print(f'Processed {total_samples} samples in this epoch.')  # Debugging print

        average_loss = total_loss / total_samples  # Calculate average loss
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')



scrambles, cube_states = load_dataset('dataset.csv')

# Define the input, hidden, and output sizes
vocab_size = 100  # Adjust based on your vocabulary size
embedding_dim = 100  # Specify the dimensionality of the embeddings
hidden_size = 128  # Specify the size of the hidden layer
output_size = 54 * 25  # Output size is flattened scramble

# Create dataset and dataloader
dataset = RubiksCubeDataset(scrambles, cube_states)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Initialize the model
model = ScramblePredictor(vocab_size, embedding_dim, hidden_size, output_size)

# Train the model
train_model(model, dataloader, num_epochs=10)

# Save the trained model if needed
torch.save(model.state_dict(), 'scramble_predictor.pth')