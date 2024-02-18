import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader
import csv
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from collections import deque
import os
import time
import logging
import copy

# Preprocess the data
def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        stateStr = []
        for row in csvreader:
            row_str = row[0]
            # Convert the scramble to a list of integers
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
                    desired_array = [int(numeric_string)-1 for numeric_string in elements]
                    facesInt.append(desired_array)
                cube.append(facesInt)
            state.append(cube)

    return state

# 0 - White, 1 - Orange, 2 - Green, 3 - Red, 4 - Blue, 5 - Yellow

class Cube:
    def __init__(self):
        self.cube = [
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],

            [[2, 2, 2],
             [2, 2, 2],
             [2, 2, 2]],

            [[3, 3, 3],
             [3, 3, 3],
             [3, 3, 3]],

            [[4, 4, 4],
             [4, 4, 4],
             [4, 4, 4]],

            [[5, 5, 5],
             [5, 5, 5],
             [5, 5, 5]]
        ]

    def reset(self):
        self.cube = [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
            [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        ]

    def is_white_cross_solved(self):
        if(self.cube[0][0][1] == 0 and self.cube[0][1][0] == 0 and self.cube[0][1][2] == 0 and self.cube[0][2][1] == 0 and self.cube[0][1][1] == 0 and self.cube[1][0][1] == 1 and self.cube[2][0][1] == 2 and self.cube[3][0][1] == 3 and self.cube[4][0][1] == 4):
            return True
        else:
            return False

    def is_orange_cross_solved(self):
        if(self.cube[1][0][1] == 1 and self.cube[1][1][0] == 1 and self.cube[1][1][2] == 1 and self.cube[1][2][1] == 1 and self.cube[1][1][1] == 1 and self.cube[0][0][1] == 0 and self.cube[2][0][1] == 2 and self.cube[3][0][1] == 3 and self.cube[5][0][1] == 5):
            return True
        else:
            return False

    def is_green_cross_solved(self):
        if(self.cube[2][0][1] == 2 and self.cube[2][1][0] == 2 and self.cube[2][1][2] == 2 and self.cube[2][2][1] == 2 and self.cube[2][1][1] == 2 and self.cube[0][0][1] == 0 and self.cube[1][0][1] == 1 and self.cube[3][0][1] == 3 and self.cube[5][0][1] == 5):
            return True
        else:
            return False

    def is_red_cross_solved(self):
        if(self.cube[3][0][1] == 3 and self.cube[3][1][0] == 3 and self.cube[3][1][2] == 3 and self.cube[3][2][1] == 3 and self.cube[3][1][1] == 3 and self.cube[0][0][1] == 0 and self.cube[1][0][1] == 1 and self.cube[2][0][1] == 2 and self.cube[5][0][1] == 5):
            return True
        else:
            return False

    def is_blue_cross_solved(self):
        if(self.cube[4][0][1] == 4 and self.cube[4][1][0] == 4 and self.cube[4][1][2] == 4 and self.cube[4][2][1] == 4 and self.cube[4][1][1] == 4 and self.cube[0][0][1] == 0 and self.cube[1][0][1] == 1 and self.cube[2][0][1] == 2 and self.cube[3][0][1] == 3):
            return True
        else:
            return False

    def is_yellow_cross_solved(self):
        if(self.cube[5][0][1] == 5 and self.cube[5][1][0] == 5 and self.cube[5][1][2] == 5 and self.cube[5][2][1] == 5 and self.cube[5][1][1] == 5 and self.cube[1][2][1] == 1 and self.cube[2][2][1] == 2 and self.cube[3][2][1] == 3 and self.cube[4][2][1] == 4):
            return True
        else:
            return False

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

    def get_flattened_state(self):
        return np.array(self.cube).flatten()

def turnCube(move, cube):
    if move == "R":
        cube.right_turn()
    elif move == "R'":
        cube.right_turn()
        cube.right_turn()
        cube.right_turn()
    elif move == "L":
        cube.left_turn()
    elif move == "L'":
        cube.left_turn()
        cube.left_turn()
        cube.left_turn()
    elif move == "U":
        cube.up_turn()
    elif move == "U'":
        cube.up_turn()
        cube.up_turn()
        cube.up_turn()
    elif move == "D":
        cube.down_turn()
    elif move == "D'":
        cube.down_turn()
        cube.down_turn()
        cube.down_turn()
    elif move == "F":
        cube.front_turn()
    elif move == "F'":
        cube.front_turn()
        cube.front_turn()
        cube.front_turn()
    elif move == "B":
        cube.back_turn()
    elif move == "B'":
        cube.back_turn()
        cube.back_turn()
        cube.back_turn()
    elif move == "R2":
        cube.right_turn()
        cube.right_turn()
    elif move == "L2":
        cube.left_turn()
        cube.left_turn()
    elif move == "U2":
        cube.up_turn()
        cube.up_turn()
    elif move == "D2":
        cube.down_turn()
        cube.down_turn()
    elif move == "F2":
        cube.front_turn()
        cube.front_turn()
    elif move == "B2":
        cube.back_turn()
        cube.back_turn()
    return cube

class RubiksCubeEnvironment:
    def __init__(self, file_path):
        self.states = preprocess_data(file_path)
        self.state = Cube()  # Initialize with a solved Rubik's Cube
        self.state.cube = copy.deepcopy(self.states[0])
        self.ACTION_SPACE = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'", "R2", "L2", "U2", "D2", "F2", "B2"]
        self.reward = 0
        self.move_counter = 0
        self.max_moves = 10
        self.solved = 0
        self.previous_moves = []
        self.stoped_at_episode = 0
        self.stoped_at_n = 1

        self.sugoi_rewards = 0

    def step(self, action):
        # Perform the action on the cube (e.g., rotate a face)
        self.state = turnCube(action, self.state)

        # Calculate the reward based on the current state
        self.reward = self._calculate_reward()

        if self._is_redundant_move(action):
            # Penalize the agent and return the current state and a negative reward
            self.reward = -10
            return self.state, self.reward, False

        # Check if the goal state is reached
        done = self._is_solved()

        return self.state, self.reward, done

    def _calculate_reward(self):
        if self._is_solved():
            self.solved += 1;
            return 10_000
        elif self.move_counter == self.max_moves:
            return -50
        else:
            return -1

    def _is_solved(self):
        return self.state.is_white_cross_solved()

    def reset(self):
        self.state.reset()

    def set_state(self, state_array):
        self.state.cube = state_array.reshape(6, 3, 3)

    #stoped here!
    def _is_redundant_move(self, action):
        # Check if the new move cancels out the effect of the last move(s) in the history
        if len(self.previous_moves) >= 2:
            last_move = self.previous_moves[-1]
            second_last_move = self.previous_moves[-2]

            # Check if the new move and the last move cancel each other out
            if action + "'" == last_move and action[:-1] == second_last_move:
                return True
            elif action[:-1] == last_move and action + "'" == second_last_move:
                return True

        return False

# Define the Q-network architecture
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the DQN agent
class DQNAgent:
    def __init__(self, input_size, output_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, buffer_size=10000, batch_size=32):
        self.q_network = QNetwork(input_size, output_size)
        self.target_network = QNetwork(input_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Initialize target network with same weights
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.input_size = input_size
        self.output_size = output_size

        self.buffer_size = buffer_size  # Replay buffer size
        self.batch_size = batch_size    # Mini-batch size
        self.replay_buffer = deque(maxlen=self.buffer_size)  # Replay buffer

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_mini_batch(self):
        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        return states, actions, rewards, next_states, dones

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.output_size)  # Explore
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                return torch.argmax(q_values).item()  # Exploit

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples in the replay buffer

        states, actions, rewards, next_states, dones = self.sample_mini_batch()

        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        # Calculate target Q-values
        with torch.no_grad():
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * torch.max(self.target_network(next_states_tensor), dim=1).values

        # Calculate current Q-values
        q_values = self.q_network(states_tensor)
        current_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze()

        # Update Q-values using Bellman equation
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# Create environment
env = RubiksCubeEnvironment(file_path="new_data/18to22/File(30).csv")

# Create DQN agent
input_size = 6*3*3  # Size of flattened Rubik's Cube state
output_size = len(env.ACTION_SPACE)  # Number of possible actions

model_path = 'models/trained_model_episode_50.pth'

agent = DQNAgent(input_size, output_size)
if os.path.isfile(model_path):
    # Load the model
    agent.q_network.load_state_dict(torch.load(model_path))
    print("Model loaded successfully!")
    time.sleep(5)
else:
    print(f"No model found at {model_path}. Please train a model first.")


TARGET_UPDATE_FREQUENCY = 100

# Training loop
env.max_moves = 20  # Set your maximum limit
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(message)s')
start_from_n = env.stoped_at_n;

current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
log_message = f", {current_time},"
logging.info(log_message)

for n in range(start_from_n, 10):
    start_from = env.stoped_at_episode
    for episode in range(start_from, 10_000):
        done = False
        env.state.cube = copy.deepcopy(env.states[episode])  # Set the state for the next episode
        while not done:
            total_reward = 0
            env.move_counter = 0  # Initialize move counter at the start of each episode

            while not done and env.move_counter < env.max_moves:
                state = env.state.get_flattened_state()

                action = agent.select_action(state)
                next_state, reward, done = env.step(env.ACTION_SPACE[action])

                next_state = next_state.get_flattened_state()
                agent.train()
                total_reward += reward
                env.set_state(next_state)
                env.move_counter += 1  # Increment move counter after each action

                print(f"Action: {env.ACTION_SPACE[action]}")

            if(not done):
                env.state.cube = copy.deepcopy(env.states[episode])  # Set the state for the next episode

            if episode % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()

            print(f"Episode: {episode}, Total Reward: {total_reward}")
            env.sugoi_rewards += total_reward;
        if episode % 50 == 0:
            torch.save(agent.q_network.state_dict(), f'models/trained_model_episode_{episode}.pth')
            print(f"Model saved at episode {episode}.")
            time.sleep(1)  # Pause execution for 1 second

        # Log the saving time and total rewards
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"{episode}, {current_time}, {env.sugoi_rewards}"
        logging.info(log_message)

    print(f"Total Sugoi Rewards: {env.sugoi_rewards}")
    print(f"Solved: {env.solved}")

    env.states = preprocess_data(f"new_data/18to22/File({n+30}).csv")


# Evaluate the agent
# (You can implement a separate evaluation loop to measure the agent's performance)