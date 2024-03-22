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
import torch.nn.functional as F
from cube import Cube

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
    elif move == "y":
        cube.y_rotation()
    elif move == "y'":
        cube.y_rotation()
        cube.y_rotation()
        cube.y_rotation()
    elif move == "x":
        cube.x_rotation()
    elif move == "x'":
        cube.x_rotation()
        cube.x_rotation()
        cube.x_rotation()
    elif move == "z":
        cube.z_rotation()
    elif move == "z'":
        cube.z_rotation()
        cube.z_rotation()
        cube.z_rotation()
    elif move == "y2":
        cube.y_rotation()
        cube.y_rotation()
    elif move == "x2":
        cube.x_rotation()
        cube.x_rotation()
    elif move == "z2":
        cube.z_rotation()
        cube.z_rotation()
    return cube

class RubiksCubeEnvironment:
    def __init__(self, file_path):
        self.states = preprocess_data(file_path)
        self.state = Cube()  # Initialize with a solved Rubik's Cube
        self.state.cube = copy.deepcopy(self.states[0])
        self.ACTION_SPACE = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'", "R2", "L2", "U2", "D2", "F2", "B2", "y", "y'", "x", "x'", "z", "z'", "y2", "x2", "z2"]
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
        self.previous_moves.append(action)
        self.state = turnCube(action, self.state)

        # Calculate the reward based on the current state
        self.reward = self._calculate_reward()

        # Check if the goal state is reached
        done = self._is_solved()

        return self.state, self.reward, done

    def _calculate_reward(self):
        if self._is_solved():
            self.solved += 1
            return 10_000
        elif self.move_counter == self.max_moves:
            return 0
        elif self._is_redundant_move(self.previous_moves[-1]):
            return -1
        else:
            return 0

    def _is_solved(self):
        return (self.state.is_back_cross_solved(0) or self.state.is_front_cross_solved(0) or self.state.is_right_cross_solved(0) or self.state.is_left_cross_solved(0) or self.state.is_up_cross_solved(0) or self.state.is_down_cross_solved(0))

    def set_state(self, state_array):
        self.state.cube = state_array.reshape(6, 3, 3)

    #stoped here!
    def _is_redundant_move(self, action):
        if len(self.previous_moves) >= 3:
            last_move = self.previous_moves[-1]
            second_last_move = self.previous_moves[-2]
            third_last_move = self.previous_moves[-3]

            # Check if the last 3 moves are identical
            if action == last_move == second_last_move == third_last_move:
                return True

        return False

# Define the Q-network architecture
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3_adv = nn.Linear(64, action_size)  # Advantage stream
        self.fc3_val = nn.Linear(64, 1)           # Value stream

    def forward(self, x):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Now you can pass state_tensor to your network
        x = F.relu(self.fc1(state_tensor))
        x = F.relu(self.fc2(x))
        adv = F.relu(self.fc3_adv(x))
        val = self.fc3_val(x)
        return val + adv - adv.mean()


# Define the DQN agent
class DuelingDQNAgent:
    def __init__(self, input_size, output_size,seed = None, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, buffer_size=10000, batch_size=32, weight_decay=0.01):
        self.q_network = DuelingQNetwork(input_size, output_size, seed)
        self.target_network = DuelingQNetwork(input_size, output_size, seed)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.gamma = gamma
        self.epsilon = epsilon  # Initialize epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.input_size = input_size
        self.output_size = output_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = deque(maxlen=self.buffer_size)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_mini_batch()

        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

        q_values = self.q_network(states_tensor)
        next_q_values = self.target_network(next_states_tensor)

        max_next_q_values = next_q_values.detach().max(-1)[0].unsqueeze(-1)
        target_q_values = rewards_tensor + (self.gamma * max_next_q_values * (1 - dones_tensor))

        current_q_values = q_values.gather(0, actions_tensor.squeeze())

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)



    def select_action(self, state):
        if np.random.rand() < self.epsilon:  # Explore with probability epsilon
            return np.random.randint(self.output_size)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                return torch.argmax(q_values).item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_mini_batch(self):
        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        return states, actions, rewards, next_states, dones

# Create environment
env = RubiksCubeEnvironment(file_path="new_data/18to22/File(30).csv")

# Create DQN agent
input_size = 6*3*3  # Size of flattened Rubik's Cube state
output_size = len(env.ACTION_SPACE)  # Number of possible actions

model_path = 'modelsD/trained_model_episode_50.pth'

agent = DuelingDQNAgent(input_size, output_size, seed=42)
if os.path.isfile(model_path):
    # Load the model
    agent.q_network.load_state_dict(torch.load(model_path))
    print("Model loaded successfully!")
    time.sleep(5)
else:
    print(f"No model found at {model_path}. Please train a model first.")


TARGET_UPDATE_FREQUENCY = 100

# Training loop
env.max_moves = 12  # Set your maximum limit
logging.basicConfig(filename='training_logD.txt', level=logging.INFO, format='%(message)s')
start_from_n = env.stoped_at_n;

current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
log_message = f",{current_time},"
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

                # Store the experience in the replay buffer
                agent.store_experience(state, action, reward, next_state, done)

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
            torch.save(agent.q_network.state_dict(), f'modelsD/trained_model_episode_{episode}.pth')
            print(f"Model saved at episode {episode}.")
            time.sleep(1)  # Pause execution for 1 second

        # Log the saving time and total rewards
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"{episode},{current_time},{env.sugoi_rewards}"
        logging.info(log_message)

    print(f"Total Sugoi Rewards: {env.sugoi_rewards}")
    print(f"Solved: {env.solved}")

    env.states = preprocess_data(f"new_data/18to22/File({n+30}).csv")


# Evaluate the agent
# (You can implement a separate evaluation loop to measure the agent's performance)