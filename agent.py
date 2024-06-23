import torch
import random
import numpy as np
from collections import deque
from rubiks_game_ai import RubiksGameAI, Move
import csv
from model import Linear_QNet, QTrainer
from helper import plot
from givemecrossUsable import CrossSolver


def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        stateStr = []
        for row in csvreader:
            row_str = row[0]
            # Convert the scramble to a list of integers
            stateStr.append(row_str.split(";")[0])

    return stateStr

import os

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
file_path = os.path.join(script_dir, "File(30).csv")

states = preprocess_data(file_path)

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.00001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(102, 256, 25)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        insertedPairs = game.cube.get_inserted_pairs()
        state = [
            #pair 1 paired
            game.cube.is_pair_paired(0),

            #pair 2 paired
            game.cube.is_pair_paired(1),

            #pair 3 paired
            game.cube.is_pair_paired(2),

            #pair 4 paired
            game.cube.is_pair_paired(3),

            #pair 5 paired
            game.cube.is_pair_paired(4),

            #pair 6 paired
            game.cube.is_pair_paired(5),

            #pair 7 paired
            game.cube.is_pair_paired(6),

            #pair 8 paired
            game.cube.is_pair_paired(7),

            #pair 9 paired
            game.cube.is_pair_paired(8),

            #pair 10 paired
            game.cube.is_pair_paired(9),

            #pair 11 paired
            game.cube.is_pair_paired(10),

            #pair 12 paired
            game.cube.is_pair_paired(11),

            #pair 13 paired
            game.cube.is_pair_paired(12),

            #pair 14 paired
            game.cube.is_pair_paired(13),

            #pair 15 paired
            game.cube.is_pair_paired(14),

            #pair 16 paired
            game.cube.is_pair_paired(15),

            #pair 17 paired
            game.cube.is_pair_paired(16),

            #pair 18 paired
            game.cube.is_pair_paired(17),

            #pair 19 paired
            game.cube.is_pair_paired(18),

            #pair 20 paired
            game.cube.is_pair_paired(19),

            #pair 21 paired
            game.cube.is_pair_paired(20),

            #pair 22 paired
            game.cube.is_pair_paired(21),

            #pair 23 paired
            game.cube.is_pair_paired(22),

            #pair 24 paired
            game.cube.is_pair_paired(23),

            #pair 1 is inserted
            0 in insertedPairs,

            #pair 2 is inserted
            1 in insertedPairs,

            #pair 3 is inserted
            2 in insertedPairs,

            #pair 4 is inserted
            3 in insertedPairs,

            #pair 5 is inserted
            4 in insertedPairs,

            #pair 6 is inserted
            5 in insertedPairs,

            #pair 7 is inserted
            6 in insertedPairs,

            #pair 8 is inserted
            7 in insertedPairs,

            #pair 9 is inserted
            8 in insertedPairs,

            #pair 10 is inserted
            9 in insertedPairs,

            #pair 11 is inserted
            10 in insertedPairs,

            #pair 12 is inserted
            11 in insertedPairs,

            #pair 13 is inserted
            12 in insertedPairs,

            #pair 14 is inserted
            13 in insertedPairs,

            #pair 15 is inserted
            14 in insertedPairs,

            #pair 16 is inserted
            15 in insertedPairs,

            #pair 17 is inserted
            16 in insertedPairs,

            #pair 18 is inserted
            17 in insertedPairs,

            #pair 19 is inserted
            18 in insertedPairs,

            #pair 20 is inserted
            19 in insertedPairs,

            #pair 21 is inserted
            20 in insertedPairs,

            #pair 22 is inserted
            21 in insertedPairs,

            #pair 23 is inserted
            22 in insertedPairs,

            #pair 24 is inserted
            23 in insertedPairs
        ]

        flattenedCube = []

        for face in game.cube.cube:
            for row in face:
                for val in row:
                    flattenedCube.append(val)

        flattenedCube = []

        for face in game.cube.cube:
            for row in face:
                for val in row:
                    flattenedCube.append(val)

        state += flattenedCube

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 300 - self.n_games
        final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if random.randint(0, 600) < self.epsilon:
            move = random.randint(0, 24)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_inserted_pairs = []
    plot_mean_inserted_pairs = []
    total_score = 0
    record = 0
    agent = Agent()
    game = RubiksGameAI()

    state = states[0]
    stateArr = state.split(" ")
    for i in stateArr:
        game.cube.do_moves(i)

    solver = CrossSolver(states[0])
    solver.solve()

    crossSolution = solver.allSol[0]
    crossSolutionArr = crossSolution.split(" ")
    for i in crossSolutionArr:
        game.cube.do_moves(i)


    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, pairsPaired, pairsInserted = game.play_step(final_move)

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            state = states[agent.n_games]
            stateArr = state.split(" ")
            for i in stateArr:
                game.cube.do_moves(i)

            solver = CrossSolver(states[agent.n_games])
            solver.solve()

            crossSolution = solver.allSol[0]
            crossSolutionArr = crossSolution.split(" ")
            for i in crossSolutionArr:
                game.cube.do_moves(i)

            agent.n_games += 1

            agent.train_long_memory()

            if reward > record:
                record = reward
                agent.model.save()

            print(f'Game {agent.n_games}, Pairs paired: {pairsPaired}, Pairs inserted: {pairsInserted} Record: {record}')

            plot_inserted_pairs.append(reward)
            total_score += reward
            mean_score = total_score / agent.n_games
            plot_mean_inserted_pairs.append(mean_score)
            plot(plot_inserted_pairs, plot_mean_inserted_pairs)

if __name__ == '__main__':
    train()