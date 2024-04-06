import gym
from gym import spaces
import numpy as np
from rubiks_cube import Cube
import csv
import copy

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

states = preprocess_data("new_data/18to22/file(30).csv");

class RubiksCubeEnv(gym.Env):
    def __init__(self):
        super(RubiksCubeEnv, self).__init__()
        self.cube = Cube()
        self.MOVES = ["U","U2", "U'", "D", "D2", "D'","R","R2","R'","L","L2","L'","F","F2","F'","B","B2","B'"]
        self.action_space = spaces.Discrete(len(self.MOVES))  # 12 possible actions (6 faces * 2 directions)
        self.observation_space = spaces.Box(low=0, high=5, shape=(6, 3, 3), dtype=np.int32)  # 6 faces, 3x3 grid, 6 possible colors
        self.prevmoves = []
        self.ALTERNATINGMOVES = [["R","L"], ["U","D"], ["F","B"]]
        self.which_state = 0;

    def reset(self):
        self.cube.cube = copy.deepcopy(states[self.which_state])
        return self.cube.get_flattened_state()

    def step(self, action):
        # Perform the action on the Rubik's Cube
        self._perform_action(action)

        self.prevmoves.append(self.MOVES[action])

        # Calculate reward and check if the episode is done
        reward = self._calculate_reward()
        done = self._is_solved()

        # Return observation, reward, done, info
        return self.cube.get_flattened_state(), reward, done, {}

    def _perform_action(self, action):
        # Mapping action indices to Rubik's Cube moves
        if action == 0:
            self.cube.up_turn()
        elif action == 1:
            self.cube.up_turn()
            self.cube.up_turn()
        elif action == 2:
            self.cube.up_turn()
            self.cube.up_turn()
            self.cube.up_turn()
        elif action == 3:
            self.cube.down_turn()
        elif action == 4:
            self.cube.down_turn()
            self.cube.down_turn()
        elif action == 5:
            self.cube.down_turn()
            self.cube.down_turn()
            self.cube.down_turn()
        elif action == 6:
            self.cube.right_turn()
        elif action == 7:
            self.cube.right_turn()
            self.cube.right_turn()
        elif action == 8:
            self.cube.right_turn()
            self.cube.right_turn()
            self.cube.right_turn()
        elif action == 9:
            self.cube.left_turn()
        elif action == 10:
            self.cube.left_turn()
            self.cube.left_turn()
        elif action == 11:
            self.cube.left_turn()
            self.cube.left_turn()
            self.cube.left_turn()
        elif action == 12:
            self.cube.front_turn()
        elif action == 13:
            self.cube.front_turn()
            self.cube.front_turn()
        elif action == 14:
            self.cube.front_turn()
            self.cube.front_turn()
            self.cube.front_turn()
        elif action == 15:
            self.cube.back_turn()
        elif action == 16:
            self.cube.back_turn()
            self.cube.back_turn()
        elif action == 17:
            self.cube.back_turn()
            self.cube.back_turn()
            self.cube.back_turn()

    def _calculate_reward(self):
        total_reward = 0
        if(len(self.prevmoves) > 2 and self.prevmoves[-1][0] == self.prevmoves[-2][0] == self.prevmoves[-3][0]):
            total_reward -= 10

        elif(len(self.prevmoves) > 1 and self.prevmoves[-1][0] == self.prevmoves[-2][0]):
            total_reward -=10

        for moves in self.ALTERNATINGMOVES:
            if(len(self.prevmoves) > 2
               and ((self.prevmoves[-1][0] == moves[0]
                     and self.prevmoves[-2][0] == moves[1]
                     and self.prevmoves[-3][0] == moves[0])
                    or (self.prevmoves[-1][0] == moves[1]
                        and self.prevmoves[-2][0] == moves[0]
                        and self.prevmoves[-3][0] == moves[1]))):
                total_reward -= 10

        if self._is_solved():
            total_reward += 1000

        return total_reward

    def _is_solved(self):
        return self.cube.is_white_cross_solved() or self.cube.is_orange_cross_solved() or self.cube.is_green_cross_solved() or self.cube.is_red_cross_solved() or self.cube.is_blue_cross_solved() or self.cube.is_yellow_cross_solved()

    def render(self, mode='human'):
        # Implement visualization of the Rubik's Cube if needed
        pass

    def close(self):
        pass


class QLearningAgent:
    def __init__(self, action_space, MOVES, learning_rate=0.1, discount_factor=0.99, epsilon=0.9):
        self.action_space = action_space
        self.MOVES = MOVES
        self.last_action = None
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((NUM_STATES, NUM_ACTIONS))  # Initialize Q-table

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            next_action = self.action_space.sample()  # Exploration: choose a random action

            while self.last_action != None and self.MOVES[next_action][0] == self.MOVES[self.last_action][0]:
                next_action = self.action_space.sample()

            self.last_action = next_action

            return next_action  # Exploration: choose a random action
        else:
            next_action = np.argmax(self.q_table[state])  # Exploration: choose a random action

            while self.last_action != None and self.MOVES[next_action][0] == self.MOVES[self.last_action][0]:
                next_action = self.action_space.sample()

            self.last_action = next_action

            return next_action# Exploitation: choose the action with the highest Q-value

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

# Parameters
NUM_STATES = 54  # Number of states in the Rubik's Cube (flattened representation)
EPISODES = 1000  # Number of training episodes
MAX_MOVES = 20  # Maximum number of moves in a single episode

# Initialize Gym environment and Q-learning agent
env = RubiksCubeEnv()
NUM_ACTIONS = len(env.MOVES)  # Number of possible actions (rotations)
agent = QLearningAgent(env.action_space, env.MOVES)

howManyDone = 0

episode = 0

# Training loop
while episode < EPISODES:
    state = env.reset()
    done = False
    total_reward = 0

    move_count = 0

    while not done and move_count < MAX_MOVES:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        total_reward += reward
        state = next_state
        print(env.MOVES[action])
        move_count += 1

    if done:
        howManyDone += 1
        env.which_state += 1
        episode += 1

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Moves: {move_count}, Done: {done}")

print(howManyDone)
