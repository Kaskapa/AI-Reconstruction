import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple gridworld environment
class GridWorld:
    def __init__(self, size):
        self.size = size
        self.state = [0, 0]  # Agent's initial position
        self.goal = [size-1, size-1]  # Goal position
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.state = [0, 0]  # Reset agent's position
        return self.state

    def step(self, action):
        if action == 'up':
            self.state[0] = max(0, self.state[0] - 1)
        elif action == 'down':
            self.state[0] = min(self.size - 1, self.state[0] + 1)
        elif action == 'left':
            self.state[1] = max(0, self.state[1] - 1)
        elif action == 'right':
            self.state[1] = min(self.size - 1, self.state[1] + 1)

        done = self.state == self.goal  # Check if goal is reached
        reward = 1 if done else 0  # Give reward of 1 if goal is reached, else 0
        return self.state, reward, done

# Define a Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(state_size, action_size)

    def forward(self, x):
        return self.fc(x)

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.99, epsilon=0.1):
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.SGD(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Exploration: randomly choose action
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()  # Exploitation: choose action with highest Q-value

    def update_q_function(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)

        # Compute TD target
        with torch.no_grad():
            next_q_values = self.q_network(next_state)
            td_target = reward + self.gamma * torch.max(next_q_values).item() * (1 - done)

        # Compute TD error
        q_values = self.q_network(state)
        td_error = td_target - q_values[action]

        # Update Q-values
        self.optimizer.zero_grad()
        loss = td_error ** 2
        loss.backward()
        self.optimizer.step()

# Define training function
def train(agent, env, episodes=10):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(env.actions[action])
            agent.update_q_function(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

# Instantiate environment and agent
env = GridWorld(size=5)
print("Environment size:", env.size)
state_size = 2  # Size of state space
action_size = len(env.actions)  # Number of actions
agent = QLearningAgent(state_size, action_size)
print("Agent's action space:", env.actions)

# Train agent
train(agent, env)

# Test trained agent
state = env.reset()
done = False
while not done:
    action = agent.select_action(state)
    next_state, reward, done = env.step(env.actions[action])
    state = next_state
    print("Agent's position:", state)
