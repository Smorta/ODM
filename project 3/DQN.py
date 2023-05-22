# Custom imports
import utils

import numpy as np
from tqdm import tqdm
import gymnasium as gym
import os

from torch.utils.tensorboard import SummaryWriter
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, beta, input_size, hidden_size, output_size, name, chkpt_dir='tmp/dqn'):
        super(QNetwork, self).__init__()
        self.beta = beta
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_dqn')

        self.fc1 = nn.Linear(self.input_size, self.hidden_size * 2)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class DQN:
    def __init__(self, alpha, tau=1, env_name='Pendulum-v1', gamma=0.99,
                 max_size=100000, batch_size=64, hidden_size=256, Nbr_action=5):
        self.alpha = alpha
        self.tau = tau
        self.epsilon = 1
        self.epsilon_end = 200
        self.epsilon_decay = 0.999
        self.gamma = gamma
        self.max_size = max_size
        self.batch_size = batch_size
        self.env = gym.make(env_name)
        self.Nbr_features = self.env.observation_space.shape[0]
        self.Nbr_action = Nbr_action
        self.Action_space = self.env.action_space
        self.action_list = np.linspace(self.env.action_space.low[0], self.env.action_space.high[0], Nbr_action)

        self.memory = utils.ReplayBuffer(max_size, self.Nbr_features, self.Action_space.shape[0])
        self.noise = utils.OUNoise(0)

        self.Q = QNetwork(alpha, self.Nbr_features, hidden_size, self.Nbr_action, name='Q')
        self.Q_target = QNetwork(alpha, self.Nbr_features, hidden_size, self.Nbr_action, name='Q_target')
        self.Q_target.load_state_dict(self.Q.state_dict())

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(self.Nbr_action, 1)[0]
        else:
            state = T.tensor(observation, dtype=T.float32).to(self.Q.device)
            Q_values = self.Q.forward(state)
            action_idx = T.argmax(Q_values).item()
        return self.action_list[action_idx], action_idx

    def update_epsilon(self, episode):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01) * (episode < self.epsilon_end)

    def save_models(self):
        self.Q.save_checkpoint()
        self.Q_target.save_checkpoint()

    def load_models(self):
        self.Q.load_checkpoint()
        self.Q_target.load_checkpoint()

    def do_episode(self, N):
        observation, _ = self.env.reset()
        score = 0
        step = 0
        mean_loss = 0
        while True:
            action, action_idx = self.choose_action(observation)
            next_obs, reward, terminated, truncated, _ = self.env.step([action])
            if terminated:
                reward = -100
            score += reward

            self.memory.add(observation, action_idx, reward, next_obs, terminated)

            step += 1
            observation = next_obs
            loss = self.learn()
            self.update_network_weights()
            self.update_epsilon(N)
            if loss is not None:
                mean_loss += loss

            if terminated or truncated:
                break
        return score, step, mean_loss / step

    def train(self, n_episodes=100):
        writer = SummaryWriter()
        scores = []
        for epoch in tqdm(range(n_episodes)):
            score, step, loss = self.do_episode(epoch)
            scores.append(score)
            if loss is not None:
                writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("score/train", score, epoch)
            writer.add_scalar("epsilon/train", self.epsilon, epoch)
        writer.close()
        return scores

    def learn(self):
        """Updates the neural network model by performing a learning
         step using the Q-learning algorithm.
        """
        if self.memory.used_size < self.batch_size:
            return  # Not enough samples in the memory buffer for a batch

        # Retrieve a batch of samples from the memory buffer
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)

        # Convert data to PyTorch tensors and move to the specified device
        states = T.tensor(states).to(self.Q.device)
        actions = T.tensor(actions, dtype=T.int64).to(self.Q.device)
        rewards = T.tensor(rewards).to(self.Q.device)
        states_ = T.tensor(states_).to(self.Q.device)
        dones = T.tensor(dones).to(self.Q.device)

        # Compute Q-values for the next states using the target Q-network
        with T.no_grad():
            Q_next = self.Q_target.forward(states_)
        # Calculate the target Q-values using immediate rewards and discounted maximum Q-value of next states
        target = rewards + self.gamma * T.max(Q_next, dim=1)[0] * (1 - dones)
        target = target.unsqueeze(1)  # Add an extra dimension to match the shape of Q-values
        # Compute Q-values for the current states using the main Q-network
        Q = self.Q.forward(states)
        # Extract Q-values corresponding to chosen actions
        Q_values = Q.gather(1, actions)
        loss = F.smooth_l1_loss(Q_values, target)
        # Calculate the loss using mean squared error (MSE) between Q-values and target Q-values
        self.Q.optimizer.zero_grad()
        loss.backward()  # Perform backpropagation
        T.nn.utils.clip_grad_value_(self.Q.parameters(), 100)
        self.Q.optimizer.step()
        return loss.item()

    def update_network_weights(self):
        """Updates the weights of the target Q-network by copying the weights
        of the main Q-network using the specified tau value.
        """
        self.Q_target.named_parameters()
        self.Q.named_parameters()

        Q_dict = self.Q.state_dict()
        Q_target_dict = self.Q_target.state_dict()

        for key in Q_dict:
            Q_target_dict[key] = Q_dict[key] * self.tau + Q_target_dict[key] * (1 - self.tau)
            self.Q_target.load_state_dict(Q_target_dict)

        self.Q_target.load_state_dict(Q_target_dict)

    def render(self, mode):
        self.env = gym.make(self.env_name, render_mode=mode)


if __name__ == "__main__":
    Dqn = DQN(alpha=0.001, tau=0.005, env_name='InvertedDoublePendulum-v4',
              gamma=0.99, max_size=10000, batch_size=128, hidden_size=128)
    Dqn.train(10000)
