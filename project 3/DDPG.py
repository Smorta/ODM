# Custom imports
import utils
from Network import Network

import numpy as np
from tqdm import tqdm
import gymnasium as gym
import time
import os

from torch.utils.tensorboard import SummaryWriter
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DDPG:
    def __init__(self, alpha_actor, alpha_critic, tau=0.01, env_name='Pendulum-v1', gamma=0.99,
                 max_size=100000, batch_size=64, hidden_size=128, name='model'):
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.tau = tau
        self.gamma = gamma
        self.max_size = max_size
        self.batch_size = batch_size
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.hidden_size = hidden_size
        self.observation_space = self.env.observation_space
        self.Nbr_features = self.env.observation_space.shape[0]
        self.Nbr_action = self.env.action_space.shape[0]
        self.action_space = self.env.action_space
        self.noise = utils.OUNoise(np.zeros(self.Nbr_action))
        self.memory = utils.ReplayBuffer(self.max_size, self.Nbr_features, self.Nbr_action)
        self.actor = Network(self.alpha_actor, self.Nbr_features, hidden_size, self.Nbr_action,
                             name=self.env_name+name+'Actor_net', chkpt_dir='./DDPG')
        self.critic = Network(self.alpha_critic, self.Nbr_features + self.Nbr_action, hidden_size, 1,
                              name=self.env_name+name+'Critic_net', chkpt_dir='./DDPG')

        # Implementing target networks
        self.target_actor = Network(self.alpha_actor, self.Nbr_features, hidden_size, self.Nbr_action,
                                    name=self.env_name + '_TargetActor', chkpt_dir='./DDPG')
        self.target_critic = Network(self.alpha_critic, self.Nbr_features + self.Nbr_action, hidden_size, 1,
                                     name=self.env_name + 'TargetCritic', chkpt_dir='./DDPG')

        # Initialize target networks with the same weights as the original networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self, actor_path, critic_path):
        self.actor.load_checkpoint(actor_path)
        self.critic.load_checkpoint(critic_path)
        self.target_actor = Network(self.alpha_actor, self.Nbr_features, self.hidden_size, self.Nbr_action,
                                    name='TargetActor')
        self.target_critic = Network(self.alpha_critic, self.Nbr_features + self.Nbr_action, self.hidden_size, 1,
                                     name='TargetCritic')

    def render(self, mode):
        self.env = gym.make(self.env_name, render_mode=mode)

    def choose_action(self, observation):
        observation = T.tensor(observation, dtype=T.float32).to(self.actor.device)
        with T.no_grad():
            action = self.actor.forward(observation).item()
        return action

    def learn(self):
        """Updates the neural network model by performing a learning
         step using the Q-learning algorithm.
        """
        if self.memory.used_size < self.batch_size:
            return  # Not enough samples in the memory buffer for a batch

        # Retrieve a batch of samples from the memory buffer
        states, actions, rewards, next_state, dones = self.memory.sample(self.batch_size)

        # Convert data to PyTorch tensors and move to the specified device
        states = T.tensor(states).to(self.actor.device)
        actions = T.tensor(actions).to(self.actor.device)
        rewards = T.tensor(rewards).to(self.actor.device)
        next_state = T.tensor(next_state).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        Q_values = self.critic.forward(T.hstack((states, actions)))
        # Calculate the target Q-values using immediate rewards and discounted maximum Q-value of next states

        # ----- IMPLEMENTATION WITHOUT TARGET-----
        # next_action = self.actor.forward(next_state)
        # next_input = T.hstack((next_state, next_action))
        # next_q = self.gamma * self.critic.forward(next_input)
        # target = rewards.unsqueeze(1) + next_q * (1 - dones.unsqueeze(1))
        # target = target
        # ----- FIN IMPLEMENTATION WITHOUT TARGET -----

        # ----- IMPLEMENTATION TARGET-----
        next_action = self.target_actor.forward(next_state)
        next_input = T.hstack((next_state, next_action))
        next_q = self.gamma * self.target_critic.forward(next_input)
        target = rewards.unsqueeze(1) + next_q * (1 - dones.unsqueeze(1))
        # ----- FIN IMPLEMENTATION TARGET -----

        critic_loss = F.smooth_l1_loss(Q_values, target)
        # Calculate the loss using mean squared error (MSE) between Q-values and target Q-values
        self.critic.optimizer.zero_grad()
        critic_loss.backward()  # Perform backpropagation
        T.nn.utils.clip_grad_value_(self.critic.parameters(), 10)
        self.critic.optimizer.step()

        # Update the actor policy using the sampled policy gradient
        actor_loss = -self.critic.forward(T.hstack((states, self.actor.forward(states)))).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()  # Perform backpropagation
        T.nn.utils.clip_grad_value_(self.actor.parameters(), 10)
        self.actor.optimizer.step()

        # Update target networks voir dernière fonction ligne 157
        self.update_target_networks()

        return critic_loss.item(), actor_loss.item()

    def do_episode(self, early_stop=200):
        observation, _ = self.env.reset()
        score = 0
        step = 0
        mean_actor_loss = 0
        mean_critic_loss = 0
        while True:
            action = self.noise.add_noise(self.choose_action(observation))
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            if terminated:
                reward = -100
            score += reward

            self.memory.add(observation, action, reward, next_obs, terminated)

            step += 1
            observation = next_obs
            loss = self.learn()
            if loss is not None:
                mean_critic_loss += loss[0]
                mean_actor_loss += loss[1]
            if terminated or truncated or step > early_stop:
                break
        return score, step, mean_critic_loss / step, mean_actor_loss / step

    def train(self, early_stop=200, capTime=30):
        total_duration = 0
        writer = SummaryWriter(comment='_' + self.env_name + '_DDPG')
        scores = []
        epoch = 0
        solved = False
        solve = 3
        while not solved:
            start_time = time.time()
            score, step, critic_loss, actor_loss = self.do_episode(early_stop)
            scores.append(score)
            duration = time.time() - start_time
            total_duration += duration
            if critic_loss is not None:
                writer.add_scalar("CriticLoss/train", critic_loss, epoch)
                writer.add_scalar("ActorLoss/train", actor_loss, epoch)
            writer.add_scalar("score/train", score, epoch)
            writer.add_scalar("step/train", step, epoch)
            if epoch % 10 == 0:
                print('epoch:', epoch, '; mean_performance: ', step, '; duration:', str(round(duration, 2)) + "s",
                      '(time elapse :' + str(round(total_duration / 60, 2)) + ' min)')
            if step >= early_stop:
                solve -= 1
            else:
                solve = 3
                solved = False
            if solve == 0 or total_duration / 60 > capTime:
                solved = True
            epoch += 1
        writer.close()
        return scores

    # Update target networks : 
    def update_target_networks(self):
        """Updates the weights of the target Q-network by copying the weights
        of the main Q-network using the specified tau value.
        """
        actor_dict = self.actor.state_dict()
        actor_target_dict = self.target_actor.state_dict()

        critic_dict = self.critic.state_dict()
        critic_target_dict = self.target_critic.state_dict()

        for key in actor_dict:
            actor_target_dict[key] = actor_dict[key] * self.tau + actor_target_dict[key] * (1 - self.tau)
        self.target_actor.load_state_dict(actor_target_dict)

        for key in critic_dict:
            critic_target_dict[key] = critic_dict[key] * self.tau + critic_target_dict[key] * (1 - self.tau)
        self.target_critic.load_state_dict(critic_target_dict)


if __name__ == "__main__":
    ddpg = DDPG(alpha_actor=0.0001, alpha_critic=0.001, tau=0.001, env_name='InvertedPendulum-v4', gamma=0.99)
    ddpg.train(200, 60)
    ddpg.save_models()
    # actor_path = './DDPG/modelActor_net_dqn'
    # critic_path = './DDPG/modelCritic_net_dqn'
    # ddpg.load_models(actor_path, critic_path)
