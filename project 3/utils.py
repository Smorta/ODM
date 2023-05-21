import numpy as np


class OUNoise():
    """ The OUNoise class implements the Ornstein-Uhlenbeck process,
     which is a stochastic differential equation used for generating correlated
     random noise. This class provides a way to generate noise with specific
     parameters such as mean, standard deviation, and correlation.
    """
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer:
    def __init__(self, buffer_size, n_states, n_actions):
        self.buffer_size = buffer_size
        self.used_size = 0
        self.cursor = 0

        self.state_buffer = np.zeros((buffer_size, n_states), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.next_state_buffer = np.zeros((buffer_size, n_states), dtype=np.float32)
        self.done_buffer = np.zeros(buffer_size, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Updates the replay buffer with a new experience.

        Parameters:
        ----------
        state: The current state.
        action: The action taken in the current state.
        reward: The reward received for taking the action.
        next_state: The resulting state after taking the action.
        done: A boolean indicating whether the episode has terminated.

        Returns:
        -------
        None
        """
        self.state_buffer[self.cursor] = state
        self.action_buffer[self.cursor] = action
        self.reward_buffer[self.cursor] = reward
        self.next_state_buffer[self.cursor] = next_state
        self.done_buffer[self.cursor] = done

        self.cursor = (self.cursor + 1) % self.buffer_size
        if self.used_size < self.buffer_size:
            self.used_size += 1

    def sample(self, batch_size):
        """Randomly samples a batch of experiences from the replay buffer.

        Parameters:
        batch_size: The number of experiences to sample.

        Returns:
        states: An array of sampled states.
        actions: An array of sampled actions.
        rewards: An array of sampled rewards.
        next_states: An array of sampled next states.
        dones: An array of sampled termination statuses.
        """
        idx = np.random.choice(self.used_size, batch_size, replace=False)

        states = self.state_buffer[idx]
        actions = self.action_buffer[idx]
        rewards = self.reward_buffer[idx]
        next_states = self.next_state_buffer[idx]
        dones = self.done_buffer[idx]

        return states, actions, rewards, next_states, dones


