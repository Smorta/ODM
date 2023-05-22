import matplotlib.pyplot as plt
import numpy as np  

import gymnasium as gym
from tqdm import tqdm

EPOCHS = 20000
ALPHA = 0.001
GAMMA = 0.99
#DISC_STEP = [2, 5, 10, 20]

def make_trajectory(env, theta, size=1000):
    trajectory = []
    grad_log_pi = []
    rewards = []
    s = env.reset()[0]
    for _ in range(size):
        pi = policy(s, theta)
        #actions = np.linspace(-3, 3, DISC_STEP) # discrete actions
        #actions = np.array([-3, -2, -1.5, -1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3])
        actions = np.array([-3, -1.5, -0.75, -0.1, 0, 0.1, 0.75, 1.5, 3])

        action = [np.random.choice(actions , p=pi)]

        index = np.where(actions == action)[0][0]
        next_s, r, terminated, truncated, info = env.step(action)
        epoch = [s, action, r]
        trajectory.append(epoch)

        grad_pi = policy_grad(pi)[index,:] # grad of the action w.r.t. theta
        grad_log_pi.append(s.reshape(4, 1).dot((grad_pi / pi[index])[None,:]))
        rewards.append(r)

        s = next_s
        if terminated or truncated:
            break
    return trajectory, grad_log_pi, rewards
   
def policy(state, theta):
    exp = np.exp(state.dot(theta)) # exp(theta^T * state)	
    return exp/np.sum(exp) # softmax over all actions

def policy_grad(policy):
    temp = policy.reshape(-1,1)
    return np.diagflat(temp) - np.dot(temp, temp.T)


hyp_tune = []
for DISC_STEP in [9]:
    env = gym.make("InvertedPendulum-v4", render_mode= None)

    theta = np.random.rand(4, DISC_STEP)
    len_trajectory = []

    for i in tqdm(range(EPOCHS)):
        trajectory, grad_log_pi, rewards = make_trajectory(env, theta)
        len_trajectory.append(len(trajectory))

        for t in range(len(trajectory)):
            theta = theta + ALPHA * grad_log_pi[t] * sum([ r * (GAMMA ** k) for k,r in enumerate(rewards[t:])])
    
    len_trajectory = np.array(len_trajectory)
    len_trajectory = len_trajectory.reshape(-1, 100)
    len_trajectory = np.mean(len_trajectory, axis=1)
    hyp_tune.append(len_trajectory)
# Calculate the mean along the second axis (column-wise) to get the reduced array

plt.figure()
plt.plot(hyp_tune[0], label='Custom discretization')
# plt.plot(hyp_tune[1], label='5')
# plt.plot(hyp_tune[2], label='7')
# plt.plot(hyp_tune[3], label='9')
# plt.plot(hyp_tune[4], label='12')
plt.grid()
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Mean Trajectory Length', fontsize=16)
plt.text(0.9, -0.1, "x100", transform=plt.gca().transAxes, fontsize=10)
plt.legend()
plt.savefig('REINFORCE_softmax_disc-cust2.png')
plt.show()

    