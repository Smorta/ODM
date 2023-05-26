import matplotlib.pyplot as plt
import numpy as np  
import time
import gymnasium as gym
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from scipy.stats import multivariate_normal 

def make_trajectory(theta, env, size=200):
   trajectory = []

   mu = theta[:4]
   Sigma = 0.8

   s = env.reset()[0]
   for _ in range(size):
      action = np.array([multivariate_normal.rvs(mean = np.dot(mu, s), cov = Sigma)])

      action = np.clip(action, -3, 3)
      s, r, terminated, truncated, info = env.step(action)

      epoch = [s, action, r]
      trajectory.append(epoch)

      if terminated or truncated:
         break

   return trajectory

def main():
   gamma = 0.99
   alpha = 0.00001

   theta = np.random.uniform(-3, 3, 4)

   #theta = [-3.17348774,  8.74304374, -4.36935588, -3.19732884, 0.5]
   len_trajectory = []
   hyp_tune = []
   Gs = []
   mu1 = []
   mu2 = []
   mu3 = []
   mu4 = []
   #Sigmas = []
   for i in tqdm(range(10000)):
      env = gym.make("InvertedPendulum-v4", render_mode= None)
      trajectory = make_trajectory(theta, env)
      len_trajectory.append(len(trajectory))
      mu1.append(theta[0])
      mu2.append(theta[1])
      mu3.append(theta[2])
      mu4.append(theta[3])
      #Sigmas.append(theta[4])

      for t in range(len(trajectory)):
         s1, s2, s3, s4 = trajectory[t][0]
         a = trajectory[t][1]
         r = trajectory[t][2]
         G = 0
         for k in range(t, len(trajectory)):
            G += r * (gamma**(k-t)) 
            r = trajectory[k][2]

         mu = theta[:4] @ np.array([s1, s2, s3, s4])
         Sigma = 0.8

         gradient = np.array([s1, s2, s3, s4]) * ((a - mu) / (Sigma**2))
         theta = theta + alpha*(gamma**t)*G*gradient
      env.close()

   len_trajectory = np.array(len_trajectory)
   len_trajectory = len_trajectory.reshape(-1, 100)
   len_trajectory = np.mean(len_trajectory, axis=1)
   hyp_tune.append(len_trajectory)
   #plt.plot(len_trajectory, label = "len_trajectory")
   #plt.plot(Gs, label = "G")
   # plt.plot(mu1, label = "mu1")
   # plt.plot(mu2, label = "mu2")
   # plt.plot(mu3, label = "mu3")
   # plt.plot(mu4, label = "mu4")
   #plt.plot(Sigmas, label = "Sigma")
   plt.plot(hyp_tune[0])
   plt.grid()
   plt.legend()
   plt.xlabel("epoch")
   plt.ylabel("Mean Trajectory Length")
   plt.text(0.9, -0.1, "x100", transform=plt.gca().transAxes, fontsize=10)
   plt.savefig("normal_reinforce_no_sigma_len_dist.pdf")
   plt.show()
if __name__ == "__main__":
    main()   