import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


x = np.linspace(-3, 3, 100)

mu1, sigma1 = 0, 0.8  
y1 = np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) / (sigma1 * np.sqrt(2 * np.pi))

mu3, sigma3 = 0, 2.8  
y3 = np.exp(-((x - mu3) ** 2) / (2 * sigma3 ** 2)) / (sigma3 * np.sqrt(2 * np.pi))


plt.plot(x, y1, label='Beginning of the simulation')
plt.plot(x, y3, label='End of the simulation')


plt.xlim(-3, 3)
plt.xlabel('Action space')
plt.ylabel('Probability Density')
plt.grid()

plt.legend()
plt.savefig("norm_distr.pdf")
plt.show()