import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read in data
folder = './fqi/data/'
df_1 = pd.read_csv(folder + 'InvertedPendulum-v4_model3_20000_learning_curve.csv')
df_2 = pd.read_csv(folder + 'InvertedPendulum-v4_model3_50000_learning_curve.csv')
df_3 = pd.read_csv(folder + 'InvertedPendulum-v4_model3_100000_learning_curve.csv')

# Plot data
plt.figure()
plt.plot(df_1['mean'], label='20000')
plt.plot(df_2['mean'], label='50000')
plt.plot(df_3['mean'], label='100000')
plt.xlabel('Episode')
plt.ylabel('Mean steps')
plt.legend()
plt.grid()
folder = './fqi/figs/'
plt.savefig(folder + 'InvertedPendulum-v4_model3_buffer_comp.pdf')

# Read in data
folder = './fqi/data/'
df_1 = pd.read_csv(folder + 'InvertedPendulum-v4_model3_20000_learning_curve.csv')
df_2 = pd.read_csv(folder + 'InvertedPendulum-v4_model5_20000_learning_curve.csv')
df_3 = pd.read_csv(folder + 'InvertedPendulum-v4_model9_20000_learning_curve.csv')

# Plot data
plt.figure()
plt.plot(df_1['mean'], label='3')
plt.plot(df_2['mean'], label='5')
plt.plot(df_3['mean'], label='9')
plt.xlabel('Episode')
plt.ylabel('Mean steps')
plt.legend()
plt.grid()
folder = './fqi/figs/'
plt.savefig(folder + 'InvertedPendulum-v4_model20000_action_comp.pdf')


# Read in data
folder = './fqi/data/'
df_1 = pd.read_csv(folder + 'InvertedDoublePendulum-v4_model3_20000_learning_curve.csv')
df_2 = pd.read_csv(folder + 'InvertedDoublePendulum-v4_model3_50000_learning_curve.csv')
df_3 = pd.read_csv(folder + 'InvertedDoublePendulum-v4_model3_100000_learning_curve.csv')

# Plot data
plt.figure()
plt.plot(df_1['mean'], label='20000')
plt.plot(df_2['mean'], label='50000')
plt.plot(df_3['mean'], label='100000')
plt.xlabel('Episode')
plt.ylabel('Mean steps')
plt.legend()
plt.grid()
folder = './fqi/figs/'
plt.savefig(folder + 'InvertedDoublePendulum-v4_model3_Buffer_comp.pdf')

# Read in data
folder = './fqi/data/'
df_1 = pd.read_csv(folder + 'InvertedDoublePendulum-v4_model3_50000_learning_curve.csv')
df_2 = pd.read_csv(folder + 'InvertedDoublePendulum-v4_model5_50000_learning_curve.csv')
df_3 = pd.read_csv(folder + 'InvertedDoublePendulum-v4_model9_50000_learning_curve.csv')

# Plot data
plt.figure()
plt.plot(df_1['mean'], label='3')
plt.plot(df_2['mean'], label='5')
plt.plot(df_3['mean'], label='9')
plt.xlabel('Episode')
plt.ylabel('Mean steps')
plt.legend()
plt.grid()
folder = './fqi/figs/'
plt.savefig(folder + 'InvertedDoublePendulum-v4_model500000_Action_comp.pdf')