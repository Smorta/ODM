import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
csv_file = './fqi/data/May19_15-10-14_TABLET-NQ0DUIMQ.csv'
data = pd.read_csv(csv_file)

# Extract the columns
step = data['Step']
value = data['Value']

smoothing_window = 25
value_s = pd.Series(value)
value_s_ma = value_s.rolling(smoothing_window, min_periods=smoothing_window).mean()

# Create a clean plot
plt.figure()
plt.plot(step, value_s_ma, c='darkorange')
plt.plot(step, value, alpha=0.4, c='darkorange')
plt.xlabel('Episode', fontsize=16)
plt.ylabel(r'$\hat{J}$  ', rotation=0, fontsize=16)
plt.grid(True)
plt.show()




