import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV file
def plot_csv(csv_file, save_file, x_label='Episode', y_label=r'$\hat{J}$'):
    csv_file = './data/' + csv_file
    data = pd.read_csv(csv_file)

    # Extract the columns
    step = data['Step']
    value = data['Value']

    smoothing_window = 10
    value_s = pd.Series(value)
    value_s_ma = value_s.rolling(smoothing_window, min_periods=smoothing_window).mean()

    # Create a clean plot
    plt.figure()
    plt.plot(step, value_s_ma, c='darkorange')
    plt.plot(step, value, alpha=0.4, c='darkorange')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig('./figs/' + save_file + '.pdf')


if __name__ == "__main__":
    plot_csv('NoTarget_InvertedDoublePendulum-v4_DDPG_ActorLoss.csv', 'NoTarget_InvertedDoublePendulum-v4_DDPG_ActorLoss', y_label='Loss', x_label='epoch')
    plot_csv('NoTarget_InvertedPendulum-v4_DDPG_ActorLoss.csv', 'NoTarget_InvertedPendulum-v4_DDPG_ActorLoss', y_label='Loss', x_label='epoch')
    plot_csv('Target_InvertedDoublePendulum-v4_DDPG_ActorLoss.csv', 'Target_InvertedDoublePendulum-v4_DDPG_ActorLoss', y_label='Loss', x_label='epoch')
    plot_csv('Target_InvertedPendulum-v4_DDPG_ActorLoss.csv', 'Target_InvertedPendulum-v4_DDPG_ActorLoss', y_label='Loss', x_label='epoch')

    plot_csv('NoTarget_InvertedDoublePendulum-v4_DDPG_CriticLoss.csv',
             'NoTarget_InvertedDoublePendulum-v4_DDPG_CriticLoss', y_label='Loss', x_label='epoch')
    plot_csv('NoTarget_InvertedPendulum-v4_DDPG_CriticLoss.csv', 'NoTarget_InvertedPendulum-v4_DDPG_CriticLoss',
             y_label='Loss', x_label='epoch')
    plot_csv('Target_InvertedDoublePendulum-v4_DDPG_CriticLoss.csv', 'Target_InvertedDoublePendulum-v4_DDPG_CriticLoss',
             y_label='Loss', x_label='epoch')
    plot_csv('Target_InvertedPendulum-v4_DDPG_CriticLoss.csv', 'Target_InvertedPendulum-v4_DDPG_CriticLoss',
             y_label='Loss', x_label='epoch')

    plot_csv('NoTarget_InvertedDoublePendulum-v4_DDPG_step.csv',
             'NoTarget_InvertedDoublePendulum-v4_DDPG_step', y_label='Step', x_label='epoch')
    plot_csv('NoTarget_InvertedPendulum-v4_DDPG_step.csv', 'NoTarget_InvertedPendulum-v4_DDPG_step',
             y_label='Step', x_label='epoch')
    plot_csv('Target_InvertedDoublePendulum-v4_DDPG_step.csv', 'Target_InvertedDoublePendulum-v4_DDPG_step',
             y_label='Step', x_label='epoch')
    plot_csv('Target_InvertedPendulum-v4_DDPG_step.csv', 'Target_InvertedPendulum-v4_DDPG_step',
             y_label='Step', x_label='epoch')