import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class Network(nn.Module):
    def __init__(self, alpha, input_size, hidden_size, output_size, name, chkpt_dir='tmp/dqn'):
        super(Network, self).__init__()
        self.alpha = alpha
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_dqn')

        self.fc1 = nn.Linear(self.input_size, self.hidden_size*2)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

        self.init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, path):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(path))


