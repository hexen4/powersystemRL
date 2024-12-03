import torch
import torch.nn as nn
import torch.nn.functional as F
from setting import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorPiModel(nn.Module):
    def __init__(self, num_inputs ,n_action=N_ACTION,init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 64)
        
        self.mean_linear = nn.Linear(64, n_action)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(64, n_action)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
    

class CriticVModel(nn.Module):
    def __init__(self,state_dim,init_w=3e-3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),  
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.q = nn.Linear(128, 1)
        self.q.weight.data.uniform_(-init_w, init_w)
        self.q.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state = self.fc(state)
        q_value = self.q(state)
        return q_value

class CriticQModel(nn.Module):
    def __init__(self, num_inputs, num_actions = N_ACTION , init_w=3e-3):
        super().__init__()
        
        # Input layer: concatenated state and action
        self.fc = nn.Sequential(
            nn.Linear(num_inputs + num_actions, 256),  
            nn.ReLU(),
            nn.Linear(256, 128),  
            nn.ReLU()
        )
        self.q = nn.Linear(128, 1)
        self.q.weight.data.uniform_(-init_w, init_w)
        self.q.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)
        x = self.fc(state_action)
        q_value = self.q(x)
        return q_value
        