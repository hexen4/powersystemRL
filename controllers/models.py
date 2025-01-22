import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from setting import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorPiModel(nn.Module):
    def __init__(self, num_inputs ,n_action=N_ACTION,init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.num_actions = N_ACTION
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = 1

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
    
    def evaluate(self, state, epsilon=1e-6): 
        '''
        generate sampled action with state as input wrt the policy network => this is the probability of state transition
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action_0 = torch.sigmoid(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick, using sigmoid now
        action = (action_0 + 1) * 0.5
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper -> deleted
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) 
        #- torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std
    
    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(device)
        action = self.action_range* torch.sigmoid(mean + std*z)
        #action = (action + 1) * 0.5 
        action = self.action_range* torch.sigmoid(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        return action

    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(0, 1)
        return self.action_range*a.numpy()
    
    
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
        