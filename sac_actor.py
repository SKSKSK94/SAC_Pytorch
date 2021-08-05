#%%
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import math

# continuous action space
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, action_bound, hidden_dim=[256, 256], gamma=0.99, lr=1e-4):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.device = device
        self.action_bound = action_bound

        self.k = (action_bound[1]-action_bound[0])/2

        self.intermediate_dim_list = [state_dim] + hidden_dim

        self.layer_intermediate = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(self.intermediate_dim_list[:-1], self.intermediate_dim_list[1:])]
        )
        
       
        self.mu_log_std_layer = nn.Linear(self.intermediate_dim_list[-1], 2*self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.apply(self.weights_init_)

    
    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):        
        x = state     
        
        for linear_layer in self.layer_intermediate:
            x = self.relu(linear_layer(x))

        # mu = self.k*self.mu_layer(x)
        # mu = torch.clip(mu, self.action_bound[0], self.action_bound[1])
        # log_std = torch.clip(self.log_std_layer(x), min=-20, max=1.5)
        # std = torch.exp(log_std)

        x = self.mu_log_std_layer(x)
        mu, log_std = x[:, :self.action_dim], torch.clamp(x[:, self.action_dim:], -20, 2)
        std = torch.exp(log_std)

        return mu, std
    

    def get_action_log_prob(self, state, stochstic=True):
        mu, std = self.forward(state) # mu = (batch, num_action), std = (batch, num_action)

        ######### log_prob => see the appendix of paper
        var = std**2
        u = mu + std * torch.normal(mean=0, std=1, size=mu.shape).to(self.device).float()
        action = self.k*torch.tanh(u)
        gaussian_log_prob = -0.5*((u - mu)**2/(var+1e-6) + 2*torch.log(std + 1e-6)).sum(dim=-1, keepdim=True) -0.5*mu.shape[-1]*math.log(2*math.pi)

        log_prob = gaussian_log_prob - torch.log(self.k*(1-(action/self.k)**2 + 1e-6)).sum(dim=-1, keepdim=True) # (batch,)

        if not stochstic:
            # action = np.clip(mu.detach().cpu().numpy(), self.action_bound[0], self.action_bound[1])
            action = mu.detach().cpu().numpy() * self.k

        return action, log_prob  

    def get_action(self, state, stochastic=True):
        if not stochastic:
            self.eval()

        mu, std = self.forward(state) # mu = (batch, num_action), std = (batch, num_action)

        ######### log_prob => see the appendix of paper
        var = std**2
        u = mu +  std * torch.normal(mean=0., std=1., size=mu.shape).to(self.device).float()
        action = self.k * torch.tanh(u)

        if not stochastic:
            # action = np.clip(mu.detach().cpu().numpy(), self.action_bound[0], self.action_bound[1])
            action = mu.detach().cpu().numpy() * self.k

        return action      

    def learn(self, log_probs, Q_min, alpha):

        loss = -torch.mean(Q_min - alpha*log_probs)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
