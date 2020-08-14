# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:58:25 2020

@author: jackm

Uses Neural Networks for both policy and value functions
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# set seeds for reproducability
np.random.seed(0)
torch.manual_seed(0)

dt = 1/52 #  time increments
T = 5*dt # time horizon
x_0 = 1.0 # inital wealth
N = math.floor(T/dt) # number of time steps
EPOCHS = 10000 # number of training episodes

# Some Parameters
mu = 0.5 # drift of stock
sigma = 0.2 # volatility of stock
r =  0.02 # risk free rate
rho = (mu - r)/sigma # sharpe ratio
lam = 0.1 # temperature parameter for entropy
z = 1.05 # desired rate of return

w = (z*np.exp(rho**2*T)-x_0)/(np.exp(rho**2*T)-1) # Lagrange multiplier value

class PolicyNetwork(nn.Module):
    ''' Neural Network for the policy, which is taken to be normally distributed hence
    this network returns a mean and variance - inputs are current wealth and time 
    left in investment horizon'''
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_returns):
        super(PolicyNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_returns = n_returns
        self.lr = lr
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # inputs should be wealth and time to maturity
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,n_returns) # returns mean and sd of normal dist
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        
    def forward(self, observation):
        ''' function to pass data through network'''
        state = torch.Tensor(observation).float().unsqueeze(0)
        x = F.leaky_relu(self.fc1(state), negative_slope=1) # restrictions on first two layers
        x = F.leaky_relu(self.fc2(x), negative_slope=1)
        x = self.fc3(x)
        first_slice = x[:,0]
        second_slice = x[:,1]
        tuple_of_activated_parts = (
                first_slice, # let mean be negative
                F.softplus(second_slice) # make sd positive but dont trap below 1
                )
        out = torch.cat(tuple_of_activated_parts, dim=-1)
        return out
            
class ValueFuncNetwork(nn.Module):
    ''' Neural Network for estimating the value function - inputs are current wealth
    and time left in investment horizon, outputs are value function approximation for 
    input state'''
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_returns):
        super(ValueFuncNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_returns = n_returns
        self.lr = lr
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # input is wealth and time to maturity 
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,n_returns) # output is value of the state
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        
    def forward(self, observation):
        ''' function to pass data through network'''
        state = torch.Tensor(observation).float()
        x = F.leaky_relu(self.fc1(state), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01) # leaky relu to help with derivs
        return x     
    
class Agent(object):
    ''' Investment agent class '''
    def __init__(self, alpha, beta, input_dims, gamma = 1, l1_size = 256, l2_size = 256):
        self.gamma = gamma
        self.reward_memory = [] #  to store episodes rewards
        self.action_memory = [] # to store episodes actions
        self.value_memory = [] # to store value of being in each state
        self.mean_memory = []
        self.sd_memory = []
        self.value = ValueFuncNetwork(beta, input_dims, l1_size, l2_size, n_returns = 1)
        self.policy = PolicyNetwork(alpha, input_dims, l1_size, l2_size, n_returns = 2)
        
    def choose_action(self, current_wealth, t):
        '''Inputs a state of environment, returns an allocation weight '''
        state = [current_wealth, T-t]
        mean, sd = self.policy.forward(state) # obtain mean and sd of policy from policy network
        self.mean = mean.item()
        self.sd = sd
        action_dist = torch.distributions.normal.Normal(mean,sd) # define distribution
        action = action_dist.sample() # sample from distribution     
        log_probs = action_dist.log_prob(action) # obtain log_probs of the action
        self.action_memory.append(log_probs) # store log probs
        self.reward = -lam*action_dist.entropy().item() # store running reward
        
        return action.item()
    
    def get_value(self, current_wealth, t):
        ''' get value of state'''
        state = [current_wealth, T-t]
        value_ = self.value.forward(state)
        return value_
    
    def store_value(self, value):
        '''store values'''
        self.value_memory.append(value)
    
    def store_rewards(self, reward):
        '''store rewards'''
        self.reward_memory.append(reward)
    
    def store_means(self, mean):
        '''store rewards'''
        self.mean_memory.append(mean)

    def store_sds(self, sd):
        '''store rewards'''
        self.sd_memory.append(sd)
        
    def learn(self):
        '''learn - done after each episode, is the heart of the REINFORCE with 
        baseline algorithm'''
        self.policy.optimizer.zero_grad()
        self.value.optimizer.zero_grad()
        
        deltas = [] # to store delta values
        G = [] # to store gain values
        for j in range(N):
            R = 0
            for k in range(j,N):
                R += self.reward_memory[k+1]
            G.append(R)
        for j in range(N):
            deltas.append(G[j] - self.value_memory[j].item())
        self.score = G[0]
            
        '''obtain total losses'''
        val_loss = 0
        for d, vals in zip(deltas, self.value_memory):
            val_loss += -d*vals
        
        policy_loss = 0
        for d, logprob in zip(deltas, self.action_memory):
            policy_loss += d*logprob
            
        total_loss = (val_loss + policy_loss)
        total_loss.backward() # compute gradients
        
        # take steps
        self.value.optimizer.step() 
        self.policy.optimizer.step()
        
        # empty caches
        self.reward_memory = []
        self.action_memory = []
        self.value_memory = []

    
def wealth( x, sample):
    '''obtain new wealth sample'''
    x_new =  x + sigma*sample*(rho*dt + np.sqrt(dt)*np.random.randn())
    return x_new

def true_value(x, t):
    '''True value function'''
    V = ((x - w)**2)*np.exp(-(rho**2)*(T-t)) + (lam*rho**2)*(T**2 - t**2)/4- lam/2*(rho**2*T - np.log((sigma**2)/(np.pi*lam)))*(T-t) - (w - z)**2
    return V

def true_mean(x):
    '''Mean of true optimal policy'''
    y = (rho*(x-w))/sigma
    return y 

def true_sd(t):
    ''' Stan. dev of true optimal policy'''
    sd = (lam*np.exp(rho**2*(T-t)))/(2*sigma**2)
    return sd

def surface_plot(matrix1, matrix2, x_vec, y_vec, **kwargs):
    ''' Function to make 3d plot'''
    # x is cols, y is rows
    (x, y) = np.meshgrid(x_vec, y_vec)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf1 = ax.plot_surface(x, y, matrix1, label = 'Approximated Surface', **kwargs)
    surf2 = ax.plot_surface(x, y, matrix2, label = 'True Surface', **kwargs)
    return (fig, ax, surf1, surf2)

# ------ Main ------ # 
# alpha is learning rate for policy
# beta is learning rate for value function
agent = Agent(alpha = 0.004, beta = 0.005, input_dims = [2], gamma = 1, l1_size = 32, l2_size = 32)
score_mat = np.zeros(EPOCHS)
td_memory = np.zeros(EPOCHS)

# ---------- Training ---------- #
episode_scores = np.array([])
terminal_wealths = [] # list to store each episodes terminal wealth
episode_values = []
for epoch in range(EPOCHS):
    episode_wealths = [] # list to store wealth at each time step of episode
    curr_wealth = np.random.uniform(0.0,1.2) # sample initial wealth
    for i in range(N):
        t = i*dt
        episode_wealths.append(curr_wealth)
        value = agent.get_value( curr_wealth, t) # get value of current state from NN
        agent.store_value(value) # store this value
        action = agent.choose_action( curr_wealth, t) # choose actions 
        agent.store_rewards(agent.reward) # store reward from taking this action
        new_wealth = wealth(curr_wealth, action) # obtain new wealth
        if i == 0:
            episode_values.append(value.item())
            agent.store_sds(agent.sd)
            agent.store_means(agent.mean)
        if new_wealth < 0:
            new_wealth = 0
        curr_wealth = new_wealth # set new wealth to current wealth
        if curr_wealth == 0:
            #break code if wealth falls below zero
            for k in range(N-1-i):
                agent.store_rewards(0)
                agent.store_value(torch.Tensor([0]))
            break
    agent.store_rewards((new_wealth-w)**2) # add terminal wealth to 
    terminal_wealths.append(new_wealth)
                 
    agent.learn() # perform learning step
    episode_scores = np.append(episode_scores,agent.score)

# ----------- TESTING ---------- #
# Some code to test a trained model
testing_actions_init = []
testing_actions_final = []

terminal_wealths = []
for epoch in range(int(EPOCHS/10)):
    episode_wealths = []
    curr_wealth = x_0
    for i in range(N):
        t = i*dt
        episode_wealths.append(curr_wealth)
        action = agent.choose_action( curr_wealth, t) # choose actions 
        new_wealth = wealth(curr_wealth, action) # obtain new wealth
        if i ==0:
            testing_actions_init.append(action)
        elif i == N-1:
            testing_actions_final.append(action)
        if new_wealth < 0:
            new_wealth = 0
        curr_wealth = new_wealth # set new wealth to current wealth
        if curr_wealth == 0:
            break
    terminal_wealths.append(new_wealth)

mean_tw = np.mean(terminal_wealths) # get mean of testing terminal wealths
var_tw = np.var(terminal_wealths)# get variance of testing terminal wealths

# This is where .grad is stored
value_params = list(agent.value.parameters()) 
policy_params = list(agent.policy.parameters())

textstr = '\n'.join((
    r'$\mu=%.2f$' % (mu, ),
    #r'$r=%.2f$' % (r, ),
    r'$\sigma=%.2f$' % (sigma, ),
    #r'$\rho=%.2f$' % (rho, ),
    r'$\lambda=%.2f$' % (lam, )))

plt.figure()
plt.plot(range(EPOCHS), episode_scores)
plt.title('Learning Curve for Reinforce with Baseline - T = 0.5 year')
plt.xlabel('Episodes')
plt.ylabel('G_0 - total reward on episode')

plt.figure()
plt.plot(range(int(EPOCHS/10)), terminal_wealths)
plt.title('Terminal Wealth - T = 0.5 year')
plt.xlabel('Episodes')
plt.ylabel('Terminal Wealth')

# ------- 3D Surface Splot -------- #
x_points = list(np.linspace(0.0, 1.2, 100))
t_points = list(np.linspace(0, T, 100))
tmat_points = [T-i for i in t_points]
values = np.zeros((len(x_points),len(t_points)))
true_values = np.zeros((len(x_points),len(t_points)))

xg,tg = np.meshgrid(x_points,tmat_points)

for i in range(len(t_points)):
    for j in range(len(x_points)):
        values[i,j] = agent.value.forward(torch.Tensor([x_points[j],tmat_points[i]])) - (w-z)**2
        true_values[i,j] = true_value(x_points[j], t_points[i])


(fig, ax, surf1, surf2) = surface_plot(values, true_values, x_points, tmat_points)#, cmap=plt.cm.coolwarm)

ax.set_xlabel('Wealth')
ax.set_ylabel('T-t')
ax.set_zlabel('Value')
fake2Dline = mpl.lines.Line2D([1],[1], linestyle="none", c='b', marker = 'o')
fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
ax.legend([fake2Dline, fake2Dline2], ['Approximated Surface', 'True Surface'], fontsize = 20)

ax.text(1.2,0,0.8, textstr, fontsize = 20)

plt.show()