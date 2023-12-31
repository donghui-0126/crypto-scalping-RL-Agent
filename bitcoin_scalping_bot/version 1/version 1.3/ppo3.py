# https://github.com/seungeunrho/minimalRL


#PPO-LSTM
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np

#Hyperparameters
learning_rate = 0.001
gamma         = 0.98
lmbda         = 0.99
eps_clip      = 0.001
K_epoch       = 12

# action과 position을 저장하는 queue의 len
len_action_queue = 32

class PPO3(nn.Module):
    def __init__(self):
        super(PPO3, self).__init__()
        self.data = []

        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(12,2), stride=(12,2))
        self.conv2d_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,17), stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,9), stride=1)
        
        self.lstm1 = nn.LSTM(256, 64)

        self.fc_pi1 = nn.Linear(64, 32)
        self.fc_pi2 = nn.Linear(32+len_action_queue, 16)
        self.fc_pi3 = nn.Linear(16, 5)

        self.fc_v1 = nn.Linear(64, 32)
        self.fc_v2 = nn.Linear(32+len_action_queue, 16)
        self.fc_v3 = nn.Linear(16, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden1, previous_action):
        x = x.reshape([-1,1,24,64])
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))        
        x0 = x.shape[1]
        x1 = x.shape[2]
        x2 = x.shape[3]
        x = x.view(-1, 1, x0*x1*x2)

        x, lstm_hidden1 = self.lstm1(x, hidden1)
        
        x = F.relu(self.fc_pi1(x))
        previous_action = previous_action.view(-1,1,32)
        x =  torch.cat([x, previous_action], axis=-1)
        x = F.relu(self.fc_pi2(x))

        prob = F.softmax(self.fc_pi3(x), dim=2)

        return prob, lstm_hidden1 

    def v(self, x, hidden1, previous_action):
        x = x.reshape([-1,1,24,64])
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))        
        x0 = x.shape[1]
        x1 = x.shape[2]
        x2 = x.shape[3]
        x = x.view(-1, 1, x0*x1*x2)

        x, lstm_hidden1 = self.lstm1(x, hidden1)
        
        x = F.relu(self.fc_v1(x))
        previous_action = previous_action.view(-1,1,32)
        x =  torch.cat([x, previous_action], axis=-1)
        
        x = F.relu(self.fc_v2(x))

        v = self.fc_v3(x)

        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst, position_action_lst = [], [], [], [], [], [], [], [], []


        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done, position_action = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            position_action_lst.append([position_action])

        s, a, r, s_prime, done_mask, prob_a, position_action = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst, dtype=torch.float),\
                                         torch.tensor(position_action_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0], position_action

    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in_1, h1_in_2), (h1_out_1, h1_out_2), position_action = self.make_batch()
        first_hidden  = (h1_in_1.detach(), h1_in_2.detach())
        second_hidden  = (h1_out_1.detach(), h1_out_2.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden, position_action).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden, position_action).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)



            pi, _ = self.pi(s, first_hidden, position_action) 
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()