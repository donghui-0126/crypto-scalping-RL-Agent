# https://github.com/seungeunrho/minimalRL


#PPO-LSTM
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import math


class PPO4(nn.Module):
    def __init__(self, action_num=5, len_action_position = 64, learning_rate=0.001, gamma=0.98, lmbda=0.9, eps_clip=0.001, K_epoch=5):
        super(PPO4, self).__init__()
        self.data = [] # transition을 저장하는 임시 공간
        self.action_num = action_num
        self.len_action_position = len_action_position
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip=eps_clip
        self.K_epoch = K_epoch

        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(12, 2), stride=(12, 2))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2d_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 17), stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 9), stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.lstm1 = nn.LSTM(512, 64)

        self.fc_pi1 = nn.Linear(64, 32)
        self.fc_pi2 = nn.Linear(32 + len_action_position, 32)
        self.fc_pi3 = nn.Linear(32, action_num)
        

        self.fc_v1 = nn.Linear(64, 32)
        self.fc_v2 = nn.Linear(32 + len_action_position, 32)
        self.fc_v3 = nn.Linear(32, 1)        

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden1, previous_action_position):
        x = x.reshape([-1, 1, 30, 64])
        x = F.relu(self.bn1(self.conv2d_1(x)))
        x = F.relu(self.bn2(self.conv2d_2(x)))
        x = F.relu(self.bn3(self.conv2d_3(x)))
        
        x = x.view(-1, 1, x.shape[1] * x.shape[2] * x.shape[3])
        x, lstm_hidden1 = self.lstm1(x, hidden1)

        x = self.fc_pi1(x)
        previous_action_position = previous_action_position.view(-1, 1, self.len_action_position)
        x = torch.cat([x, previous_action_position], axis=-1)
        x = F.relu(self.fc_pi2(x))
        prob = F.softmax(self.fc_pi3(x), dim=2)
        return prob, lstm_hidden1

    def v(self, x, hidden1, previous_action_position):
        x = x.reshape([-1, 1, 30, 64])
        x = F.relu(self.bn1(self.conv2d_1(x)))
        x = F.relu(self.bn2(self.conv2d_2(x)))
        x = F.relu(self.bn3(self.conv2d_3(x)))
        
        x = x.view(-1, 1, x.shape[1] * x.shape[2] * x.shape[3])
        x, lstm_hidden1 = self.lstm1(x, hidden1)

        x = F.relu(self.fc_v1(x))
        previous_action_position = previous_action_position.view(-1, 1, self.len_action_position)
        x = torch.cat([x, previous_action_position], axis=-1)

        x = F.relu(self.fc_v2(x))
        v = self.fc_v3(x)
        
        return v


    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst, action_position_lst = [], [], [], [], [], [], [], [], []


        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done, action_position = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            action_position_lst.append([action_position])

        s, a, r, s_prime, done_mask, prob_a, action_position = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst, dtype=torch.float),\
                                         torch.tensor(action_position_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0], action_position

    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in_1, h1_in_2), (h1_out_1, h1_out_2), action_position = self.make_batch()
        first_hidden  = (h1_in_1.detach(), h1_in_2.detach())
        second_hidden  = (h1_out_1.detach(), h1_out_2.detach())

        for i in range(self.K_epoch):
            v_prime = self.v(s_prime, second_hidden, action_position).squeeze(1)
            td_target = r + self.gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden, action_position).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden, action_position) 
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))
            
            entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=2).detach()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + 0.5*F.smooth_l1_loss(v_s, td_target.detach()) + 0.1*entropy

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()