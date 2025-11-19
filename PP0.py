import os
import json
import numpy as np 
import torch
from torch import nn
from torch.nn.modules import adaptive
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "global_config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)




class NN(nn.Module):
    def __init__(self , input_size , hidden_layers , output_size , act_fun = nn.ReLU) -> None:
        super().__init__()

        layers = []
        _in = input_size
        for h in hidden_layers: 
            layers.append(nn.Linear(_in , h));
            layers.append(act_fun())
            _in = h

        layers.append(nn.Linear(_in , output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

class ActorCritic(nn.Module):
    def __init__(self, Input_Dim , Hidden_Dim , Output_Dim):
        super().__init__()

        #    NOTE:  Need to add dropout 
        self.actor = NN(Input_Dim , Hidden_Dim , Output_Dim);
        self.critic = NN(Input_Dim , Hidden_Dim , 1);

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred


class PPOAgent():
    def __init__(self ,  cfg) -> None:
        #  NOTE: I will put all the configuration for the  PPOAgent 
        self.hidden_layers  = cfg['hidden_layers']
        self.input_size = cfg['input_size'] 
        self.output_size = cfg['output_size']
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon']
        self.enthropy_coff = cfg['enthropy_coff']
        self.agent = ActorCritic(self.input_size , self.hidden_layers , self.output_size) 






    def cal_reward(self, rewards):
        dis_reward =  []
        prev_reward = 0
        for reward in reversed(rewards):
            dis_reward.insert(0 , self.gamma*prev_reward + reward)
            prev_reward = dis_reward[0] 
        
        dis_reward = torch.tensor(dis_reward)
        
        # NOTE: Normaized return 
        dis_reward = (dis_reward - dis_reward.mean())/dis_reward.std()
        return dis_reward
        
    
    def cal_Advantage(self, rewards , values):
        advantages = rewards - values
        # NOTE: Normalized advantage
        advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages
        
    def cal_surrogate_loss(self , log_prob_old ,log_prob_new , advantages):
        advantages = advantages.detach()
        policy_ratio = ( log_prob_new - log_prob_old).exp()
        surrogate_loss_1 = policy_ratio * advantages
        surrogate_loss_2 = torch.clamp(policy_ratio, min=1.0-self.epsilon, max=1.0+self.epsilon) * advantages
        surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
        return surrogate_loss

    def cal_loss(self , surrogate_loss , enthropy , rewards , pred_value):
        enthropy_bonus = self.enthropy_coff * enthropy
        policy_loss = -(surrogate_loss + enthropy_bonus).sum()
        value_loss =  F.smooth_l1_loss(rewards , pred_value).sum()
        return policy_loss , value_loss


    def update(self):
        #  NOTE: Update logic 
        #  :
        pass

    def train(self):
        #  NOTE: train logic
       pass

