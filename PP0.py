import numpy as np 
import torch
from torch import nn




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


class PPOAgent():
    def __init__(self ,  cfg) -> None:
        #  NOTE: I will put all the configuration for the  PPOAgent 
        pass

    def action(self):
        # NOTE:  I will sample action
        # Sample the action
        pass

    def update(self):
        #  NOTE: Update logic 
        #  :
        pass

    def train(self):
        #  NOTE: train logic

