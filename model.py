import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_block(in_channels, out_channels, activation='ReLU'):
    if activation == 'ReLU':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               torch.nn.Dropout(p=0.5),
               nn.ReLU()
               )
    elif activation == 'Sigmoid':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               torch.nn.Dropout(p=0.5),
               nn.Sigmoid()
               )
    elif activation == None:
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               )


class FullModel(nn.Module):
    def __init__(self, input_size=1000, output_size=1):
        super(FullModel, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
                linear_block(input_size, 2048),
                linear_block(2048, 1024),
                linear_block(1024, 32),
                linear_block(32, output_size, activation=None)
                )

    def forward(self, x):
        return self.model(x)

