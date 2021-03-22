import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_block(in_channels, out_channels, activation='ReLU'):
    if activation == 'ReLU':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               torch.nn.Dropout(p=0.5),
               nn.LeakyReLU()
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
                linear_block(input_size, 128),
                linear_block(128, 16),
                linear_block(16, output_size, activation=None)
                )

    def forward(self, x):
        return self.model(x)

