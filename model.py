import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_block(in_channels, out_channels, activation='ReLU'):
    if activation == 'ReLU':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               # torch.nn.Dropout(p=0.5),
               nn.LeakyReLU()
               )
    elif activation == 'Sigmoid':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               # torch.nn.Dropout(p=0.5),
               nn.Sigmoid()
               )

class FullModel(nn.Module):
    def __init__(self, input_size=1000, output_size=1):
        super(FullModel, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
                linear_block(input_size, 2048),
                linear_block(2048, 1024),
                linear_block(1024, 32),
                linear_block(32, output_size)
                )

    def forward(self, x):
        return self.model(x)


class ExportModel(nn.Module):
    def __init__(self, input_size=1000, output_size=1):
        super(ExportModel, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
                linear_block(input_size, 2048),
                linear_block(2048, 1024),
                linear_block(1024, 32),
                linear_block(32, output_size)
                )

    def forward(self, x):
        return self.model(x)

class AttractivenessModel(nn.Module):
    def __init__(self, input_size=1000, output_size=1):
        super(AttractivenessModel, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
                linear_block(input_size, 2048),
                linear_block(2048, 1024),
                linear_block(1024, 32),
                linear_block(32, output_size)
                )

    def forward(self, x):
        return self.model(x)

class FrictionModel(nn.Module):
    def __init__(self, input_size=1000, output_size=1):
        super(FrictionModel, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
                linear_block(input_size, 2048),
                linear_block(2048, 1024),
                linear_block(1024, 32),
                linear_block(32, output_size)
                )

    def forward(self, x):
        return self.model(x)

