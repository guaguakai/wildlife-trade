from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv, BatchNorm

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

aggregation_function = 'add' # either mean or add

# Conv = SAGEConv
Conv = GraphConv

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

class GCN(nn.Module):
    def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[32, 32], nn_hidden_layer_sizes=[2048, 1024, 32]):
        super(GCN, self).__init__()

        r0 = raw_feature_size
        r1, r2 = gcn_hidden_layer_sizes
        n1, n2, n3 = nn_hidden_layer_sizes

        # Define the layers of gcn 
        self.gcn1 = Conv(r0, r1, aggr=aggregation_function)
        self.gcn2 = Conv(r1, r2, aggr=aggregation_function)
        # self.gcn3 = Conv(r2, r3, aggr=aggregation_function)
        # self.gcn4 = Conv(r3, r4, aggr=aggregation_function)

        self.batchnorm1 = BatchNorm(r1)
        self.batchnorm2 = BatchNorm(r2)
        # self.batchnorm3 = BatchNorm(r3)
        # self.batchnorm4 = BatchNorm(r4)

        #Define the layers of NN to predict the attractiveness function for every node
        self.nn_linear = nn.Sequential(
                linear_block(r2, n1),
                linear_block(n1, n2),
                linear_block(n2, n3),
                linear_block(n3, 1, activation=None),
                )

        # self.activation = nn.Softplus()
        self.activation = F.relu
        # self.activation = nn.LeakyReLU
        # self.activation = nn.Sigmoid()

        self.dropout = F.dropout

        #self.node_adj=A

    def forward(self, x, edge_index):

        ''' 
        Input:
            x is the nXk feature matrix with features for each of the n nodes.
            A is the adjacency matrix for the graph under consideration
        '''

        x = self.activation(self.gcn1(x, edge_index))
        # x = self.dropout(x)
        x = self.batchnorm1(x)
        x = self.activation(self.gcn2(x, edge_index))
        # x = self.dropout(x)
        x = self.batchnorm2(x)
        # x = self.activation(self.gcn3(x, edge_index))
        # x = self.dropout(x)
        # x = self.batchnorm3(x)
        # x = self.activation(self.gcn4(x, edge_index))
        # x = self.dropout(x)
        # x = self.batchnorm4(x)

        x = self.nn_linear(x)

        return x

class GCNNN(nn.Module):
    def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[16, 16], nn_hidden_layer_sizes=[2048, 1024, 32]):
        super(GCNNN, self).__init__()

        r0 = raw_feature_size
        r1, r2 = gcn_hidden_layer_sizes
        n1, n2, n3 = nn_hidden_layer_sizes

        #Define the layers of NN to predict the attractiveness function for every node
        self.nn_linear = nn.Sequential(
                linear_block(r0, n1),
                linear_block(n1, n2),
                linear_block(n2, n3),
                linear_block(n3, 1, activation=None),
                )

        # self.activation = nn.Softplus()
        self.activation = F.relu
        # self.activation = nn.LeakyReLU
        # self.activation = nn.Sigmoid()

        self.dropout = F.dropout

        #self.node_adj=A

    def forward(self, x, edge_index):

        ''' 
        Input:
            x is the nXk feature matrix with features for each of the n nodes.
            A is the adjacency matrix for the graph under consideration
        '''

        x = self.nn_linear(x)

        return x

