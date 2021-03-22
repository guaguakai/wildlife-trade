import pandas as pd
import numpy as np
import pickle
import tqdm
import os
import networkx as nx

from os import path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import processTradeData, readTradeData, readCountry, readWorldIndicators, compileData
from model import FullModel
from gcn import GCN

if __name__ == '__main__':
    preprocess = False
    restart = False

    WDI_df = readWorldIndicators()
    country2ll = readCountry()

    if restart:
        all_trade_df = readTradeData()
    else:
        all_trade_df = None
    processed_trade, item_list = processTradeData(all_trade_df, restart=restart)
    _, (gcn_train_data, gcn_test_data) = compileData(preprocess=preprocess, processed_trade=processed_trade, item_list=item_list)

    full_feature_size = 410 # 10

    train_label_mean = torch.mean(torch.cat([gcn_train_data[i][4] for i in range(len(gcn_train_data))]))
    test_label_mean  = torch.mean(torch.cat([gcn_test_data[i][4]  for i in range(len(gcn_test_data))]))

    # compute normalizing constant
    all_train_features = []
    for _, _, _, features, _, _ in gcn_train_data:
        all_train_features.append(features)

    all_train_features = torch.cat(all_train_features)
    train_mean = torch.mean(all_train_features, dim=0)
    train_mean[-10:] = 0
    train_std  = torch.std(all_train_features, dim=0)
    train_std[-10:] = 1

    # training GCN
    # model initiailization
    device = 'cuda'
    lr = 0.001
    model = GCN(raw_feature_size=full_feature_size)
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # loss functions
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    loss_fn = mse_loss

    f_train_result = open('results/gcn/train.csv', 'a')
    f_test_result  = open('results/gcn/test.csv', 'a')

    # training
    for epoch in range(10001):
        train_r2_list, train_mae_list, train_mse_list = [], [], []
        test_r2_list,  test_mae_list,  test_mse_list  = [], [], []
        train_counter, test_counter = 0, 0
        loaders = [
                ('train', gcn_train_data, train_r2_list, train_mae_list, train_mse_list),
                ('test',  gcn_test_data,  test_r2_list,  test_mae_list,  test_mse_list)
                ]

        for mode, loader, r2_list, mae_list, mse_list in loaders:
            # for node_graph, node_features, edge_graph, edge_features, edge_labels, route2index in tqdm.tqdm(loader):
            for _, _, edge_graph, features, labels, route2index in tqdm.tqdm(loader):
                if len(features) == 0:
                    continue
                features = (features - train_mean) / (train_std + 0.001) # for normalization only
                features = features.to(device=device)
                labels = labels.to(device=device)
                # features = features[:,-11:-1]
                edge_index = torch.Tensor(list(edge_graph.edges())).long().t().to(device=device)

                if mode == 'train':
                    model.train()
                    predictions = model(features, edge_index).flatten()
                    loss = loss_fn(predictions, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    model.eval()
                    predictions = model(features, edge_index).flatten()

                labels = labels.detach()
                predictions = predictions.detach()

                r2      = r2_score(labels.cpu().numpy(), predictions.cpu().numpy())
                mae     = mae_loss(labels, predictions).item()
                mse     = mse_loss(labels, predictions).item()
                if mode == 'train':
                    train_counter += len(labels)
                elif mode == 'test':
                    test_counter  += len(labels)

                r2_list.append(r2 * len(labels))
                mae_list.append(mae * len(labels))
                mse_list.append(mse * len(labels))

        train_r2, train_mae, train_mse = np.sum(train_r2_list) / train_counter, np.sum(train_mae_list) / train_counter, np.sum(train_mse_list) / train_counter
        test_r2,  test_mae, test_mse  = np.sum(test_r2_list) / test_counter,  np.sum(test_mae_list) / test_counter, np.sum(test_mse_list) / test_counter
        train_nmse, test_nmse = train_mse / train_label_mean ** 2, test_mse / test_label_mean ** 2

        f_train_result.write('epoch, {}, r2, {}, mae, {}, mse, {}, nmse, {}\n'.format(epoch, train_r2, train_mae, train_mse, train_nmse))
        f_test_result.write('epoch, {}, r2, {}, mae, {}, mse, {}, nmse, {}\n'.format(epoch, test_r2, test_mae, test_mse, test_nmse))

        if epoch % 1 == 0:
            print('Epoch {}, training set r2 score: {}, mae: {}, mse: {}, nmse: {}'.format(epoch, train_r2, train_mae, train_mse, train_nmse))
            print('Epoch {}, testing set r2 score: {},  mae: {}, mse: {}, nmse: {}'.format(epoch, test_r2, test_mae, test_mse, test_nmse))

    f_train_result.close()
    f_test_result.close()
