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
from torch.utils.data import TensorDataset, DataLoader

from utils import *
from model import *

if __name__ == '__main__':
    preprocess = True
    restart = True

    WDI_df = readWorldIndicators()
    country2ll = readCountry()

    if restart:
        all_trade_df = readTradeData()
    else:
        all_trade_df = None
    processed_trade, item_list = processTradeData(all_trade_df, restart=restart)
    item_list = item_list[:10]
    (all_train_dict, all_test_dict), _ = compileData(preprocess=preprocess, processed_trade=processed_trade, item_list=item_list)

    full_feature_size = 410
    export_feature_size = 1001
    attractiveness_feature_size = 1001
    friction_feature_size = 2001

    # merging nn data
    train_data, train_label, test_data, test_label = [], [], [], []
    for item_name, _ in item_list:
        print(item_name)
        tmp_train_data, tmp_train_label = all_train_dict[item_name]['data'], all_train_dict[item_name]['label']
        tmp_test_data,  tmp_test_label  = all_test_dict[item_name]['data'],  all_test_dict[item_name]['label']

        train_data.append(tmp_train_data)
        train_label.append(tmp_train_label)
        test_data.append(tmp_test_data)
        test_label.append(tmp_test_label)

    train_data  = torch.cat(train_data)
    train_label = torch.cat(train_label)
    test_data   = torch.cat(test_data)
    test_label  = torch.cat(test_label)

    # normalization
    train_mean = torch.mean(train_data, dim=0)
    train_std  = torch.std(train_data, dim=0)
    train_mean[-10:] = 0
    train_std[-10:] = 1

    train_data = (train_data - train_mean) / train_std
    test_data  = (test_data  - train_mean) / train_std

    # model initiailization
    device = 'cuda'
    lr = 0.005
    fullModel = FullModel(input_size=full_feature_size)
    fullModel = fullModel.to(device=device)
    optimizer = torch.optim.Adam(fullModel.parameters(), lr=lr)

    # data loader
    training_set = TensorDataset(train_data, train_label)
    testing_set  = TensorDataset(test_data,  test_label)

    training_loader = DataLoader(training_set, batch_size=100, shuffle=True)
    testing_loader  = DataLoader(testing_set,  batch_size=100, shuffle=True)

    # loss functions
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    loss_fn = mse_loss

    f_train_result = open('results/nn/train.csv', 'a')
    f_test_result  = open('results/nn/test.csv', 'a')

    # training
    for epoch in range(10001):
        train_r2_list, train_mae_list, train_mse_list = [], [], []
        test_r2_list,  test_mae_list,  test_mse_list  = [], [], []
        loaders = [
                ('train', training_loader, train_r2_list, train_mae_list, train_mse_list),
                ('test',  testing_loader,  test_r2_list,  test_mae_list,  test_mse_list)
                ]
        for mode, loader, r2_list, mae_list, mse_list in loaders:
            for features, labels in tqdm.tqdm(loader):
                features = features.to(device=device)
                labels = labels.to(device=device)
                if mode == 'train':
                    fullModel.train()
                    predictions = fullModel(features).flatten()
                    loss = loss_fn(predictions, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    fullModel.eval()
                    predictions = fullModel(features).flatten()

                labels = labels.detach()
                predictions = predictions.detach()

                r2      = r2_score(labels.cpu().numpy(), predictions.cpu().numpy())
                mae     = mae_loss(labels, predictions).item()
                mse     = mse_loss(labels, predictions).item()

                r2_list.append(r2)
                mae_list.append(mae)
                mse_list.append(mse)

        train_r2, train_mae, train_mse = np.mean(train_r2_list), np.mean(train_mae_list), np.mean(train_mse_list)
        test_r2,  test_mae,  test_mse  = np.mean(test_r2_list),  np.mean(test_mae_list),  np.mean(test_mse_list)
        train_nmse, test_nmse = train_mse / torch.mean(train_label) ** 2, test_mse / torch.mean(test_label) ** 2

        f_train_result.write('epoch, {}, r2, {}, mae, {}, mse, {}, nmse, {}\n'.format(epoch, train_r2, train_mae, train_mse, train_nmse))
        f_test_result.write('epoch, {}, r2, {}, mae, {}, mse, {}, nmse, {}\n'.format(epoch, test_r2, test_mae, test_mse, test_nmse))

        if epoch % 1 == 0:
            print('Epoch {}, training set r2 score: {}, mae: {}, mse: {}, nmse: {}'.format(epoch, train_r2, train_mae, train_mse, train_nmse))
            print('Epoch {}, testing set r2 score: {},  mae: {}, mse: {}, nmse: {}'.format(epoch, test_r2, test_mae, test_mse, test_nmse))

