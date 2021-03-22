import pandas as pd
import numpy as np
import pickle
import tqdm
import os
import torch

from os import path
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import seaborn as sns

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils import shuffle

from utils import processTradeData, readTradeData, readCountry, readWorldIndicators, compileData

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
    item_list = item_list[:10]
    gcn_data = compileData(preprocess=preprocess, processed_trade=processed_trade, item_list=item_list)

    data_size = len(gcn_data)
    train_size = int(data_size * 0.8)
    test_size  = data_size - train_size
    gcn_train_data = gcn_data[:train_size]
    gcn_test_data  = gcn_data[train_size:]

    # ========== using gcn data to form nn data ==============
    train_data  = torch.cat([gcn_train_data[i][3] for i in range(len(gcn_train_data))])
    train_label = torch.cat([gcn_train_data[i][4] for i in range(len(gcn_train_data))])
    test_data   = torch.cat([gcn_test_data[i][3] for i in range(len(gcn_test_data))])
    test_label  = torch.cat([gcn_test_data[i][4] for i in range(len(gcn_test_data))])

    full_feature_size = 410
    export_feature_size = 1001
    attractiveness_feature_size = 1001
    friction_feature_size = 2001

    # normalization
    train_mean = torch.mean(train_data, dim=0)
    train_std  = torch.std(train_data, dim=0)
    # train_mean[-10:] = 0
    # train_std[-10:] = 1

    train_data = (train_data - train_mean) / train_std
    test_data  = (test_data  - train_mean) / train_std

    train_data = train_data[:,-1:]
    test_data  = test_data[:,-1:]

    # plotting histograms
    threshold = 200
    total_train_frequency = torch.sum(train_label).item()
    outside_train_frequency = len(train_label[train_label > threshold])
    region_train_label = [int(label.item()) if label.item() <= threshold else np.nan for label in train_label]

    # plt.tight_layout(pad=0)
    plt.figure(figsize=(10,7))
    ax_train = sns.histplot(x=region_train_label, stat='frequency')
    # ax_train.set(xlabel='Trading frequency per route', ylabel='# routes')
    plt.xlabel('Trading frequency per route', fontsize=16)
    plt.ylabel('# routes', fontsize=16)
    plt.title('Training Route Frequency Histogram \n ({:.3f} % of routes with frequency > {})'.format(outside_train_frequency / total_train_frequency * 100, threshold), fontsize=16)
    plt.savefig('results/plots/train_distribution.png')
    plt.close()
    print('Training outlier', outside_train_frequency / total_train_frequency)

    total_test_frequency = torch.sum(test_label).item()
    outside_test_frequency = len(test_label[test_label > threshold])
    region_test_label = [int(label.item()) if label.item() <= threshold else np.nan for label in test_label]

    # plt.tight_layout(pad=0)
    plt.figure(figsize=(10,7))
    ax_test = sns.histplot(x=region_test_label, stat='frequency')
    # ax_test.set(xlabel='Trading frequency per route', ylabel='# routes')
    plt.xlabel('Trading frequency per route', fontsize=16)
    plt.ylabel('# routes', fontsize=16)
    plt.title('Testing Route Frequency Histogram \n ({:.2f} % of routes with frequency > {})'.format(outside_test_frequency / total_test_frequency * 100, threshold), fontsize=16)
    plt.savefig('results/plots/test_distribution.png')
    plt.close()
    print('Testing outlier', outside_test_frequency / total_test_frequency)

    # setting up regression model
    # reg = linear_model.TweedieRegressor(power=1, alpha=0.5, link='log')
    reg = linear_model.Ridge(alpha=0.5)
    # reg = linear_model.Lasso(alpha=0.5)
    # reg = GaussianProcessRegressor()
    reg.fit(train_data, train_label)

    train_predict = train_data[:,-1] 
    # train_predict = np.zeros(train_data.shape[0]) 
    train_predict = reg.predict(train_data)
    train_r2      = r2_score(train_label, train_predict)
    train_mae     = mean_absolute_error(train_label, train_predict)
    train_mse     = mean_squared_error(train_label, train_predict)
    normalization_const = (torch.mean(train_label)) ** 2
    train_nmse    = train_mse / normalization_const
    print('training set r2 score: {}, mae: {}, mse: {}, nmse: {}'.format(train_r2, train_mae, train_mse, train_nmse))

    # test_predict = test_data[:,-1]
    # test_predict = np.zeros(test_data.shape[0])
    test_predict = reg.predict(test_data)
    test_r2      = r2_score(test_label, test_predict)
    test_mae     = mean_absolute_error(test_label, test_predict)
    test_mse     = mean_squared_error(test_label, test_predict)
    normalization_const = (torch.mean(test_label)) ** 2
    test_nmse    = test_mse / normalization_const
    print('testing set r2 score: {}, mae: {}, mse: {}, nmse: {}'.format(test_r2, test_mae, test_mse, test_nmse))

