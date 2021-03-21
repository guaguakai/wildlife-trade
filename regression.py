import pandas as pd
import numpy as np
import pickle
import tqdm
import os

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

from utils import *

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

    train_data = train_data[:,-1:]
    test_data  = test_data[:,-1:]

    # plotting histograms
    sns.displot(x=train_label)
    plt.show()

    sns.displot(x=test_label)
    plt.show()

    # setting up regression model
    # reg = linear_model.TweedieRegressor(power=1, alpha=0.5, link='log')
    # reg = linear_model.Ridge(alpha=0.5)
    reg = linear_model.Lasso(alpha=0.5)
    # reg = GaussianProcessRegressor()
    reg.fit(train_data, train_label)

    train_predict = train_data[:,-1] 
    # train_predict = np.zeros(train_data.shape[0]) 
    # train_predict = reg.predict(train_data)
    train_r2      = r2_score(train_label, train_predict)
    train_mae     = mean_absolute_error(train_label, train_predict)
    train_mse     = mean_squared_error(train_label, train_predict)
    normalization_const = (torch.mean(train_label)) ** 2
    train_nmse    = train_mse / normalization_const
    print('training set r2 score: {}, mae: {}, mse: {}, nmse: {}'.format(train_r2, train_mae, train_mse, train_nmse))

    test_predict = test_data[:,-1]
    # test_predict = np.zeros(test_data.shape[0])
    # test_predict = reg.predict(test_data)
    test_r2      = r2_score(test_label, test_predict)
    test_mae     = mean_absolute_error(test_label, test_predict)
    test_mse     = mean_squared_error(test_label, test_predict)
    normalization_const = (torch.mean(test_label)) ** 2
    test_nmse    = test_mse / normalization_const
    print('testing set r2 score: {}, mae: {}, mse: {}, nmse: {}'.format(test_r2, test_mae, test_mse, test_nmse))

