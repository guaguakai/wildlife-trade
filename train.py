import pandas as pd
import numpy as np
import pickle
import tqdm
import os

from os import path
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import torch
from torch.utils.data import TensorDataset, DataLoader

from utils import *
from model import *

if __name__ == '__main__':
    preprocess = False
    restart = False

    WDI_df = readWorldIndicators()
    country2ll = readCountry()

    # '''
    if preprocess == True:
        all_trade_df = readTradeData()
        processed_trade, item_list = processTradeData(all_trade_df, restart=restart)

        # item_list = processed_trade.keys() # sorted([(x, np.sum(processed_trade[x].values)) for x in processed_trade.keys()], key=lambda x: x[1])
        train_year_list = [x for x in range(2000,2015)]
        test_year_list  = [x for x in range(2015,2019)]
        all_year_list   = [x for x in range(2000,2019)]

        all_train_dict = {}
        all_test_dict  = {}
        for item_name, _ in item_list:
            print(item_name)
            train_data,  test_data  = [], []
            train_label, test_label = [], []
            all_data,    all_label  = [], []
            for year in all_year_list:
                for route in processed_trade[item_name]:
                    exporter, importer = route.split(' ')
                    if exporter in country2ll and importer in country2ll:
                        exporter_info = WDI_df[(WDI_df['Country Code'] == country2ll[exporter]['Alpha-3 code'])][str(year)].tolist()
                        importer_info = WDI_df[(WDI_df['Country Code'] == country2ll[importer]['Alpha-3 code'])][str(year)].tolist()
                        past_trade = [processed_trade[item_name][route][past_year] for past_year in range(year-5,year)]
                        if len(exporter_info) == 1000 and len(importer_info) == 1000:
                            feature = exporter_info + importer_info + past_trade + [year]
                            label   = processed_trade[item_name][route][year] # - processed_trade[item_name][route][year-1]
                            if label != 0:
                                all_data.append(feature)
                                all_label.append(label)

            # =================== all data shuffling ==================
            # all_data, all_label = shuffle(all_data, all_label)

            data_size = len(all_data)
            train_size = int(data_size * 0.8)
            test_size  = data_size - train_size
            train_data,  test_data  = all_data[:train_size],  all_data[train_size:]
            train_label, test_label = all_label[:train_size], all_label[train_size:]

            train_data, test_data   = torch.Tensor(train_data),  torch.Tensor(test_data)
            train_label, test_label = torch.Tensor(train_label), torch.Tensor(test_label)

            all_train_dict[item_name] = {'data': train_data, 'label': train_label}
            all_test_dict[item_name]  = {'data': test_data,  'label': test_label}

        with open(datasetPath + 'processed/all_training_data.p', 'wb') as f:
            pickle.dump((all_train_dict, all_test_dict, item_list), f)
    else:
        f = open(datasetPath + 'processed/all_training_data.p', 'rb')
        all_train_dict, all_test_dict, item_list = pickle.load(f)

    full_feature_size = 2006
    export_feature_size = 1001
    attractiveness_feature_size = 1001
    friction_feature_size = 2001

    for item_name, _ in item_list:
        print(item_name)
        train_data, train_label = all_train_dict[item_name]['data'], all_train_dict[item_name]['label']
        test_data,  test_label  = all_test_dict[item_name]['data'],  all_test_dict[item_name]['label']

        # normalization
        train_mean = torch.mean(train_data, dim=0)
        train_std  = torch.std(train_data, dim=0)

        train_data = (train_data - train_mean) / train_std
        test_data  = (test_data  - train_mean) / train_std

        # model initiailization
        lr = 0.01
        fullModel = FullModel(input_size=full_feature_size)
        optimizer = torch.optim.Adam(fullModel.parameters(), lr=lr)

        # data loader
        training_set = TensorDataset(train_data, train_label)
        testing_set  = TensorDataset(test_data,  test_label)

        training_loader = DataLoader(training_set, batch_size=100, shuffle=True)
        testing_loader  = DataLoader(testing_set,  batch_size=100, shuffle=True)

        # loss functions
        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        loss_fn = mae_loss

        # training
        for epoch in range(100):
            train_r2_list, train_mae_list, train_mape_list, train_mse_list = [], [], [], []
            loaders = [
                    ('train', training_loader, train_r2_list, train_mae_list, train_mape_list, train_mse_list),
                    ('test',  testing_loader,  test_r2_list,  test_mae_list,  test_mape_list,  test_mse_list)
                    ]
            for mode, loader, r2_list, mae_list, mape_list, mse_list in loaders:
                for features, labels in training_loader:
                    if mode == 'train':
                        fullModel.train()
                        predictions = fullModel(batch_features)
                        loss = loss_fn(predictions, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        fullModel.eval()
                        predictions = fullModel(batch_features)

                    labels = labels.detach().numpy()
                    predictions = predictions.detach().numpy()

                    r2      = r2_score(labels, predictions)
                    mae     = mean_absolute_error(labels, predictions)
                    mape    = mean_absolute_percentage_error(labels, predictions)
                    mse     = mean_squared_error(labels, predictions)

                r2_list.append(r2)
                mae_list.append(mae)
                mape_list.append(mape)
                mse_list.append(mse)

            train_r2, train_mae, train_mape, train_mse = torch.mean(train_r2_list), torch.mean(train_mae_list), torch.mean(train_mape_list), torch.mean(train_mse_list)
            test_r2,  test_mae,  test_mape,  test_mse  = torch.mean(test_r2_list),  torch.mean(test_mae_list),  torch.mean(test_mape_list),  torch.mean(test_mse_list)
            print('Epoch {}, training set r2 score: {}, mae: {}, mape: {}, mse: {}'.format(epoch, train_r2, train_mae, train_mape, train_mse))
            print('Epoch {}, testing set  r2 score: {}, mae: {}, mape: {}, mse: {}'.format(epoch, test_r2,  test_mae,  test_mape,  test_mse))

