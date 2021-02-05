import pandas as pd
import numpy as np
import pickle
import tqdm
import os

from os import path
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils import shuffle

from utils import *

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
                            # feature = past_trade
                            label   = processed_trade[item_name][route][year] # - processed_trade[item_name][route][year-1]
                            if label != 0:
                                all_data.append(feature)
                                all_label.append(label)

                            # if year in train_year_list:
                            #     train_data.append(feature)
                            #     train_label.append(label)

                            # if year in test_year_list:
                            #     test_data.append(feature)
                            #     test_label.append(label)


            # =================== all data shuffling ==================
            # all_data, all_label = shuffle(all_data, all_label)

            data_size = len(all_data)
            train_size = int(data_size * 0.8)
            test_size  = data_size - train_size
            train_data,  test_data  = all_data[:train_size],  all_data[train_size:]
            train_label, test_label = all_label[:train_size], all_label[train_size:]

            train_data, test_data   = np.array(train_data), np.array(test_data)
            train_label, test_label = np.array(train_label), np.array(test_label)

            all_train_dict[item_name] = {'data': train_data, 'label': train_label}
            all_test_dict[item_name]  = {'data': test_data,  'label': test_label}

        with open(datasetPath + 'processed/all_regression_data.p', 'wb') as f:
            pickle.dump((all_train_dict, all_test_dict, item_list), f)
    else:
        f = open(datasetPath + 'processed/all_regression_data.p', 'rb')
        all_train_dict, all_test_dict, item_list = pickle.load(f)


    for item_name, _ in item_list:
        train_data, train_label = all_train_dict[item_name]['data'], all_train_dict[item_name]['label']
        test_data,  test_label  = all_test_dict[item_name]['data'],  all_test_dict[item_name]['label']

        scaler = preprocessing.StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data  = scaler.transform(test_data)

        # reg = linear_model.TweedieRegressor(power=1, alpha=0.5, link='log')
        reg = linear_model.Ridge(alpha=0.5)
        # reg = linear_model.Lasso(alpha=0.5)
        # reg = GaussianProcessRegressor()
        reg.fit(train_data, train_label)

        train_predict = reg.predict(train_data)
        train_r2      = r2_score(train_label, train_predict)
        train_mae     = mean_absolute_error(train_label, train_predict)
        train_mape    = mean_absolute_percentage_error(train_label, train_predict)
        train_mse     = mean_squared_error(train_label, train_predict)
        print('training set r2 score: {}, mae: {}, mape: {}, mse: {}'.format(train_r2, train_mae, train_mape, train_mse))

        test_predict = reg.predict(test_data)
        test_r2      = r2_score(test_label, test_predict)
        test_mae     = mean_absolute_error(test_label, test_predict)
        test_mape    = mean_absolute_percentage_error(test_label, test_predict)
        test_mse     = mean_squared_error(test_label, test_predict)
        print('testing set r2 score: {}, mae: {}, mape: {}, mse: {}'.format(test_r2, test_mae, test_mape, test_mse))

