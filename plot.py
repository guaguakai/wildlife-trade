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
    WDI_df = readWorldIndicators()
    country2ll = readCountry()

    # ==================== plotting flight map =====================
    all_trade_df = readTradeData()
    threshold = 10
    small_trade_df = all_trade_df[all_trade_df['Quantity'] <  threshold]
    large_trade_df = all_trade_df[all_trade_df['Quantity'] >= threshold]
    raw_item_list = list(set(large_trade_df['Taxon']))
    processed_item_list = []

    for item_name in raw_item_list:
        count = sum(large_trade_df['Taxon'] == item_name)
        processed_item_list.append((item_name, count))

    sorted_item_list = sorted(processed_item_list, key=lambda x: x[1], reverse=True)

    for i, (item_name, count) in enumerate(sorted_item_list):
        print(item_name)
        plotFlights(large_trade_df[large_trade_df['Taxon'] == item_name], country2ll, item_name='{}_{}'.format(i+1,item_name), count=count)

    # =================== plotting GDP vs trade ===================
    WDI_df = readWorldIndicators()
    plotSmugglingAmount(all_trade_df)

