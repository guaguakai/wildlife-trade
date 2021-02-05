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

datasetPath = '../data/'
rm_quote = lambda x: x.replace('"', '').replace(' ', '')
rm_quote_float = lambda x: float(x.replace('"', '').replace(' ', ''))

def readTradeData():
    frames = []
    for i in range(1,45):
        datasetName = 'TRADE/trade_db_{}.csv'.format(str(i))
        tmp_df = pd.read_csv(datasetPath + datasetName)

        item_list = tmp_df['Taxon']
        item_dict = {}
        for item in item_list:
            if item in item_dict:
                item_dict[item] += 1
            else:
                item_dict[item] = 1

        item_set = []
        for item in item_dict:
            if item_dict[item] >= 1000:
                item_set.append(item)
        item_set = set(item_set)
        tmp_df = tmp_df[tmp_df['Taxon'].isin(item_set)]
        frames.append(tmp_df)

    trade_df = pd.concat(frames)
    return trade_df

def readCountry():
    filename = datasetPath + 'raw/country.csv'
    dtype = {'Country': str,
            'Alpha-2 code': str,
            'Alpha-3 code': str,
            'Numeric code': np.float64,
            'Latitude (average)': np.float64,
            'Longitude (average)': np.float64
            }
    converters = {'Alpha-2 code': rm_quote,
            'Alpha-3 code': rm_quote,
            'Numeric code': rm_quote,
            'Latitude (average)': rm_quote_float,
            'Longitude (average)': rm_quote_float
            }

    country2ll = {}
    country_df = pd.read_csv(filename, delimiter=',', converters=converters)
    for _, row in country_df.iterrows():
        country2ll[row['Alpha-2 code']] = {
                'country': row['Country'],
                'latitude': row['Latitude (average)'],
                'longitude': row['Longitude (average)'], 
                'Alpha-3 code': row['Alpha-3 code']
                }

    return country2ll

def readWorldIndicators():
    processed_filename = datasetPath + 'processed/WDIEXCEL.csv'
    if path.exists(processed_filename):
        processed_WDI_df = pd.read_csv(processed_filename)
    else:
        filename = datasetPath + 'raw/WDIEXCEL.xlsx'
        WDI_df = pd.read_excel(filename)
        raw_indicators = list(set(WDI_df['Indicator Name']))
        valid_indicators = []
        year_list = [str(year) for year in range(1980,2019)]
        for indicator in raw_indicators:
            number_valid_entries = np.sum(WDI_df[WDI_df['Indicator Name'] == indicator][year_list].isnull().values)
            valid_indicators.append((indicator, number_valid_entries))
            print(number_valid_entries)

        number_indicators = 1000
        valid_indicators = [x[0] for x in sorted(valid_indicators, key=lambda x: x[1])[:number_indicators]]

        default_keys = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        processed_WDI_df = WDI_df[WDI_df['Indicator Name'].isin(valid_indicators)][default_keys + year_list]
        # handling the nan data
        print('Handling nan data...')
        for row_index, row in tqdm.tqdm(processed_WDI_df.iterrows(), total=processed_WDI_df.shape[0]):
            null_list = row[year_list].isnull()
            last_index, next_index = -1, -1
            last_list, next_list = [-1] * len(year_list), [-1] * len(year_list)
            for i in range(len(null_list)):
                if not null_list[i]:
                    last_index = i
                last_list[i] = last_index
                if not null_list[-i-1]:
                    next_index = len(year_list) - i - 1
                next_list[-i-1] = next_index
            for i in range(len(null_list)):
                if last_list[i] == -1 and next_list[i] == -1:
                    processed_WDI_df.at[row_index, year_list[i]] = 0
                elif last_list[i] != -1 and next_list[i] == -1:
                    processed_WDI_df.at[row_index, year_list[i]] = row[year_list[last_list[i]]]
                elif last_list[i] == -1 and next_list[i] != -1:
                    processed_WDI_df.at[row_index, year_list[i]] = row[year_list[next_list[i]]]
                elif last_list[i] != next_list[i]:
                    processed_WDI_df.at[row_index, year_list[i]] = (row[year_list[last_list[i]]] * (next_list[i] - i) + row[year_list[next_list[i]]] * (i - last_list[i])) / (next_list[i] - last_list[i])

        processed_WDI_df.to_csv(processed_filename)

    return processed_WDI_df

def processTradeData(all_trade_df=None, restart=False):
    processed_filename = datasetPath + 'processed/trade.p'
    if not restart and path.exists(processed_filename):
        f = open(processed_filename, 'rb')
        processed_trade, sorted_item_list = pickle.load(f)
    else:
        processed_trade = {}
        # threshold = 1
        # trade_df = trade_df[trade_df['Quantity'] <  threshold]
        # trade_df = all_trade_df[all_trade_df['Quantity'] >= threshold]
        trade_df = all_trade_df
        raw_item_list = list(set(trade_df['Taxon']))
        processed_item_list = []

        for item_name in raw_item_list:
            count = sum(trade_df['Taxon'] == item_name)
            processed_item_list.append((item_name, count))

        sorted_item_list = sorted(processed_item_list, key=lambda x: x[1], reverse=True)[:10]
        year_list = [year for year in range(1980,2019)]

        for _, (item_name, count) in enumerate(tqdm.tqdm(sorted_item_list)):
            processed_trade[item_name] = {}
            route_dict = {}
            # precomputed all the route choices
            item_trade_df = trade_df[(trade_df['Taxon'] == item_name)]
            for i, row in item_trade_df.iterrows():
                exporter = row['Exporter']
                importer = row['Importer']
                quantity = row['Quantity'] # or 1 to just count the trade amount
                if type(exporter) is str and type(importer) is str:
                    route_name = exporter + ' ' + importer
                    if route_name in route_dict:
                        route_dict[route_name] += quantity
                    else:
                        route_dict[route_name] = quantity

            route_list = [(x, route_dict[x]) for x in route_dict]
            route_list = sorted(route_list, key=lambda x: x[1], reverse=True)[:200] # top 100 popular routes

            # iterate through all the possible route
            print(len(route_list), len(year_list))
            for route_name, _ in route_list:
                processed_trade[item_name][route_name] = {}
                exporter, importer = route_name.split(' ')
                route_trade_df = item_trade_df[(item_trade_df['Exporter'] == exporter) & (item_trade_df['Importer'] == importer)]

                for year in year_list:
                    trade_quantity = np.sum(route_trade_df[(route_trade_df['Year'] == year)]['Quantity'])
                    processed_trade[item_name][route_name][year] = trade_quantity

        with open(processed_filename, 'wb') as f:
            pickle.dump((processed_trade, sorted_item_list), f)

    return processed_trade, sorted_item_list

def readTranstats(restart=False):
    processed_filename = '../data/processed/transtats.p'
    if not restart:
        f = open(processed_filename, 'rb')
        transtats_dict = pickle.load(f)
    else:
        filepath = '../data/transtats/'
        transtats_dict = {}
        for year in range(1990, 2020):
            print('Processing year: {}...'.format(year))
            annual_transtats = {}
            filename = filepath + 'transtats_{}.csv'.format(year)
            annual_transtats_df = pd.read_csv(filename)
            for i, row in annual_transtats_df.iterrows():
                seats           = row['SEATS']
                passengers      = row['PASSENGERS']
                freight         = row['FREIGHT']
                mail            = row['MAIL']
                distance        = row['DISTANCE']
                origin_country  = row['ORIGIN_COUNTRY']
                dest_country    = row['DEST_COUNTRY']

                if (type(origin_country) is str) and (type(dest_country) is str):
                    route = origin_country + ' ' + dest_country
                    if route in annual_transtats:
                        annual_transtats[route]['count'] += 1
                        annual_transtats[route]['passengers'] += passengers
                        annual_transtats[route]['freight'] += freight
                    else:
                        annual_transtats[route] = {}
                        annual_transtats[route]['count'] = 1
                        annual_transtats[route]['passengers'] = passengers
                        annual_transtats[route]['freight'] = freight
            transtats_dict[year] = annual_transtats
        f = open(processed_filename, 'wb')
        pickle.dump(transtats_dict, f)

    return transtats_dict

def plotFlights(trade_df, country2ll, item_name, count):
     
    # Add a connection between new york and London
    year_list = [x for x in range(1980, 2019)]
    trade_quantity = {}
    for year in year_list:
        trade_quantity[year] = {}

    for i, row in trade_df.iterrows():
        exporter = row['Exporter']
        importer = row['Importer']
        quantity = row['Quantity']
        year     = row['Year']
        if type(exporter) is str and type(importer) is str:
            route_name = exporter + ' ' + importer
            if year in trade_quantity:
                if route_name in trade_quantity[year]:
                    trade_quantity[year][route_name] += quantity
                else:
                    trade_quantity[year][route_name] = quantity
            # else:
            #     trade_quantity[year] = {}
            #     trade_quantity[year][route_name] = quantity

    fig = plt.figure(figsize=(24,14))
    for year in year_list:
        total_quantity = sum(trade_quantity[year].values())
        # A basic map
        ax  = fig.add_axes([0.1,0.1,0.8,0.8])
        _, raw_item_name = item_name.split('_')
        ax.set_title('Year: {}, item name: {}, total count: {}'.format(year, raw_item_name, total_quantity))

        m = Basemap(llcrnrlon=-160, llcrnrlat=-60,urcrnrlon=160,urcrnrlat=70)
        # m = Basemap(projection='eck4', lon_0=0, resolution='c') # llcrnrlon=-170, llcrnrlat=-70,urcrnrlon=170,urcrnrlat=70)
        m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
        m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')
        m.drawcoastlines(linewidth=0.1, color="white")
        # m.drawcountries()

        for route_name in trade_quantity[year]:
            exporter, importer = route_name.split()
            if (exporter in country2ll) and (importer in country2ll) and (exporter != importer):
                startlat = country2ll[exporter]['latitude']; startlon = country2ll[exporter]['longitude']
                arrlat   = country2ll[importer]['latitude']; arrlon   = country2ll[importer]['longitude']
                quantity = trade_quantity[year][route_name]
                 
                linewidth = 0.2 + quantity / total_quantity * 20
                gcline, = m.drawgreatcircle(startlon,startlat,arrlon,arrlat, linewidth=linewidth, color='orange')
    
                # ==================== adding red arrow to the tail ===================
                path = gcline.get_path()  # get path from the great circle
    
                head = m(arrlon, arrlat)             # get location of arrow's head
                tail = path.vertices[-len(path)//6]  # get location of arrow's tail
                
                # draw annotation with arrow in 'red' color
                # blank text is specified, because we need the arrow only
                # adjust facecolor and other arrow properties as needed
                ax.annotate('',
                            xy=(head[0], head[1]),
                            xycoords='data',
                            xytext=(tail[0], tail[1]),
                            textcoords='data',
                            size=1,
                            arrowprops=dict(width=linewidth, \
                                            headwidth=linewidth*3, \
                                            headlength=linewidth*3, \
                                            facecolor="red", \
                                            edgecolor="none", \
                                            connectionstyle="arc3, rad=0.001") )
    
                head = path.vertices[len(path)//6]       # get location of arrow's tail
                tail = m(startlon, startlat)             # get location of arrow's head
                ax.annotate('',
                            xy=(head[0], head[1]),
                            xycoords='data',
                            xytext=(tail[0], tail[1]),
                            textcoords='data',
                            size=1,
                            arrowprops=dict(width=linewidth, \
                                            headwidth=linewidth*3, \
                                            headlength=linewidth*3, \
                                            facecolor="green", \
                                            edgecolor="none", \
                                            connectionstyle="arc3, rad=0.001") )

        try:
            imagefolder = 'visualization/maps/{}'.format(item_name)
            if not os.path.exists(imagefolder):
                os.makedirs(imagefolder)
            plt.savefig('visualization/maps/{}/{}.png'.format(item_name, year))
        except:
            print(item_name, 'failed...')
        plt.clf()

# def plotFolium(trade_df, country2ll, item_name):
#     import numpy as np
#     import matplotlib.pyplot as plt
#      
#     # A basic map
#     m = Basemap(projection='robin', lon_0=0, resolution='c') # llcrnrlon=-170, llcrnrlat=-70,urcrnrlon=170,urcrnrlat=70)
#     m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
#     m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')
#     m.drawcoastlines(linewidth=0.1, color="white")
#      
#     # Add a connection between new york and London
#     trade_quantity = {}
#     for i, row in trade_df.iterrows():
#         exporter = row['Exporter']
#         importer = row['Importer']
#         quantity = row['Quantity']
#         route_name = exporter + ' ' + importer
#         if route_name in trade_quantity:
#             trade_quantity[route_name] += quantity
#         else:
#             trade_quantity[route_name] = quantity
# 
#     for route_name in trade_quantity:
#         exporter, importer = route_name.split()
#         if (exporter in country2ll) and (importer in country2ll) and (exporter != importer):
#             startlat = country2ll[exporter]['latitude']; startlon = country2ll[exporter]['longitude']
#             arrlat   = country2ll[importer]['latitude']; arrlon   = country2ll[importer]['longitude']
#             print(startlat, startlon, arrlat, arrlon)
#             quantity = trade_quantity[route_name]
#              
#         m.drawgreatcircle(startlon,startlat,arrlon,arrlat, linewidth=1, color='orange')
# 
#     plt.show()

def readGDP():
    filename = '../data/raw/GDP.csv'
    GDP_df = pd.read_csv(filename).fillna(0)
    return GDP_df

def plotItemCounts(sorted_item_list):
    fig = plt.figure(figsize=(24,14))
    item_list = [x[0] for x in sorted_item_list]
    count_list = [x[1] for x in sorted_item_list]
    y_pos = np.arange(len(item_list))
    plt.bar(y_pos, count_list, align='center', alpha=0.5)
    plt.xticks(y_pos, item_list, rotation=90)
    plt.xlabel('item name')
    plt.ylabel('count')
    plt.title('Trading item popularity')

    plt.savefig('visualization/bar/counts.png')
    plt.close()

def plotSmugglingAmount(trade_df, item_name='all'):
    GDP_df = readGDP()
    year_count = {}
    year_list = list(set(trade_df['Year']) - set([2019,2020]))
    GDP_list = []
    trade_list = []
    for year in year_list:
        year_df = trade_df[trade_df['Year'] == year]
        GDP_list.append(sum(GDP_df[str(year)]))
        trade_list.append(len(year_df))

    fig, (ax1, ax2) = plt.subplots(2, 1)
   
    ax1.set_title('GDP')
    ax1.plot(year_list, GDP_list, 'o-')
    ax1.set_ylabel('GDP')
    
    ax2.set_title('Trade amount')
    ax2.plot(year_list, trade_list, 'o-')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Trade')
    
    plt.savefig('visualization/GDP/GDP.png')
    plt.close()


if __name__ == '__main__':
    preprocess = False

    WDI_df = readWorldIndicators()
    country2ll = readCountry()

    transtats_dict = readTranstats(restart=False)

    '''
    if preprocess == True:
        all_trade_df = readTradeData()
        processed_trade, item_list = processTradeData(all_trade_df, restart=True)
    else:
        processed_trade, item_list = processTradeData()

    # item_list = processed_trade.keys() # sorted([(x, np.sum(processed_trade[x].values)) for x in processed_trade.keys()], key=lambda x: x[1])
    train_year_list = [x for x in range(2000,2015)]
    test_year_list  = [x for x in range(2015,2019)]
    all_year_list   = [x for x in range(2000,2019)]

    all_train_dict = {}
    all_test_dict  = {}
    for item_name, _ in item_list:
    # item_name = list(item_list)[0][0]
    # if True:
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
        all_tset_dict[item_name]  = {'data': test_data,  'label': test_label}

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
    # '''

    '''
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
    # '''
    # WDI_df = readWorldIndicators()
    # plotSmugglingAmount(all_trade_df)

