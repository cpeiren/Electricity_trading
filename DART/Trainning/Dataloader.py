from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import os
import sys
from datetime import date, timedelta
import gc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import random
import requests
from math import sqrt
import math
import yaml
import traceback
import pickle

class DataBase:
    def __init__(self,
                start_train_date: str,
                end_train_date: str,
                start_test_date: str,
                end_test_date: str,
                data_dir: str,
                hour_forecast = 3,
                lookback_days = 7,
                batch_size = 32,
                train_shuffle = True,
                val_shuffle = False,
                cv = True,
                k = 5,
                trading_hub = 'Houston',
                train_split = 0.7,
                predict_variable = 'ACTUAL_ERC_HLoad_excessive'
                ):
        

        self.start_train_date = start_train_date
        self.end_train_date = end_train_date
        self.start_test_date = start_test_date
        self.end_test_date = end_test_date
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.cv = cv
        self.k = k
        self.trading_hub = trading_hub
        self.train_split = train_split
        self.predict_variable = predict_variable


        self.price_data_path = f'{data_dir}/2023_24_ERCOT_forecast_actual_data_v3.csv'

        self.hour_forecast = hour_forecast
        self.lookback_days = lookback_days

        self.variable_selected = ['ACTUAL_ERC_Load', 'ACTUAL_ERC_HLoad', 'ACTUAL_ERC_NLoad',
                                    'ACTUAL_ERC_SLoad', 'ACTUAL_ERC_WLoad', 'ACTUAL_ERC_CWind',
                                    'ACTUAL_ERC_NWind', 'ACTUAL_ERC_PWind', 'ACTUAL_ERC_SWind',
                                    'ACTUAL_ERC_Wind', 'ACTUAL_ERC_WWind', 'ACTUAL_ERC_Solar',
                                    f'SP_Price_{trading_hub}',f'DA_Price_{trading_hub}']


    def _get_date_list(self, start_date, end_date):
        '''
        Get date list between start_date and end_date in forat "yyyymmdd"
        '''
        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")
        date_list = []
        
        while start_date <= end_date:
            date_list.append(start_date.strftime("%Y%m%d"))
            start_date += timedelta(days=1)

        return date_list
    
    def _get_price_data(self):
        self.price_data_df = pd.read_csv(self.price_data_path)

        

        self.hub = self.price_data_df.loc[:,['marketday', 'hourending'] + self.variable_selected].copy()

        self.hub.loc[:,f'{self.trading_hub}_DART'] = self.hub.loc[:,f'SP_Price_{self.trading_hub}'] - self.hub.loc[:,f'DA_Price_{self.trading_hub}']
        self.hub['date_int'] = pd.to_datetime(self.hub['marketday'], format='%m/%d/%Y').dt.strftime('%Y%m%d').astype(int)

        self.hub.index = self.hub['date_int'] * 100 + self.hub['hourending'].astype(int)
        #calculate the excessive value
        for value in self.variable_selected + [f'{self.trading_hub}_DART']:
            hour_group_mean = self.hub.groupby('hourending')[value].rolling(self.lookback_days).mean().reset_index().rename(columns={value:f'{value}_mean',
                                                                                                                                                  'level_1':'timestamp'})
            # print(hour_group_mean[hour_group_mean['hourending'] == 2].iloc[:10,:])
            hour_group_mean.index = hour_group_mean.timestamp
            hour_group_std = self.hub.groupby('hourending')[value].rolling(self.lookback_days).std().reset_index().rename(columns={value:f'{value}_std',
                                                                                                                                                  'level_1':'timestamp'})
            hour_group_std.index = hour_group_std.timestamp
            # self.houston = self.houston.merge(hour_group_mean.loc[:,[f'{value}_mean','timestamp']], on=['timestamp'], how='left')
            self.hub  = pd.concat([self.hub, hour_group_mean.loc[:,[f'{value}_mean']],hour_group_std.loc[:,[f'{value}_std']]], axis=1)
            
            self.hub[f'{value}_excessive'] = (self.hub[value] - self.hub[f'{value}_mean']) / self.hub[f'{value}_std']
            
            # print(self.houston[self.houston['date_int'] == 20230201])
            
            # raise Exception('stop')
        self.hub.loc[:,'datetime'] = self.hub.index
        self.hub = self.hub.dropna()

        #get all the unique dates
        all_dates = self.hub['date_int'].unique()

        #create a dataframe with date_int and hour_ending 1-24
        all_hours = range(1, 25)  # Hours 1-24
        multi_index = pd.MultiIndex.from_product(
                                                    [all_dates, all_hours],
                                                    names=['date_int', 'hourending']
                                                )
        
        self.hub = (
                        self.hub.set_index(['date_int', 'hourending'])
                        .reindex(multi_index)
                        .reset_index()
                    )

        # Sort by date and hourending (optional)
        self.hub.sort_values(['date_int', 'hourending'], inplace=True)

        self.hub = self.hub.ffill()
        # print(self.hub[self.hub['marketday'] == '2/15/2023'])

    def _get_date_one_days_later(self,date_str):
        # Convert string to datetime object
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        
        # Add two days
        future_date_obj = date_obj + timedelta(days=1)
        
        # Convert back to string in YYYYMMDD format
        future_date_str = future_date_obj.strftime("%Y%m%d")
    
        return future_date_str
    
    def _get_date_x_days_before(self,date_str,x=1):
        # Convert string to datetime object
        date_obj = datetime.strptime(date_str, "%Y%m%d")

        past_date_obj = date_obj - timedelta(days=x)
        
        # Convert back to string in YYYYMMDD format
        past_date_str = past_date_obj.strftime("%Y%m%d")
    
        return past_date_str
    


    def _get_data_list(self,modes):
        weather_data_list = []
        time_series_lookback_list = []
        time_series_forecast_list = []

        if modes == 'train':
            self.date_list = self._get_date_list(self.start_train_date, self.end_train_date)
        elif modes == 'test':
            self.date_list = self._get_date_list(self.start_test_date, self.end_test_date)
        else:
            raise ValueError('modes should be either train or test')

        for date_str in self.date_list:
            date = int(date_str)

            yesterday = self._get_date_x_days_before(date_str,x=1)

            start_lookback_date = self._get_date_x_days_before(date_str,x=self.lookback_days+1)
            
            #get the time series data
            time_series_lookback = self.hub.loc[(self.hub['date_int'] < int(yesterday))].copy()
            date_unique = time_series_lookback['date_int'].unique()

            if len(date_unique) < self.lookback_days:
                continue

            time_series_lookback = time_series_lookback[time_series_lookback['date_int'].isin(date_unique[-self.lookback_days:])]
            

            time_series_forecast = self.hub.loc[self.hub['date_int'] == date].copy()

            if len(time_series_forecast) == 0:
                continue

            # print(self.hub[self.hub['marketday'] == '2/15/2023'])
            # print(date_str)
            # print(time_series_lookback.date_int.unique())
            # print(time_series_lookback)
            # print(time_series_forecast)
            # raise ValueError

            forecast_date = self._get_date_one_days_later(date_str)
            weather_data_path = f'{self.data_dir}/GFS_forecast/forecast_{date}_on_{yesterday}.pkl'
            # print(weather_data_path)
            with open(weather_data_path, 'rb') as file:
                daily_weather_data = pickle.load(file)

            # print(daily_weather_data.keys())
            #print(daily_weather_data['data'].shape)


            prediction_frequency = len(daily_weather_data['hour_forecast']) 
            if prediction_frequency * self.hour_forecast != 24:
                raise ValueError(f'hour_forecast len {prediction_frequency} or {self.hour_forecast} not correct')

            for i in range(prediction_frequency):
                #print(len(time_series_lookback))
                # if '3/12/2023' in time_series_lookback.marketday.values:
                #     print(len(time_series_lookback))
                #     print(time_series_lookback[time_series_lookback.marketday == '3/12/2023'])
                #     print(time_series_lookback.groupby('marketday').count())
                #     raise Exception('stop')
                time_series_lookback_list.append(time_series_lookback)
                time_series_forecast_list.append(time_series_forecast.iloc[i*self.hour_forecast:(i+1)*self.hour_forecast])
                # print(len(time_series_forecast.iloc[i*self.hour_forecast:(i+1)*self.hour_forecast]))
                weather_data_list.append(daily_weather_data['data'][i,:,:,:])

                # print(self.time_series_forecast_list[0].datetime)
                # print(self.time_series_lookback_list[0])                

                # raise Exception('stop')
            # print(len(self.weather_data_list))
            # print(len(self.time_series_lookback_list))
            # print(len(self.time_series_forecast_list))
        return weather_data_list, time_series_lookback_list, time_series_forecast_list  

    def create_dataloader(self):

        self._get_price_data()

        weather_data_list_train, time_series_lookback_list_train, time_series_forecast_list_train  = self._get_data_list(modes='train')
        filtered_index = list(range(len(weather_data_list_train)))

        if self.cv:
            samples_in_fold = len(filtered_index)// self.k
            split = int(samples_in_fold*self.train_split)

            trainning_index_all = []
            validation_index_all = []

            for i in range(self.k):
                trainning_index = filtered_index[i*samples_in_fold:i*samples_in_fold+split]
                validation_index = filtered_index[i*samples_in_fold+split:(i+1)*samples_in_fold]

                trainning_index_all = trainning_index_all + list(trainning_index)
                validation_index_all = validation_index_all + list(validation_index)
        else:
            split = int(len(filtered_index)*self.train_split)

            trainning_index_all = list(filtered_index[:split])
            validation_index_all = list(filtered_index[split:])

        # print(trainning_index_all)
        # print(validation_index_all)
        # raise ValueError
        excess_variables = [f'{v}_excessive' for v in self.variable_selected + [f'{self.trading_hub}_DART']]

        all_variables = self.variable_selected + excess_variables + [f'{self.trading_hub}_DART']

        trainning_dataset = BTDataset(weather_data_list_train, 
                                      time_series_lookback_list_train, 
                                      time_series_forecast_list_train, 
                                      trainning_index_all, 
                                      all_variables, 
                                      self.predict_variable)
        
        trainning_dataloader = DataLoader(trainning_dataset, 
                                          batch_size=self.batch_size, 
                                          shuffle=self.train_shuffle)

        validation_dataset = BTDataset(weather_data_list_train, 
                                       time_series_lookback_list_train, 
                                       time_series_forecast_list_train, 
                                       validation_index_all, 
                                       all_variables, 
                                       self.predict_variable)
        
        validation_dataloader = DataLoader(validation_dataset, 
                                           batch_size=self.batch_size, 
                                           shuffle=self.val_shuffle)
        
        return trainning_dataloader, validation_dataloader
        
        

class BTDataset(Dataset):
    def __init__(self,
                 weather_data_list,
                 time_series_lookback_list,
                 time_series_forecast_list,
                 filtered_index,
                 feature_required,
                 predict_value,
                 ):
        super().__init__()
        self.weather_data_list = weather_data_list
        self.time_series_lookback_list = time_series_lookback_list
        self.time_series_forecast_list = time_series_forecast_list
        self.index_list = filtered_index
        self.feature_required = feature_required
        self.predict_value = predict_value

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):

        sample_index = self.index_list[index]
        time_series_lookback_df = self.time_series_lookback_list[sample_index]
        time_series_forecast_df = self.time_series_forecast_list[sample_index]
        weather_data = self.weather_data_list[sample_index]

        raw_data_ts = time_series_lookback_df.loc[:,self.feature_required].copy().values
        raw_data_weather = weather_data.copy()
        label = time_series_forecast_df.loc[:,self.predict_value].values

        train_month = time_series_lookback_df['date_int'].values // 100 %100
        train_day = time_series_lookback_df['date_int'].values % 100 
        train_hour = time_series_lookback_df['hourending'].values

        label_month = time_series_forecast_df['date_int'].values // 100 %100
        label_day = time_series_forecast_df['date_int'].values % 100 
        label_hour = time_series_forecast_df['hourending'].values
        
        # print(time_series_lookback_df['date_int'].values)
        # print(train_month)
        # print(train_day)
        # print(train_hour)

        # print(label_month)
        # print(label_day)
        # print(label_hour)
        # raise ValueError
        # print(raw_data_ts.shape)
        # print(raw_data_weather.shape)
        # print(label.shape)
        return {'raw_data_ts':torch.from_numpy(raw_data_ts).float(),
                'train_month':torch.from_numpy(train_month).long(),
                'train_day':torch.from_numpy(train_day).long(),
                'train_hour':torch.from_numpy(train_hour).long(),
                'raw_data_weather':torch.from_numpy(raw_data_weather).float(),
                'label':torch.from_numpy(label).float(),
                'label_month':torch.from_numpy(label_month).long(),
                'label_day':torch.from_numpy(label_day).long(),
                'label_hour':torch.from_numpy(label_hour).long(),
                }