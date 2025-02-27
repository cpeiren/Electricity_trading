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

from Dataloader import DataBase
from ModelsV2 import CombinedGridLoadPredictor
#from Models import CombinedGridLoadPredictor
#from Simple_IT import SimpleIT
from Trainner import Trainner
from Loss_fn import Customized_MAE, Customized_MSE

model_config = {'input_dim':16,
                'embed_dim':64,
                'n_heads':4,
                'tf_num_layers':3,
                'grid_size':(8,9),
                'seq_len':168,
                'dropout':0,
                'pred_hours':3,
                }

trainner_config = {'lr':0.01,
                    'scheduler_name':None,
                    'scheduler_max_step':20,
                    'scheduler_min_lr':1e-9,
                    'max_epoch':100,
                    'exit_count':50,
                    'main_dir':'/Users/leroy/Documents/GitHub/Electricity_trading/DART/Trainning',
                    'model_name':'Results/best_model_weights.pt',
                    'output_filaname':'Results/output.txt',
                    'loss_plot':'Results/loss.png',
                    'show':True,
                    'gradient_clip':True,
                    'max_gradient_norm':1,
                    'gamma':2,
                    'poly_power':2,
                    'mixed_percision_type':torch.bfloat16,
                    }

data_config = {'start_train_date':'20230201',
                'end_train_date':'20230508',
                'start_test_date':'20230301',
                'end_test_date':'20230331',
                'data_dir':'/Users/leroy/Documents/GitHub/Electricity_trading/DART/Data',
                'batch_size':8,
                'train_split':0.7,
                }

if __name__ == '__main__':

    database = DataBase(
                        hour_forecast = 3,
                        lookback_days = 7,
                        train_shuffle = True,
                        val_shuffle = True,
                        cv = True,
                        k = 3,
                        trading_hub = 'Houston',
                        predict_variable = f'DA_Price_Houston_excessive',
                        **data_config
                        )

    trainning_dataloader, validation_dataloader = database.create_dataloader()

    model = CombinedGridLoadPredictor(model_config)
    #model = SimpleIT(model_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr = trainner_config['lr'])
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainner = Trainner(model = CombinedGridLoadPredictor(model_config),
                        optim = optimizer,
                        scaler = scaler,
                        device = device,
                        loss_fn = Customized_MSE(),
                        trainning_dataloader = trainning_dataloader,
                        validation_dataloader = validation_dataloader,
                        testing_dataloader = None,
                        **trainner_config)

    trainner.train_main()