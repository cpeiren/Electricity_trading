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

class Customized_MAE(nn.Module):
    def __init__(self):

        super().__init__()

        self.mae = nn.L1Loss()

    def __call__(self, output_dict,label):
        pred = output_dict['pred']
        # print(pred.shape)
        # print(label.shape)
        # raise ValueError
        loss = self.mae(pred,label)

        return {'loss':loss, 
               }
    
class Customized_MSE(nn.Module):
    def __init__(self):

        super().__init__()

        self.mae = nn.MSELoss()

    def __call__(self, output_dict,label):
        pred = output_dict['pred']
        # print(pred.shape)
        # print(label.shape)
        # raise ValueError
        loss = self.mae(pred,label)

        return {'loss':loss, 
               }