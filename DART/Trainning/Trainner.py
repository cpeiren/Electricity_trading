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

class Trainner:
    def __init__(self,
                 model,
                 optim,
                 scaler,
                 device,
                 lr,
                 loss_fn,
                 trainning_dataloader,
                 validation_dataloader,
                 testing_dataloader,
                 scheduler_name = None,
                 scheduler_max_step = 20,
                 scheduler_min_lr = 1e-9,
                 max_epoch = 50,
                 exit_count = 15,
                 main_dir = '/Users/leroy/Documents/GitHub/Electricity_trading/DART/Trainning',
                 model_name = 'best_model_weights.pt',
                 output_filaname = 'output.txt',
                 loss_plot = 'loss.png',
                 show = True,
                 gradient_clip = True,
                 max_gradient_norm = 1,
                 gamma = 2,
                 poly_power = 2,
                 mixed_percision_type = torch.bfloat16
                 ):
        '''
        This Trainer class handles the model training, validation, and testing process. 
        It supports various learning rate schedules, gradient clipping, mixed precision training, 
        and backtesting of the model performance.
        '''
        
        # Initialize model, optimizers, dataloaders, and training settings
        self.model = model.to(device)
        self.trainning_dataloader = trainning_dataloader
        self.validation_dataloader = validation_dataloader
        self.testing_dataloader = testing_dataloader
        self.device = device
        self.optim = optim
        self.loss_fn = loss_fn
        self.lr = lr
        self.scheduler_name = scheduler_name
        self.max_epcoh = max_epoch
        self.exit_count = exit_count
        self.validation_loss_list = []
        self.trainning_loss_list = []

        self.show = show
        self.gradient_clip = gradient_clip
        self.max_gradient_norm = max_gradient_norm
        self.scaler = scaler
        self.mixed_percision_type = mixed_percision_type

        self.main_dir = main_dir
        self.loss_plot = f'{self.main_dir}/{loss_plot}'
        self.model_name = f'{self.main_dir}/{model_name}'
        self.output_filaname = f'{self.main_dir}/{output_filaname}'
        
    
        # Learning rate scheduler
        if self.scheduler_name == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=scheduler_max_step, eta_min=scheduler_min_lr)

        elif self.scheduler_name == 'expo':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=gamma)

        elif self.scheduler_name == 'poly':
            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optim, power=poly_power,total_iters=max_epoch)


    def log_print(self,out):
        '''
        Print logs to both console and output file (if `self.show` is True).
        '''
        if self.show:
            with open(self.output_filaname, "a") as f:
                print(out, file=f)
                print(out)

    def validation(self):
        '''
        Run validation on the validation dataset and return the average loss.
        This function also computes the Sharpe ratio and PnL per trade for validation.
        '''

        loss_list = []
        element_loss = []

        # Disable gradient calculation for validation
        with torch.no_grad():
            self.model.eval()
            if self.show:
                pbar = tqdm(self.validation_dataloader, desc='Validation',unit='batch')
            else:
                pbar = self.validation_dataloader

            for batch in pbar:
                # Move batch data to the correct device (GPU or CPU)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                label = batch['label']

                output_dict = self.model(batch)

                # Calculate the loss
                loss_dict = self.loss_fn(output_dict,label)

                loss = loss_dict['loss']
                loss_list.append(loss.cpu().numpy())
                element_loss.append([i.item() for i in loss_dict.values()])

        element_loss_df = pd.DataFrame(element_loss,columns=list(loss_dict.keys())).mean()
        self.log_print(f'validation element wise loss: {element_loss_df}')
        
        return np.mean(loss_list)

    def trainning_epoch(self):
        '''
        Run one epoch of training and return the average training loss and validation loss.
        This function also computes the Sharpe ratio for the training data.
        '''
        epoch_loss = []
        element_loss = []

        self.model.train()
        if self.show:
            pbar = tqdm(self.trainning_dataloader, desc='Trainning',unit='batch')
        else:
            pbar = self.trainning_dataloader


        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            label = batch['label']

            # Mixed precision training (if using GPU)
            if self.device == 'cuda':
                with torch.cuda.amp.autocast(dtype=self.mixed_percision_type):
                    output_dict = self.model(batch)
                    loss_dict = self.loss_fn(output_dict,label)
                    loss = loss_dict['loss']
                # Backpropagation
                self.scaler.scale(loss).backward()
            else:
                output_dict = self.model(batch)
                loss_dict = self.loss_fn(output_dict,label)
                loss = loss_dict['loss']

                loss.backward()

            

            # Gradient clipping to avoid exploding gradients
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)

            # Optimizer step and scaler update (for mixed precision)
            if self.device == 'cuda':
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
            else:
                self.optim.step()
                self.optim.zero_grad()


            if self.show:
                show_dict = {}

                for name,value in loss_dict.items():
                    show_dict[name] = value.item()

                pbar.set_postfix(show_dict)

            epoch_loss.append(loss.item())
            element_loss.append([i.item() for i in loss_dict.values()])

        # Run validation and calculate Sharpe ratio for training
        loss_valid = self.validation()

        #get element wise loss
        element_loss_df = pd.DataFrame(element_loss,columns=list(loss_dict.keys())).mean()
        self.log_print(f'training element wise loss: {element_loss_df}')

        return np.mean(epoch_loss), loss_valid

    def train_main(self):
        '''
        Main training loop. It performs training and validation, saves the best model,
        and implements early stopping if validation loss does not improve.
        '''

        best_loss = 1e7

        flat_count = 0
        for i in range(self.max_epcoh):

            train_loss, val_loss = self.trainning_epoch()
            self.trainning_loss_list.append(train_loss)


            self.log_print(f'epoch {i} loss {train_loss}; validation {i} loss {val_loss}')

            # Adjust learning rate with scheduler if enabled
            if self.scheduler_name is not None:
                self.scheduler.step()
                self.lr = self.scheduler.get_last_lr()
            self.log_print(f'current lr is :{self.lr}')

            #Free up memory for GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            # Save model if validation loss improves
            if val_loss < best_loss:
                torch.save(self.model.state_dict(), self.model_name)
                best_loss = val_loss
                flat_count = 0
            else:
                flat_count += 1

            if flat_count > self.exit_count:
                self.log_print('validation loss is not improving, stop')
                break

            gc.collect()
            self.validation_loss_list.append(val_loss)

        # Load the best model for final testing
        self.log_print('load model with best validation score')
        checkpoint = torch.load(self.model_name)
        self.model.load_state_dict(checkpoint)

        plt.figure(figsize=(8, 6))
        plt.plot(self.trainning_loss_list, label="Training Loss", marker="o")
        plt.plot(self.validation_loss_list, label="Validation Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curve")
        plt.legend()
        plt.grid()
        plt.savefig(self.loss_plot)  # Save the figure
        # plt.show()

        if not self.show:
            self.log_print(f'finish trainning, best valid score: {best_loss}')
        return best_loss