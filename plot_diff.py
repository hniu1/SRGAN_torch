import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pickle
# import argparse
# from sklearn.preprocessing import StandardScaler
from srgan_torch import SRGAN_g_lr_26, SRGAN_g_hr_26_64RB
from dataread import read_Daymet_yearly
from tqdm import tqdm
import os

class GCM_Visual:
    def __init__(self, downscale_version, variable):
        self.downscale_version = downscale_version
        self.variable = variable

        self.path_downscaling = f'./gcm_ds/{self.downscale_version}/{variable}'
        # self.path_gcm = path_gcm
        self.path_fig = f'{self.path_downscaling}/fig'

        os.makedirs(self.path_fig, exist_ok=True)

    def read_data(self, path):
        y = np.load(path)
        return y
    
    def plot_map(self, y_gcm, y_pred_25, y_pred_4, metric='mean'):
        x_100 = np.load(f'{self.path_downscaling}/x_100.npy')
        # x_25 = np.load(f'{self.path_downscaling}/x_25.npy')
        # x_4 = np.load(f'{self.path_downscaling}/x_4.npy')
        if metric == "95th":
            x_100 = np.percentile(x_100, 95, axis=0)#*86400*1000
            x_25 = np.percentile(x_25, 95, axis=0)#*86400*1000
            x_4 = np.percentile(x_4, 95, axis=0)#*86400*1000
            y_gcm = np.percentile(y_gcm, 95, axis=0)#*86400*1000
            y_pred_25 = np.percentile(y_pred_25, 95, axis=0)#*86400*1000
            y_pred_4 = np.percentile(y_pred_4, 95, axis=0)#*86400*1000
        elif metric == "25th":
            x_100 = np.percentile(x_100, 25, axis=0)#*86400*1000
            x_25 = np.percentile(x_25, 25, axis=0)#*86400*1000
            x_4 = np.percentile(x_4, 25, axis=0)#*86400*1000
            y_gcm = np.percentile(y_gcm, 25, axis=0)#*86400*1000
            y_pred_25 = np.percentile(y_pred_25, 25, axis=0)#*86400*1000
            y_pred_4 = np.percentile(y_pred_4, 25, axis=0)#*86400*1000
        elif metric == 'wet_days':
            # Threshold for wet day (1 mm of precipitation)
            wet_day_threshold = 1.0
            num_years = 40
            # Calculate the number of wet days for each dataset
            x_100 = (x_100 > wet_day_threshold).sum(axis=0) / num_years
            x_25 = (x_25 > wet_day_threshold).sum(axis=0) / num_years
            x_4 = (x_4 > wet_day_threshold).sum(axis=0) / num_years
            y_gcm = (y_gcm > wet_day_threshold).sum(axis=0) / num_years
            y_pred_25 = (y_pred_25 > wet_day_threshold).sum(axis=0) / num_years
            y_pred_4 = (y_pred_4 > wet_day_threshold).sum(axis=0) / num_years
        else:
            x_100 = np.mean(x_100,axis=0)#*86400*1000
            x_25 = np.mean(x_25,axis=0)#*86400*1000
            x_4 = np.mean(x_4,axis=0)#*86400*1000
            y_gcm = np.mean(y_gcm,axis=0)#*86400*1000
            y_pred_25 = np.mean(y_pred_25,axis=0)#*86400*1000
            y_pred_4 = np.mean(y_pred_4,axis=0)#*86400*1000

        ll = -1
        ul = 1
        if (self.variable == "t2"):
            cmap = 'Reds'
            ll = 0
            ul = 5
        else:
            # cmap = 'GnBu'
            cmap = 'Spectral'
            if metric == "95th":
                ll = 0
                ul = np.ceil(np.max(y_pred_25))
            elif metric == "25th":
                ll = 0
                ul = np.ceil(np.max(y_pred_25))
            elif metric == "wet_days":
                ll = 0
                ul = 250
                cmap = 'RdBu'
            elif var == "tmax" or var == "tmin":
                ll = 0
                ul = np.ceil(np.max(y_pred_25))
                cmap = 'RdBu'
            else:
                ll = 0
                ul = 5

        fig, ax = plt.subplots(2, 3, figsize=(16,6))

        mm1= ax[0][0].imshow(y_gcm[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
        ax[0][0].set_title("GCM Raw 100km")
        mm2 = ax[0][1].imshow(y_pred_25[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
        ax[0][1].set_title("Prediction 25km")
        mm3 = ax[0][2].imshow(y_pred_4[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
        ax[0][2].set_title("Prediction 4km")
        mm4= ax[1][0].imshow(x_100[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
        ax[1][0].set_title("Daymet 100km")
        mm5 = ax[1][1].imshow(x_25[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
        ax[1][1].set_title("Daymet 25km")
        mm6 = ax[1][2].imshow(x_4[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
        ax[1][2].set_title("Daymet 4km")

        plt.colorbar(mm1,ax=ax[0][0],shrink=0.2)
        plt.colorbar(mm2,ax=ax[0][1],shrink=0.2)
        plt.colorbar(mm3,ax=ax[0][2],shrink=0.2)
        plt.colorbar(mm4,ax=ax[1][0],shrink=0.2)
        plt.colorbar(mm5,ax=ax[1][1],shrink=0.2)
        plt.colorbar(mm6,ax=ax[1][2],shrink=0.2)
       # ax[0].remove()
        plt.suptitle(f'GCM Downscaling using SRGAN ({metric})')
        plt.tight_layout()
        plt.savefig(f'{self.path_fig}/gcm_{metric}.pdf')
        plt.close(fig)
    

    def plot_diff(self, y_gcm, metric='mean'):
        x_100 = np.load(f'{self.path_downscaling}/x_100.npy')
        # x_25 = np.load(f'{self.path_downscaling}/x_25.npy')
        # x_4 = np.load(f'{self.path_downscaling}/x_4.npy')
        if metric == "95th":
            x_100 = np.percentile(x_100, 95, axis=0)#*86400*1000
            x_25 = np.percentile(x_25, 95, axis=0)#*86400*1000
            x_4 = np.percentile(x_4, 95, axis=0)#*86400*1000
            y_gcm = np.percentile(y_gcm, 95, axis=0)#*86400*1000
            y_pred_25 = np.percentile(y_pred_25, 95, axis=0)#*86400*1000
            y_pred_4 = np.percentile(y_pred_4, 95, axis=0)#*86400*1000
        elif metric == "25th":
            x_100 = np.percentile(x_100, 25, axis=0)#*86400*1000
            x_25 = np.percentile(x_25, 25, axis=0)#*86400*1000
            x_4 = np.percentile(x_4, 25, axis=0)#*86400*1000
            y_gcm = np.percentile(y_gcm, 25, axis=0)#*86400*1000
            y_pred_25 = np.percentile(y_pred_25, 25, axis=0)#*86400*1000
            y_pred_4 = np.percentile(y_pred_4, 25, axis=0)#*86400*1000
        elif metric == 'wet_days':
            # Threshold for wet day (1 mm of precipitation)
            wet_day_threshold = 1.0
            num_years = 40
            # Calculate the number of wet days for each dataset
            x_100 = (x_100 > wet_day_threshold).sum(axis=0) / num_years
            x_25 = (x_25 > wet_day_threshold).sum(axis=0) / num_years
            x_4 = (x_4 > wet_day_threshold).sum(axis=0) / num_years
            y_gcm = (y_gcm > wet_day_threshold).sum(axis=0) / num_years
            y_pred_25 = (y_pred_25 > wet_day_threshold).sum(axis=0) / num_years
            y_pred_4 = (y_pred_4 > wet_day_threshold).sum(axis=0) / num_years
        else: # mean
            x_100 = np.mean(x_100,axis=0)#*86400*1000
            # x_25 = np.mean(x_25,axis=0)#*86400*1000
            # x_4 = np.mean(x_4,axis=0)#*86400*1000
            y_gcm = np.mean(y_gcm,axis=0)#*86400*1000
            # y_pred_25 = np.mean(y_pred_25,axis=0)#*86400*1000
            # y_pred_4 = np.mean(y_pred_4,axis=0)#*86400*1000

        diff1  = y_gcm-x_100
        # diff2  = y_pred_25-x_25
        # diff3  = y_pred_4-x_4

        ul = 5
        ll = -ul
        if var == 'pr' and metric == "25th":
            ul = 2
            ll = -2
        elif metric == "wet_days":
            ul = 20
            ll = -20

        fig, ax = plt.subplots(1, 1, figsize=(16,3))
        mm1= ax.imshow(diff1[::-1,:],vmin=ll,vmax=ul,cmap ='RdBu')
        ax.set_title("100km")
        # mm2 = ax[1].imshow(diff2[::-1,:],vmin=ll,vmax=ul,cmap ='RdBu')
        # ax[1].set_title("25km")
        # mm3 = ax[2].imshow(diff3[::-1,:],vmin=ll,vmax=ul,cmap ='RdBu')
        # ax[2].set_title("4km")

        plt.colorbar(mm1,ax=ax,shrink=0.2)
        # plt.colorbar(mm2,ax=ax[1],shrink=0.2)
        # plt.colorbar(mm3,ax=ax[2],shrink=0.2)
       # ax[0].remove()
        plt.suptitle(f'Diff between GCM and Daymet ({metric})')
        plt.tight_layout()
        plt.savefig(f'{self.path_fig}/diff_{metric}.pdf')
        plt.close(fig)
        print(f'fig saved for Diff between GCM and Daymet ({metric})')

    def plot_gcm(self):
        y_gcm = self.read_data(f'{self.path_downscaling}/y_gcm_100.npy')
        # y_pred_25 = self.read_data(f'{self.path_downscaling}/y_pred_25.npy')
        # y_pred_4 = self.read_data(f'{self.path_downscaling}/y_pred_4.npy')

        # self.plot_map(y_gcm, y_pred_25, y_pred_4, metric='mean')
        # self.plot_map(y_gcm, y_pred_25, y_pred_4, metric='95th')
        # self.plot_map(y_gcm, y_pred_25, y_pred_4, metric='25th')
        
        self.plot_diff(y_gcm, metric='mean')
        # self.plot_diff(y_gcm, y_pred_25, y_pred_4, metric='95th')
        # self.plot_diff(y_gcm, y_pred_25, y_pred_4, metric='25th')
        # if var=='pr':
        #     self.plot_map(y_gcm, y_pred_25, y_pred_4, metric='wet_days')
        #     self.plot_diff(y_gcm, y_pred_25, y_pred_4, metric='wet_days')


version = '0.2'
var='tmax'

GCM_visual = GCM_Visual(downscale_version = version, 
                            variable = var)
GCM_visual.plot_gcm()