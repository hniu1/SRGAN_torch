'''
The script is used to use Daymet trained model to do GCM downscaling.
Author: Haoran Niu
Date: Dec 2024
'''

import os
# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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


# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tt5  = 365*5  + 2
tt10 = 365*10 + 3
tt20 = 365*20 + 5
tt30 = 365*30 + 8
tt35 = 365*35 + 9
tt40 = 365*40 + 10

class GCM_Downscaling:
    def __init__(self, version_lr, version_hr, downscale_version, elevation, variable, path_gcm):
        self.version_lr = version_lr
        self.version_hr = version_hr
        self.downscale_version = downscale_version
        self.elevation = elevation
        self.variable = variable
        self.tt = tt40

        self.checkpoint_dir = f"./models/v{self.version_lr}"
        self.path_output = f'./output/v{self.version_lr}'
        self.checkpoint_dir_hr = f'./models/v{self.version_hr}'
        self.path_output_hr = f'./output/v{self.version_hr}'
        
        self.path_downscaling = f'./gcm_ds/{self.downscale_version}/{variable}/{gcm}'
        # self.path_gcm = f'/mnt/data/ClimateSR/gcm/pr_day_MPI-ESM1-2-HR_ssp585_r1i1p1f1_gn_198001-201912_1deg_Daymet_ERA5_VICa_BC_dims.nc'
        self.path_gcm = path_gcm
        os.makedirs(self.path_downscaling, exist_ok=True)
    
        with open(f'{self.checkpoint_dir}/scaler.pkl', 'rb') as f:
            self.loaded_scaler = pickle.load(f)
        with open(f'{self.checkpoint_dir_hr}/scaler.pkl', 'rb') as f:
            self.loaded_scaler_hr = pickle.load(f)
        

    def test_data_loading(self, mode):
        if mode == '1':
            test_lr = self.readGCM(self.tt, self.path_gcm, self.loaded_scaler, self.variable)
            if self.elevation:
                elev_scaled = np.load(f'{self.path_output}/elev_lr_scaled.npy')
                elev_scaled = np.tile(elev_scaled[0,:,:,:],(np.shape(test_lr)[0],1,1,1))
                test_lr = np.concatenate((test_lr,elev_scaled),axis=3)
        elif mode == '2':
            y_test = np.load(f'{self.path_downscaling}/y_pred_25.npy')
            scaler  = self.loaded_scaler_hr
            shp = y_test.shape
            lr_test        = y_test.flatten()
            lr_test_scaled = scaler.transform(lr_test.reshape(-1,1))
            test_lr = np.reshape(lr_test_scaled,(shp[0],shp[1],shp[2],1))
            if self.elevation:
                elev_scaled = np.load(f'{self.path_output_hr}/elev_lr_scaled.npy')
                elev_scaled = np.tile(elev_scaled[0,:,:,:],(np.shape(test_lr)[0],1,1,1))
                test_lr = np.concatenate((test_lr,elev_scaled),axis=3)
        return test_lr

    def readGCM(self, tt, path_gcm, loaded_scaler, var):
        fGCM = Dataset(path_gcm)
        y_test     = fGCM.variables[f'{var}'][:tt,:,:].astype(np.float32)
        np.save(f'{self.path_downscaling}/y_gcm_100.npy', np.ma.getdata(y_test))

        if not os.path.exists(f'{self.path_downscaling}/x_100.npy'):
            x_100 = np.load(f'./gcm_ds/daymet/Daymet_{var}_1980_2019_1deg.npy')[:tt,:,:]
            np.save(f'{self.path_downscaling}/x_100.npy', np.ma.getdata(x_100))
            del x_100
            x_25 = np.load(f'./gcm_ds/daymet/Daymet_{var}_1980_2019_0.25deg.npy', mmap_mode='r')[:tt,:,:]
            np.save(f'{self.path_downscaling}/x_25.npy', np.ma.getdata(x_25))
            del x_25
            x_4 = np.load(f'./gcm_ds/daymet/Daymet_{var}_1980_2019_0.0416deg.npy', mmap_mode='r')[:tt,:,:]
            np.save(f'{self.path_downscaling}/x_4.npy', np.ma.getdata(x_4))
            del x_4

        scaler  = loaded_scaler
        shp = y_test.shape
        lr_test        = y_test.flatten()
        lr_test_scaled = scaler.transform(lr_test.reshape(-1,1))
        lr_test_scaled = np.reshape(lr_test_scaled,(shp[0],shp[1],shp[2],1))
        return lr_test_scaled
    
    def load_model(self, mode):
        # Initialize models
        if mode == '1':
            if self.elevation:
                G = SRGAN_g_lr_26(in_channels=2).to(device)
            else:
                G = SRGAN_g_lr_26(in_channels=1).to(device)
            # D = SRGAN_d_lr_odd(hr_size=test_lr[0].shape[0]*test_lr[0].shape[1]*4*4).to(device)
        elif mode == '2':
            # Initialize models
            if self.elevation:
                G = SRGAN_g_hr_26_64RB(in_channels=2).to(device)
            else:
                G = SRGAN_g_hr_26_64RB(in_channels=1).to(device)
            # D = SRGAN_d_hr_odd(hr_size=test_lr[0].shape[0]*test_lr[0].shape[1]*6*6).to(device)
        return G

    def downscale_100_25(self):
        G = self.load_model(mode='1')

        if torch.cuda.device_count() >= 1:
            G = nn.DataParallel(G)

        G.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, 'g.pth')))
        G.eval()

        test_lr = self.test_data_loading(mode='1')

        valid_lr_img_tensor = torch.tensor(test_lr, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Convert to PyTorch tensor and move to device
        batch_size = 64  # Adjust batch size according to your GPU memory

        outputs = []
        with torch.no_grad():  # No gradient calculation for inference
            for i in tqdm(range(0, len(valid_lr_img_tensor), batch_size), desc="Downscale 100 to 25km: "):
                batch = valid_lr_img_tensor[i:i+batch_size]
                out = G(batch).cpu().numpy()  # Move to CPU and convert to numpy
                outputs.append(out)
        out = np.concatenate(outputs, axis=0)
        # Convert from CHW to HWC
        out = out.transpose(0, 2, 3, 1)
        shp = out.shape
        tt = shp[0]
        nhr1 = shp[1]
        nhr2 = shp[2]
        y = out.flatten()
        yinv = self.loaded_scaler.inverse_transform(y.reshape(-1, 1))
        if self.variable == 'pr' or self.variable == 'prcp':
            yinv[yinv < 0] = 0
        yinv = np.reshape(yinv, (tt, nhr1, nhr2))
        np.save(f'{self.path_downscaling}/y_pred_25.npy', yinv)
        print(f'Downscaling from 100 to 25 km finished and predicted data saved to {self.path_downscaling}/y_pred.npy')

    def downscale_25_4(self):
        G = self.load_model(mode='2')

        if torch.cuda.device_count() >= 1:
            G = nn.DataParallel(G)

        G.load_state_dict(torch.load(os.path.join(self.checkpoint_dir_hr, 'g.pth')))
        G.eval()

        test_lr = self.test_data_loading(mode='2')

        valid_lr_img_tensor = torch.tensor(test_lr, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Convert to PyTorch tensor and move to device
        batch_size = 8  # Adjust batch size according to your GPU memory

        outputs = []
        with torch.no_grad():  # No gradient calculation for inference
            for i in tqdm(range(0, len(valid_lr_img_tensor), batch_size), desc="Downscale 25 to 4km: "):
                batch = valid_lr_img_tensor[i:i+batch_size]
                out = G(batch).cpu().numpy()  # Move to CPU and convert to numpy
                outputs.append(out)
        out = np.concatenate(outputs, axis=0)
        # Convert from CHW to HWC
        out = out.transpose(0, 2, 3, 1)
        shp = out.shape
        tt = shp[0]
        nhr1 = shp[1]
        nhr2 = shp[2]
        y = out.flatten()
        yinv = self.loaded_scaler_hr.inverse_transform(y.reshape(-1, 1))
        if self.variable == 'pr' or self.variable == 'prcp':
            yinv[yinv < 0] = 0
        yinv = np.reshape(yinv, (tt, nhr1, nhr2))
        np.save(f'{self.path_downscaling}/y_pred_4.npy', yinv)
        print(f'Downscaling from 25 to 4 km finished \
              predicted data saved to {self.path_downscaling}/y_pred_4.npy')

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
        x_25 = np.load(f'{self.path_downscaling}/x_25.npy')
        x_4 = np.load(f'{self.path_downscaling}/x_4.npy')
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
            if self.variable == "tmax" or self.variable == "tmin":
                cmap = 'Spectral_r'
            else:
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
            elif metric == "mean":
                if self.variable == "tmax" or self.variable == "tmin":
                    ll = np.ceil(np.min(y_pred_25))
                    ul = np.ceil(np.max(y_pred_25))
                else:
                    ll = 0
                    ul = 5
            

        fig, ax = plt.subplots(2, 3, figsize=(16,7))

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

        # plt.colorbar(mm1,ax=ax[0][0],shrink=0.2)
        # plt.colorbar(mm2,ax=ax[0][1],shrink=0.2)
        # plt.colorbar(mm3,ax=ax[0][2],shrink=0.2)
        # plt.colorbar(mm4,ax=ax[1][0],shrink=0.2)
        # plt.colorbar(mm5,ax=ax[1][1],shrink=0.2)
        # plt.colorbar(mm6,ax=ax[1][2],shrink=0.2)
        fig.subplots_adjust(left=0.02, right=.98, top=0.95, bottom=0.05, hspace=0.15, wspace=0.15)

        cbar = fig.colorbar(mm1, ax=ax, orientation='horizontal', fraction=0.03, pad=0.05)

       # ax[0].remove()
        plt.suptitle(f'GCM Downscaling using SRGAN ({metric})')
        # plt.tight_layout()
        plt.savefig(f'{self.path_fig}/gcm_{metric}.pdf')
        plt.close(fig)
    

    def plot_diff(self, y_gcm, y_pred_25, y_pred_4, metric='mean'):
        x_100 = np.load(f'{self.path_downscaling}/x_100.npy')
        x_25 = np.load(f'{self.path_downscaling}/x_25.npy')
        x_4 = np.load(f'{self.path_downscaling}/x_4.npy')
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
            x_25 = np.mean(x_25,axis=0)#*86400*1000
            x_4 = np.mean(x_4,axis=0)#*86400*1000
            y_gcm = np.mean(y_gcm,axis=0)#*86400*1000
            y_pred_25 = np.mean(y_pred_25,axis=0)#*86400*1000
            y_pred_4 = np.mean(y_pred_4,axis=0)#*86400*1000

        diff1  = y_gcm-x_100
        diff2  = y_pred_25-x_25
        diff3  = y_pred_4-x_4

        ul = max(np.ceil(np.max(diff1)),np.ceil(np.max(diff2)),np.ceil(np.max(diff3)))
        ll = -ul
        if var == 'pr' and metric == "25th":
            ul = 2
            ll = -2
        elif metric == "wet_days":
            ul = 20
            ll = -20

        if var == 'tmax' or var == 'tmin':
            camp = 'RdBu_r'
        else:
            camp = 'RdBu'

        fig, ax = plt.subplots(1, 3, figsize=(16,4))
        mm1= ax[0].imshow(diff1[::-1,:],vmin=ll,vmax=ul,cmap = camp)
        ax[0].set_title("100km")
        mm2 = ax[1].imshow(diff2[::-1,:],vmin=ll,vmax=ul,cmap = camp)
        ax[1].set_title("25km")
        mm3 = ax[2].imshow(diff3[::-1,:],vmin=ll,vmax=ul,cmap = camp)
        ax[2].set_title("4km")

        # Adjust spacing to make room for the colorbar at the bottom
        fig.subplots_adjust(left=0.02, right=.98, top=0.9, bottom=0.10, wspace=0.1)

        # Create a colorbar axis at the bottom and make it bigger
        cbar = fig.colorbar(mm1, ax=ax, orientation='horizontal', fraction=0.05)

        # fig.subplots_adjust(bottom=0.2)  # Adjusts the main plot area
        # # Create a colorbar axis at the bottom and make it bigger
        # cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]
        # plt.colorbar(mm1,cax=cbar_ax, orientation='horizontal')
        # plt.colorbar(mm2,ax=ax[1],shrink=0.2)
        # plt.colorbar(mm3,ax=ax[2],shrink=0.2)
       # ax[0].remove()
        plt.suptitle(f'Diff between GCM and Daymet ({metric})')
        # plt.tight_layout()
        plt.savefig(f'{self.path_fig}/diff_{metric}.pdf')
        plt.close(fig)
        print(f'fig saved for Diff between GCM and Daymet ({metric})')

    def plot_gcm(self):
        y_gcm = self.read_data(f'{self.path_downscaling}/y_gcm_100.npy')
        y_pred_25 = self.read_data(f'{self.path_downscaling}/y_pred_25.npy')
        y_pred_4 = self.read_data(f'{self.path_downscaling}/y_pred_4.npy')

        self.plot_map(y_gcm, y_pred_25, y_pred_4, metric='mean')
        self.plot_map(y_gcm, y_pred_25, y_pred_4, metric='95th')
        self.plot_map(y_gcm, y_pred_25, y_pred_4, metric='25th')
        
        self.plot_diff(y_gcm, y_pred_25, y_pred_4, metric='mean')
        self.plot_diff(y_gcm, y_pred_25, y_pred_4, metric='95th')
        self.plot_diff(y_gcm, y_pred_25, y_pred_4, metric='25th')
        if self.variable=='pr':
            self.plot_map(y_gcm, y_pred_25, y_pred_4, metric='wet_days')
            self.plot_diff(y_gcm, y_pred_25, y_pred_4, metric='wet_days')

class DaymetData:
    @staticmethod
    def DaymetCombine(var="prcp"):
        daymet_100 = read_Daymet_yearly(var, year_start=1980, year_end=2020,deg=1)
        np.save(f'./gcm_ds/daymet/Daymet_{var}_1980_2019_1deg.npy', daymet_100)
        daymet_25 = read_Daymet_yearly(var, year_start=1980, year_end=2020,deg=0.25)
        np.save(f'./gcm_ds/daymet/Daymet_{var}_1980_2019_0.25deg.npy', daymet_25)
        daymet_4 = read_Daymet_yearly(var, year_start=1980, year_end=2020,deg=0.0416)
        np.save(f'./gcm_ds/daymet/Daymet_{var}_1980_2019_0.0416deg.npy', daymet_4)
        print

if __name__ == '__main__':
    # var='pr'
    version = '0.9'
    # var='tmax'
    for var in ['pr', 'tmax', 'tmin']:
        if var == 'tmax':
            version_lr = '7.7'
            version_hr = '7.6'
        elif var == 'tmin':
            version_lr = '8.5'
            version_hr = '8.6'
        elif var == 'pr':
            version_lr = '2.6'
            version_hr = '5.4'
    
        #####
        # gcm = 'MPI-ESM1-2-HR'
        scenario = 'ssp245'
        # year = '198001-201912'
        year = '202001-205912'
        #####
        # gcms = ['EC-Earth3-CC', 'EC-Earth3-Veg', 'GFDL-CM4', 'MPI-ESM1-2-HR', 'TaiESM1']
        # gcms = ['EC-Earth3-CC', 'EC-Earth3-Veg', 'GFDL-CM4', 'TaiESM1']
        gcms = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'MRI-ESM2-0', 'NorESM2-MM']

        for gcm in gcms:
            if gcm == 'CNRM-ESM2-1':
                ens = 'r1i1p1f2'
                grid = 'gr'
            # else:
            #     ens = 'r1i1p1f1'
            #     grid = 'gn'
            elif 'EC-Earth3' in gcm:
                ens = 'r1i1p1f1'
                grid = 'gr'
            elif gcm == 'GFDL-CM4':
                ens = 'r1i1p1f1'
                grid = 'gr1'
            else:
                ens = 'r1i1p1f1'
                grid = 'gn'
            print(f'Downscaling {var} {scenario} {gcm} .......')

            path_gcm = f'/mnt/data/ClimateSR/gcm/{var}_day_{gcm}_{scenario}_{ens}_{grid}_{year}_1deg_Daymet_ERA5_VICa_BC.nc'
            # path_gcm = f'/mnt/data/ClimateSR/gcm/{var}_day_MPI-ESM1-2-HR_ssp245_r1i1p1f1_gn_202001-205912_1deg_Daymet_ERA5_VICa_BC.nc'

            # #only run this line if the first time
            # DaymetData.DaymetCombine(var=var)

            GCM_downscaling = GCM_Downscaling(version_lr = version_lr,
                                            version_hr = version_hr,
                                            downscale_version = version,
                                            elevation = True,
                                            variable = var, 
                                            path_gcm = path_gcm)
            
            GCM_downscaling.downscale_100_25()
            GCM_downscaling.downscale_25_4()

            # GCM_visual = GCM_Visual(downscale_version = version, 
            #                         variable = var)
            # GCM_visual.plot_gcm()
