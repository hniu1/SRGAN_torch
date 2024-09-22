'''
The demo is used for calculate and plot the differences between the models:
1. dbcca
2. sup3rcc
3. srcnn
4. srgan
'''

import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import glob

def read_nc_yearly(var, year_start, year_end, deg, path_model):
    arrays = []
    for year in range(year_start, year_end): #2023
        print(f'read {deg} degree data from year {year} {path_model}')
        if path_model == path_dbcca:
            if var == 'pr':
                path_data = f'{path_model}/prcp/MPI-ESM1-2-HR_ssp245_r1i1p1f1_DBCCA_Daymet_VIC4_prcp_{year}.nc'
            else:
                path_data = f'{path_model}/{var}/MPI-ESM1-2-HR_ssp245_r1i1p1f1_DBCCA_Daymet_VIC4_{var}_{year}.nc'
        elif path_model == path_sup3rcc:
            pattern = os.path.join(
                                    path_model,
                                    f"sup3rcc_conus_mpiesm12hr_*_r1i1p1f1_{var}*_{year}.nc"
                                )
            matching_files = glob.glob(pattern)
            if matching_files:
                path_data = matching_files[0]  # Select the first matching file
                print(f"Processing file: {path_data}")
                # Add your file processing logic here
            else:
                print(f"No file found for year {year} and variable {var} in {path_model}")
        elif path_model == path_srcnn:
            path_data = f'{path_model}/{var}/{var}_daily_MPI-ESM1-2-HR_ssp245_DBCSRCNN_0.0416deg_{year}.nc'
        # elif path_model == path_srgan:
        else:
            if var == 'pr':
                path_data = f'{path_model}/{var}/{var}_MPI-ESM1-2-HR_ssp245_DBCSRGAN_0.0416deg_predict_daily_SRGAN_{year}.nc'
            else:
                path_data = f'{path_model}/tmax_tmin/{var}_MPI-ESM1-2-HR_ssp245_DBCSRGAN_0.0416deg_predict_daily_SRGAN_{year}.nc'
        
        fil  = Dataset(path_data)
        hr_var     = fil.variables[f'{var}'][:]
        arrays.append(np.array(hr_var))
    data = np.concatenate(arrays, axis=0)
    data[np.isnan(data)] = 0
    # if var == 'pr' or var == 'prcp':
    #     data[data < 0] = 0
    data = data.astype('float32')
    return data

def plot_map(var,X_test,y_test,y_test_predict,y_test_predict_init,metric='mean'):
    if metric == "95th":
        y0 = np.percentile(X_test, 95, axis=0)
        y1 = np.percentile(y_test, 95, axis=0)
        y2 = np.percentile(y_test_predict_init, 95, axis=0)
        y3 = np.percentile(y_test_predict, 95, axis=0)
    elif metric == "25th":
        y0 = np.percentile(X_test, 25, axis=0)
        y1 = np.percentile(y_test, 25, axis=0)
        y2 = np.percentile(y_test_predict_init, 25, axis=0)
        y3 = np.percentile(y_test_predict, 25, axis=0)
    else:
        y0 = np.mean(X_test,axis=0)
        y1 = np.mean(y_test,axis=0)
        y2 = np.mean(y_test_predict_init,axis = 0)
        y3 = np.mean(y_test_predict,axis = 0)

    #print(np.min(y2))
    #print(np.max(y2))
    fig, ax = plt.subplots(2, 2, figsize=(12,8))
    ll = -1
    ul = 1
    if (var == "tmax") or (var == "tmin"):
        cmap = 'Spectral_r'
        ll = np.floor(np.min(y1))
        ul = np.ceil(np.max(y1))
    else:
        # cmap = 'GnBu'
        cmap = 'Spectral'
        if metric == "95th":
            ll = 0
            ul = 25
        elif metric == "25th":
            ll = 0
            ul = np.ceil(np.max(y1))
        else:
            ll = 0
            ul = 5
    mm1= ax[0][0].imshow(y0[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
    ax[0][0].set_title("DBCCA")
    mm2 = ax[0][1].imshow(y1[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
    ax[0][1].set_title("Sup3rcc")
    mm3 = ax[1][0].imshow(y2[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
    ax[1][0].set_title("SRCNN")
    mm4 = ax[1][1].imshow(y3[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
    ax[1][1].set_title("SRGAN")
    # plt.colorbar(mm1,ax=ax[0][0],shrink=0.2)
    # plt.colorbar(mm2,ax=ax[0][1],shrink=0.2)
    # plt.colorbar(mm3,ax=ax[1][0],shrink=0.2)
    # plt.colorbar(mm4,ax=ax[1][1],shrink=0.2)
     # Adjust spacing to make room for the colorbar at the bottom
    fig.subplots_adjust(left=0.02, right=.98, top=0.9, bottom=0.10, wspace=0.1)
    # Create a colorbar axis at the bottom and make it bigger
    fig.colorbar(mm1, ax=ax, orientation='horizontal', fraction=0.03)
    # ax[0].remove()
    plt.savefig(f'{path_fig}/spatialmaps_{years}_{var}_{metric}.pdf')
    # plt.tight_layout()
    plt.close(fig)

def main():
    y_dbcca = read_nc_yearly('prcp' if var=='pr' else var, year_start=1980, year_end=2020, deg=0.0416, path_model=path_dbcca)
    y_sup3rcc = read_nc_yearly(var, year_start=1980, year_end=2020, deg=0.0416, path_model=path_sup3rcc)
    y_srcnn = read_nc_yearly(var, year_start=1980, year_end=2020, deg=0.0416, path_model=path_srcnn)
    y_srgan = read_nc_yearly(var, year_start=1980, year_end=2020, deg=0.0416, path_model=path_srgan)

    plot_map(var,y_dbcca,y_sup3rcc,y_srcnn,y_srgan)
    plot_map(var,y_dbcca,y_sup3rcc,y_srcnn,y_srgan, '95th')
    plot_map(var,y_dbcca,y_sup3rcc,y_srcnn,y_srgan, '25th')


    return


if __name__ == "__main__":

    var = "pr"
    gcm = 'MPI-ESM1-2-HR'
    scenario = 'ssp585'
    years = '198001-201912'
    ens = 'r1i1p1f2'
    grid = 'gr'

    path_dbcca = '/mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/ssp245/DBCCA'
    path_sup3rcc = '/mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/ssp245/sup3rcc'
    path_srcnn = '/mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/ssp245/srcnn'
    path_srgan = '/mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/ssp245/srgan'


    path_workspace = f'/mnt/data/home/7hn/project/SRGAN_torch/downscaled_analysis'
    path_fig = f'{path_workspace}/fig'
    os.makedirs(path_fig, exist_ok=True)

    main()


    print