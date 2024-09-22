'''
test the historical data with future data
'''
import os
import numpy as np
from netCDF4 import Dataset

def main(year1, year2):
    path_data_1 = f'{path_tmax1}/tmax_ACCESS-CM2_ssp585_BC0.0416deg_predict_daily_SRGAN_{year1}_dims.nc'
    path_data_2 = f'{path_tmax2}/tmax_ACCESS-CM2_ssp585_BC0.0416deg_predict_daily_SRGAN_{year2}_dims.nc'

    fil1  = Dataset(path_data_1)
    fil2  = Dataset(path_data_2)

    hr_var1 = fil1.variables['tmax'][:]
    hr_var2 = fil2.variables['tmax'][:]
    
    # calculate the difference between the two years and see if they are the same
    diff = hr_var2 - hr_var1
    print(f'year {year2} - {year1}: {np.max(diff)}')


    return

def main2():
    path_tmax1 = '/mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.6/tmax/BCC-CSM2-MR'
    path_tmax2 = '/mnt/data/home/7hn/project/SRGAN_torch/gcm_ds/0.7/tmax/BCC-CSM2-MR'

    # read npy file and calculate the difference
    hr_var1 = np.load(f'{path_tmax1}/y_pred_25.npy')
    hr_var2 = np.load(f'{path_tmax2}/y_pred_25.npy')
    diff = hr_var2 - hr_var1
    print(f'year 2020 - 1980: {np.max(diff)}')

    return

def main3():
    gcms = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CNRM-ESM2-1', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    var='tmax'
    scenario = 'ssp585'
    for gcm in gcms:
            if gcm == 'CNRM-ESM2-1':
                ens = 'r1i1p1f2'
                grid = 'gr'
            else:
                ens = 'r1i1p1f1'
                grid = 'gn'

            path_data_1 = f'/mnt/data/ClimateSR/gcm/{var}_day_{gcm}_{scenario}_{ens}_{grid}_202001-205912_1deg_Daymet_ERA5_VICa_BC_dims.nc'
            path_data_2 = f'/mnt/data/ClimateSR/gcm/{var}_day_{gcm}_{scenario}_{ens}_{grid}_198001-201912_1deg_Daymet_ERA5_VICa_BC.nc'

            fil1  = Dataset(path_data_1)
            fil2  = Dataset(path_data_2)

            hr_var1 = fil1.variables['tmax'][:]
            hr_var2 = fil2.variables['tmax'][:]
            
            # calculate the difference between the two years and see if they are the same
            diff = hr_var2 - hr_var1
            print(f'{gcm}: {np.max(diff)}')


    return

if __name__ == "__main__":

    # path_tmax1 = '/lustre/orion/proj-shared/cli138/7hn/BC/ssp585/tmax/ACCESS-CM2'
    # path_tmax2 = '/lustre/orion/proj-shared/cli138/7hn/BC/ssp585_1/tmax/ACCESS-CM2'

    # year1 = 1980
    # year2 = 2020
    # for year1 in range(1980, 2020):
    #     year2 = year1 + 40
    #     main(year1, year2)

    main2()
    # main3()

