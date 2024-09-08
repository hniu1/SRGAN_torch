import numpy as np
import os
from netCDF4 import Dataset
import cv2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scalers import Log1pScaler
#import keras
#from keras.models import Sequential
import datetime
import pickle

def read_Daymet(var, deg=1):
    fil        = Dataset(f'/mnt/data/ClimateSR/daymet4x/DaymetV4_VIC4_prcp_1980-2022_{deg}deg_SEtest.nc')
    # fil        = Dataset(f'/mnt/data/ClimateSR/daymet4x/DaymetV4_VIC4_prcp_2011-2022_{deg}deg_SEtest.nc')

    hr_var     = fil.variables[f'{var}'][:]
    # time       = fil.variables['time'][0:]
    # time3d     = time[:,np.newaxis,np.newaxis]
    # broadcasted_time = np.broadcast_to(time3d, (np.shape(hr_var)[0], np.shape(hr_var)[1], np.shape(hr_var)[2]))
    return hr_var

def read_Daymet_yearly(var, year_start, year_end, deg=1, Daymet_ERA5=False):
    arrays = []
    for year in range(year_start, year_end): #2023
        print(f'read {deg} degree data from year {year}')
        if Daymet_ERA5:
            fil        = Dataset(f'/mnt/data/ClimateSR/newgrid/Daymet_ERA5_VIC4a_prcp_{year}_{deg}deg.nc')
            # fil        = Dataset(f'/mnt/data/ClimateSR/daymet_ERA/Daymet_ERA5_VIC4a_prcp_{year}_{deg}deg.nc')
            # fil        = Dataset(f'/lustre/orion/cli138/proj-shared/7hn/data/Daymet-ERA5/DaymetV4-ERA5_VIC4_prcp_{year}_{deg}deg_US.nc')
        else:
            fil        = Dataset(f'/mnt/data/ClimateSR/newgrid/DaymetV4_VIC4_prcp_{year}_{deg}deg_US.nc') # newgrid data in sunsphere
            # fil        = Dataset(f'/mnt/data/ClimateSR/data-for-haoran/US/DaymetV4_VIC4_prcp_{year}_{deg}deg_US.nc') # old grid data in sunsphere
            # fil        = Dataset(f'/lustre/orion/cli138/proj-shared/7hn/data/Daymet/DaymetV4_VIC4_prcp_{year}_{deg}deg_US.nc') # data in andes

        hr_var     = fil.variables[f'{var}'][:]
        arrays.append(np.array(hr_var))
    data = np.concatenate(arrays, axis=0)
    data[np.isnan(data)] = 0
    data[data < 0] = 0
    data = data.astype('float32')
    return data

def read_elev(tt, deg=1):
    # felev      = Dataset(f'/mnt/data/ClimateSR/newgrid/daymet_ERA/VIC4a_DEM_{deg}deg.nc')
    felev      = Dataset(f'/mnt/data/ClimateSR/daymet_ERA/VIC4a_DEM_{deg}deg.nc')
    elev1      = felev.variables["DEM"]
    elev       = np.tile(elev1,(tt,1,1))
    elev = elev.astype('float32')
    return elev

def minmaxscaler(lr):
    tt = np.shape(lr)[0]
    nx = np.shape(lr)[1]
    ny = np.shape(lr)[2]
    lr       = np.reshape(lr,(tt,nx*ny*1))
    scaler   = MinMaxScaler()
    lr       = scaler.fit_transform(lr)
    lr       = np.reshape(lr,(tt,nx,ny,1))
    return lr

def low_resolution(hr):
    tt = np.shape(hr)[0]
    lr=np.zeros((tt,60,60))
    for i in range(hr.shape[0]):
        lr[i] = cv2.resize(hr[i], dsize=(60, 60), interpolation=cv2.INTER_CUBIC)
    lr = np.reshape(lr,(tt,60,60,1))
    return lr

def split(lr,hr):
    X_train, X_test, y_train, y_test = train_test_split(lr, hr, test_size=0.2, random_state=42)
    return(X_train, X_test, y_train, y_test)

def invtrans_write(y,scalar,name,path_output,elevation=False):
    shp    = np.shape(y)
    tt     = shp[0]
    nhr1   = shp[1]
    nhr2   = shp[2]
    if elevation:
        y = y[:, :, :, 0:1]
        # elev_scale = y[:, :, :, 1:2]
    y      = y.flatten()
    yinv   = scalar.inverse_transform(y.reshape(-1, 1))
    yinv[yinv<0] = 0
    yinv   = np.reshape(yinv,(tt,nhr1,nhr2))
    np.save(f'{path_output}/{name}.npy',yinv)
    # if elev:
    #     elev_scale      = elev_scale.flatten()
    #     elev_scale_inv   = scalar.inverse_transform(elev_scale.reshape(-1, 1))
    #     elev_scale_inv   = np.reshape(elev_scale_inv,(tt,nhr1,nhr2))

def daymetread(path_output, checkpoint_dir, elevation = False, elevation_hr=False, Daymet_ERA5=False, high_deg=False, scaler = 'standard'):

# Read variables nd generate low resolution version
    deg_hr = 0.25
    deg_lr = 1
    if high_deg:
        deg_hr = 0.0416
        deg_lr = 0.25

    # lr_prect = read_Daymet("prcp", deg=1)
    # hr_prect = read_Daymet("prcp", deg=0.25)
    if not Daymet_ERA5:
        lr_prect = read_Daymet_yearly("prcp", year_start=2003, year_end=2023,deg=deg_lr)
        hr_prect = read_Daymet_yearly("prcp", year_start=2003, year_end=2023, deg=deg_hr)
    else:
        lr_prect = read_Daymet_yearly("prcp", year_start=1990, year_end=2020, deg=deg_lr, Daymet_ERA5=Daymet_ERA5)
        hr_prect = read_Daymet_yearly("prcp", year_start=1990, year_end=2020, deg=deg_hr, Daymet_ERA5=Daymet_ERA5)
    # time = np.reshape(time,(tt,nhr1,nhr2,1))
    print(f'hr shape: {np.shape(hr_prect)}')
    print(f'lr shape: {np.shape(lr_prect)}')

    nhr1 = hr_prect.shape[1]
    nhr2 = hr_prect.shape[2]
    nlr1 = lr_prect.shape[1]
    nlr2 = lr_prect.shape[2]
    tt = hr_prect.shape[0]
# Scale high-resolution precipitation ("y")
    if scaler == 'standard':
        scaler_hrprect  = StandardScaler()
    elif scaler == 'robust':
        scaler_hrprect  = RobustScaler()
    elif scaler == 'minmax':
        scaler_hrprect  = MinMaxScaler()
    elif scaler == 'log':
        scaler_hrprect  = Log1pScaler()
    else:
        scaler_hrprect  = StandardScaler()

    hr_prect        = hr_prect.flatten()
    lr_prect        = lr_prect.flatten()
    combined_images = np.concatenate([hr_prect, lr_prect], axis=0)
    scaler_hrprect.fit(combined_images.reshape(-1,1))

    hr_prect_scaled = scaler_hrprect.transform(hr_prect.reshape(-1,1))
    hr_prect_scaled = np.reshape(hr_prect_scaled,(tt,nhr1,nhr2,1))

    print(f'lr size {np.shape(lr_prect)}')
    lr_prect_scaled = scaler_hrprect.transform(lr_prect.reshape(-1, 1))
    lr_prect_scaled = np.reshape(lr_prect_scaled,(tt,nlr1,nlr2,1))

    hr = hr_prect_scaled[:,:,:,:]
    lr = lr_prect_scaled[:,:,:,:]
    
    if elevation:
        scaler_elev  = MinMaxScaler()
        elev = read_elev(tt, deg=deg_lr)
        scaler_elev.fit(elev.flatten().reshape(-1, 1))
        elev_scaled = scaler_elev.transform(elev.flatten().reshape(-1, 1))
        elev_scaled = np.reshape(elev_scaled,(tt,nlr1,nlr2,1))
        np.save(f'{path_output}/elev_lr_scaled.npy',elev_scaled)
        lr = np.concatenate((lr_prect_scaled,elev_scaled),axis=3)
        if elevation_hr:
            elev_hr = read_elev(tt, deg=deg_hr)
            elev_hr_scaled = scaler_elev.transform(elev_hr.flatten().reshape(-1, 1))
            elev_hr_scaled = np.reshape(elev_hr_scaled,(tt,nhr1,nhr2,1))
            np.save(f'{path_output}/elev_hr_scaled.npy',elev_hr_scaled)
            hr = np.concatenate((hr_prect_scaled,elev_hr_scaled),axis=3)

    X_train, X_test, y_train, y_test = split(lr,hr)

    # Save the scaler
    with open(f'./{checkpoint_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler_hrprect, f)

    invtrans_write(X_train,scaler_hrprect,"x_train",path_output, elevation)
    invtrans_write(X_test,scaler_hrprect,"x_test",path_output, elevation)
    invtrans_write(y_train,scaler_hrprect,"y_train",path_output, elevation_hr)
    invtrans_write(y_test,scaler_hrprect,"y_test",path_output, elevation_hr)

    train_lr = [X_train[i] for i in range(X_train.shape[0])]
    test_lr = [X_test[i] for i in range(X_test.shape[0])]
    train_hr = [y_train[i] for i in range(y_train.shape[0])]
    test_hr = [y_test[i] for i in range(y_test.shape[0])]

    return train_lr, test_lr, train_hr, test_hr
