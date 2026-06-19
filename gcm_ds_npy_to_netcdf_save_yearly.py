import numpy as np
import os
from netCDF4 import Dataset
# import cv2
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# from sklearn.model_selection import train_test_split
#import keras
#from keras.models import Sequential
# import datetime
import pickle
import xarray as xr
from tqdm import tqdm

DAYMET_DATA_DIR = '/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data'


def _infer_trim_grid_shape(var, year):
    candidates = []
    if var == 'pr':
        candidates.extend([
            ('prcp', f'{DAYMET_DATA_DIR}/Daymet_ERA5_prcp_dy_{year}_trim.nc'),
            ('prcp', f'{DAYMET_DATA_DIR}/Daymet_ERA5_prcp_dy_{year}$_trim.nc'),
        ])
    candidates.append((var, f'{DAYMET_DATA_DIR}/Daymet_ERA5_{var}_dy_{year}_trim.nc'))

    for file_var, file_path in candidates:
        if not os.path.exists(file_path):
            continue
        with Dataset(file_path) as ds:
            var_name = f'{file_var}_dy'
            if var_name in ds.variables:
                return ds.variables[var_name].shape[1:]

    raise FileNotFoundError(
        f'Could not infer 0.0416deg grid shape for {var} {year} from files in {DAYMET_DATA_DIR}.'
    )


def _load_downscaled_array(data_path, var, year_start):
    try:
        loaded = np.load(data_path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.dtype == object:
            if loaded.shape == ():
                loaded = loaded.item()
            elif loaded.size == 1:
                loaded = loaded.flat[0]

        if not isinstance(loaded, np.ndarray):
            raise TypeError(f'Loaded object from {data_path} is type {type(loaded)}, expected NumPy array.')

        return loaded
    except (ValueError, pickle.UnpicklingError):
        grid_y, grid_x = _infer_trim_grid_shape(var, year_start)
        bytes_per_val = np.dtype(np.float32).itemsize
        file_size = os.path.getsize(data_path)
        total_vals, remainder = divmod(file_size, bytes_per_val)

        if remainder != 0:
            raise ValueError(
                f'Raw file {data_path} has {file_size} bytes, not divisible by float32 size ({bytes_per_val}).'
            )

        vals_per_day = grid_y * grid_x
        num_days, day_remainder = divmod(total_vals, vals_per_day)
        if day_remainder != 0:
            raise ValueError(
                f'Raw file {data_path} has {total_vals} float32 values, not divisible by grid size {grid_y}x{grid_x}.'
            )

        print(
            f'[INFO] Loading raw float32 memmap file: {data_path} -> shape ({num_days}, {grid_y}, {grid_x})',
            flush=True,
        )
        return np.memmap(data_path, dtype=np.float32, mode='r', shape=(num_days, grid_y, grid_x))


def convert_yearly(var, year_start, year_end, deg=1):
    downscaled_data = _load_downscaled_array(data_path, var, year_start)

    current_day_index = 0
    for year in tqdm(range(year_start, year_end)): #2023
        print(f'read {deg} degree data from year {year}')
        fil = Dataset(f'/lustre/orion/proj-shared/cli138/dr6/CMIP6-Downscaled-VIC/GCM/ACCESS-CM2_ssp245_r1i1p1f1/pr/pr_day_ACCESS-CM2_ssp245_r1i1p1f1_gn_1deg_{year}.nc')
        hr_var     = fil.variables[f'pr'][:]
        num_days = hr_var.shape[0]

        # Extract the corresponding days' data from the downscaled data
        yearly_data = downscaled_data[current_day_index:current_day_index + num_days]
        yearly_data = np.expand_dims(yearly_data, axis=-1)  # Adds a new dimension at the end

        # resize this to four dimensions

        current_day_index += num_days

        output_file = os.path.join(output_dir, f'{var}_{gcm}_{scenario}_0.0416deg_predict_daily_SRGAN_{year}.nc')

        # Create an xarray Dataset from the NumPy array
        data = xr.DataArray(yearly_data)

        # Save the xarray Dataset to a NetCDF file
        data.to_netcdf(output_file)

    if current_day_index > downscaled_data.shape[0]:
        raise ValueError(
            f'Requested {current_day_index} daily steps but loaded data only has {downscaled_data.shape[0]}.'
        )


if __name__ == "__main__":
    #####
    scenario = 'ssp585'
    # year = '198001-201912'
    # var = 'tmax' , 'tmax', 'tmin'
    #####
    gcms = [
        'ACCESS-CM2',
        'BCC-CSM2-MR',
        'CNRM-ESM2-1',
        'EC-Earth3-CC',
        'EC-Earth3-Veg',
        'GFDL-CM4',
        'MRI-ESM2-0',
        'MPI-ESM1-2-HR',
        'NorESM2-MM',
        'TaiESM1',
    ]
    for var in ['pr', 'tmax', 'tmin'][:1]:
        for gcm in gcms:
            print(f'start with {gcm} {var}')

            version = '0.1'
            year_start = 1980
            year_end = 2020
            data_path = f'/lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr/gcm_ds/{version}/{var}/{gcm}/y_pred_4.npy'
            output_dir = f'/lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr/gcm_ds/{scenario}_yearly_nc_before_BC_0416deg_dims/{var}/{gcm}/'
            os.makedirs(output_dir, exist_ok=True)
            convert_yearly(var, year_start, year_end, deg='0416')

            version = '0.2'
            year_start = 2020
            year_end = 2060
            data_path = f'/lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr/gcm_ds/{version}/{var}/{gcm}/y_pred_4.npy'
            output_dir = f'/lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr/gcm_ds/{scenario}_yearly_nc_before_BC_0416deg_dims/{var}/{gcm}/'
            os.makedirs(output_dir, exist_ok=True)
            convert_yearly(var, year_start, year_end, deg='0416')
