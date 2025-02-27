import os
import xarray as xr
import s3fs
import tempfile
import numpy as np
from datetime import datetime, timedelta
import pickle
import pytz

def get_utc_intervals_and_hours(date_str):
    # Define Houston and UTC timezones
    houston_tz = pytz.timezone('America/Chicago')
    utc_tz = pytz.utc

    # Parse input date (format: yyyymmdd)
    date_obj = datetime.strptime(date_str, "%Y%m%d")

    # Set Houston time to 06:00 AM on the given date
    houston_time = houston_tz.localize(datetime(date_obj.year, date_obj.month, date_obj.day, 7, 0))

    # Convert to UTC
    utc_time = houston_time.astimezone(utc_tz)
    print(f'utc_time: {utc_time}')

    # Get UTC hour and find the nearest 6-hour interval
    utc_hour = utc_time.hour
    nearest_six_hour_interval = utc_hour - (utc_hour % 6)

    # Calculate hours until Houston midnight (00:00 AM next day, which is 06:00 UTC next day)
    next_day = date_obj + timedelta(days=1)
    next_day_houston_midnight = houston_tz.localize(datetime(next_day.year, next_day.month, next_day.day, 0, 0))

    #next_day_houston_midnight = houston_tz.localize(datetime(date_obj.year, date_obj.month, date_obj.day + 1, 0, 0))
    next_day_utc_time = next_day_houston_midnight.astimezone(utc_tz)

    hours_until_houston_midnight = int((next_day_utc_time - utc_time).total_seconds() // 3600) + (utc_hour - nearest_six_hour_interval)

    return nearest_six_hour_interval, hours_until_houston_midnight

def get_date_one_days_later(date_str):
    # Convert string to datetime object
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    
    # Add two days
    future_date_obj = date_obj + timedelta(days=1)
    
    # Convert back to string in YYYYMMDD format
    future_date_str = future_date_obj.strftime("%Y%m%d")
    
    return future_date_str

def get_hourly_forecast(on,nearest_utc,t):
    

    url = f"noaa-gfs-bdp-pds/gfs.{on}/{nearest_utc}/atmos/gfs.t{nearest_utc}z.pgrb2.0p25.f{t:03}"
    
    fs = s3fs.S3FileSystem(anon=True)
    
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp_file:
        local_path = tmp_file.name
    
    try:
        # 4) Download from S3 to the local temporary path
        fs.get(url, local_path)

        level_variable = [{'key':{'stepType': 'accum', 
                            'typeOfLevel': 'surface'},
                        'variable':['tp'], #'Total Precipitation'
                    },
                    {'key':{'stepType': 'instant', 
                            'typeOfLevel': 'surface'},
                        'variable':['SUNSD','t'], #'Sunshine Duration','Temperature'
                    },
                    {'key':{'typeOfLevel': 'isobaricInhPa'},
                        'variable':['q','t','u','v','w'], #'Specific humidity','Temperature','U component of wind','V component of wind','Vertical velocity'
                        'level':[925, 1000]
                    },
                    {'key':{'typeOfLevel': 'lowCloudLayer'},
                        'variable':['avg_lcc'], #'Time-mean low cloud cover'
                    },
                    {'key':{'typeOfLevel': 'middleCloudLayer'},
                        'variable':['avg_mcc'], #'Time-mean middle cloud cover'
                    },
                    {'key':{'typeOfLevel': 'highCloudLayer'},
                        'variable':['avg_hcc'], #'Time-mean high cloud cover'
                    },
                    ]

        lat_bounds = (28.69, 30.62)   # (min_lat, max_lat)
        lon_bounds = (-96.63, -94.35)  # (min_lon, max_lon)

        hourly_array = np.zeros((16,8,9))
        variable_list = []
        idx = 0
        
        for config in level_variable:
            
            ds = xr.open_dataset(local_path,
                                engine="cfgrib",
                                errors='ignore',
                                backend_kwargs={
                                "filter_by_keys": config['key']
                                            })
        
            lat0, lat1 = sorted(lat_bounds)
            lon0, lon1 = sorted(lon_bounds)
        
            lon0 = (lon0 % 360 + 360) % 360
            lon1 = (lon1 % 360 + 360) % 360
            # print(ds)
            # print(lat0, lat1)
            # print(lon0, lon1)
            
            ds = ds.sel(latitude=slice(lat1, lat0), longitude=slice(lon0, lon1))
            # print(ds_sub)
            # raise ValueError
            # for var_name in ds.data_vars:
            #     print(var_name)
            #     print(ds[var_name].attrs['long_name'])
            #     print('---')
                
            if 'level' in config:
                for level in config['level']:
                    ds_level = ds.sel(isobaricInhPa=level)
        
                    for variable in config['variable']:
                    # print(variable,level)
                        variable_list.append(f'{variable}_{level}')
                        
                        variable_data = ds_level[variable].values
                        hourly_array[idx,:,:] = variable_data
                        #print(variable_data.shape)
                        idx += 1
            else:
                for variable in config['variable']:
                    variable_list.append(f'{variable}')
                    #print(variable)
                    variable_data = ds[variable].values
                    hourly_array[idx,:,:] = variable_data
                    #print(variable_data.shape)
                    idx += 1
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(local_path):
            os.remove(local_path)

    return hourly_array, variable_list

def generate_date_list(start_date, end_date):
    """
    Generate a list of dates in 'YYYYMMDD' format between the given start and end dates.

    :param start_date: The starting date in 'YYYYMMDD' format (e.g., '20230112')
    :param end_date: The ending date in 'YYYYMMDD' format (e.g., '20230115')
    :return: List of dates in 'YYYYMMDD' format
    """
    # Convert start_date and end_date strings to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y%m%d")
    end_date_obj = datetime.strptime(end_date, "%Y%m%d")

    # Generate list of dates between start and end date (inclusive)
    date_list = []
    current_date = start_date_obj
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    return date_list

if __name__ == '__main__':
    
    date_list = generate_date_list('20230131','20230508')

    for on in date_list:
        #get the utc need to be used and the hours to houston midnight
        nearest_utc, hours_diff = get_utc_intervals_and_hours(on)

        forecast = get_date_one_days_later(on)

        print(f'forcast houston time: {forecast} on {on}')
        print(f'nearest utc: {nearest_utc}')
        print(f'hours to houston midnight: {hours_diff}')


        time_list = [hours_diff+3*i for i in range(8)]

        daily_array = np.zeros((len(time_list),16,8,9))
        
        for idx,t in enumerate(time_list):
            print(t)
        
            hourly_array, variable_list = get_hourly_forecast(on,nearest_utc,t)
        
            daily_array[idx,:,:,:] = hourly_array

        final_data = {'variables':variable_list,
                    'hour_forecast':time_list,
                    'prediction_date':on,
                    'data':daily_array
                    }

        with open(f'/Users/leroy/Documents/GitHub/Electricity_trading/DART/Data/GFS_forecast/forecast_{forecast}_on_{on}.pkl', 'wb') as file:
            pickle.dump(final_data, file)