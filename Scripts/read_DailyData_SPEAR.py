"""
Read in daily data from SPEAR and create netCDF4 files from xarray
 
Author    : Zachary M. Labe
Date      : 20 July 2023
"""

import calc_Utilities as UT
import xarray as xr
import numpy as np
import cmocean
import os, glob
import calc_Stats as dSS
import sys

import time
startTime = time.time()

### Slurm commands
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--vari', help='variable name for processed data')
parser.add_argument('--variq', help='variable name for raw SPEAR data')
parser.add_argument('--reg_name', help='region of analysis')
parser.add_argument('--model', help='model to read daily data from')
args=parser.parse_args()

### Parameters
vari = args.vari
variq = args.variq
reg_name = args.reg_name
model = args.model

# vari = 'q'
# variq = 'q_ref'
# reg_name = 'US'
# model = 'SPEAR_MED'

print('ARGUMENTS ARE ---> %s, %s, %s, %s!' % (vari,variq,reg_name,model))

resolution = 'MEDS' 
junedays = np.arange(0,30,1)
julydays = np.arange(0,31,1)
augustdays = np.arange(0,31,1)
dayslength = len(junedays) + len(julydays) + len(augustdays)
lat_bounds,lon_bounds = UT.regions(reg_name)

if model == 'SPEAR_MED':
    years = np.arange(1921,2100+1)
    ENS = 30
elif model == 'SPEAR_MED_SSP534OS':
    years = np.arange(2011,2100+1)
    ENS = 30
elif model == 'SPEAR_MED_SSP245':
    years = np.arange(2011,2100+1)
    ENS = 30
elif model == 'SPEAR_MED_SSP534OS_10ye':
    years = np.arange(2031,2100+1)
    ENS = 9
elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv':
    years = np.arange(2041,2100+1)
    ENS = 9
elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv':
    years = np.arange(2041,2100+1)
    ENS = 9
elif model == 'SPEAR_MED_LM42p2_test':
    years = np.arange(1921,2070+1)
    ENS = 3

### Read in data files from server
directorydata = '/work/Zachary.Labe/Data/SPEAR/%s/daily/' % model
dataoutput = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'

for e in range(ENS):
    glob_pattern = os.path.join(directorydata, '%s/raw/raw_%s/atmos_daily.*.%s.nc' % (vari,e+1,variq))
    print('<<<<<STARTED: Read in ---> raw_%s/atmos_daily.*.%s.nc' % (e+1,variq))
    
    ### Read in data
    ds = xr.open_mfdataset(glob_pattern,concat_dim="time",combine="nested",
                            data_vars='minimal',coords='minimal',compat='override',
                            parallel=True,chunks={'time':'6GB','latitude':1,'longitude':1},
                            engine='netcdf4')
    # ds = xr.open_mfdataset(glob_pattern,concat_dim="time",combine="nested",
    #                         data_vars='minimal',coords='minimal',compat='override',
    #                         parallel=True,engine='netcdf4')
    lat1 = ds['lat'].to_numpy()
    lon1 = ds['lon'].to_numpy()
    
    if reg_name != 'Globe':
        us = ds.sel(lat=slice(lat_bounds[0],lat_bounds[1]),lon=slice(lon_bounds[0],lon_bounds[1]))
        latus1 = us['lat'].to_numpy()
        lonus1 = us['lon'].to_numpy()
    else:
        us = ds
        latus1 = us['lat'].to_numpy()
        lonus1 = us['lon'].to_numpy()
        
    ### Calculate months
    jun = us.where(((ds['time.month'] == 6)),drop=True)
    jul = us.where(((ds['time.month'] == 7)),drop=True)
    aug = us.where(((ds['time.month'] == 8)),drop=True)
    
    jun.to_netcdf(dataoutput + 'June/June_%s_%s_%s_ens%s.nc' % (vari,reg_name,model,str(e+1).zfill(2)))
    jul.to_netcdf(dataoutput + 'July/July_%s_%s_%s_ens%s.nc' % (vari,reg_name,model,str(e+1).zfill(2)))
    aug.to_netcdf(dataoutput + 'August/August_%s_%s_%s_ens%s.nc' % (vari,reg_name,model,str(e+1).zfill(2)))
    
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
