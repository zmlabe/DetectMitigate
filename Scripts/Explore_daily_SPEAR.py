"""
Practice reading in daily data from SPEAR
 
Author    : Zachary M. Labe
Date      : 19 July 2023
"""

import matplotlib.pyplot as plt
import calc_Utilities as UT
import xarray as xr
import numpy as np
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean
import os, glob
import calc_Stats as dSS
import sys

import time
startTime = time.time()

### Parameters
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})
vari = 'T2M'
variq = 't_ref'
resolution = 'MEDS' 
junedays = np.arange(0,30,1)
julydays = np.arange(0,31,1)
augustdays = np.arange(0,31,1)
dayslength = len(junedays) + len(julydays) + len(augustdays)
reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)

model = 'SPEAR_MED_SSP534OS'
if model == 'SPEAR_MED':
    years = np.arange(1921,2100+1)
    ENS = 30
elif model == 'SPEAR_MED_SSP534OS':
    years = np.arange(2011,2100+1)
    ENS = 30
elif model == 'SPEAR_MED_SSP534OS_10ye':
    years = np.arange(2031,2100+1)
    ENS = 30
elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv':
    years = np.arange(2041,2100+1)
    ENS = 9

### Read in data files from server
directorydata = '/work/Zachary.Labe/Data/SPEAR/%s/daily/' % model
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'

daysall = np.empty((ENS,years.shape[0],dayslength,72,88))
for e in range(ENS):
    glob_pattern = os.path.join(directorydata, '%s/raw/raw_%s/atmos_daily.*.%s.nc' % (vari,e+1,variq))
    print('<<<<<STARTED: Read in ---> raw_%s/atmos_daily.*.%s.nc' % (e+1,variq))
    
    ### Read in data
    ds = xr.open_mfdataset(glob_pattern,concat_dim="time",combine="nested",
                           data_vars='minimal',coords='minimal',compat='override',
                           parallel=True,chunks={'time':'6GB','latitude':1,'longitude':1},
                           engine='netcdf4')
    lat1 = ds['lat'].to_numpy()
    lon1 = ds['lon'].to_numpy()
    
    us = ds.sel(lat=slice(lat_bounds[0],lat_bounds[1]),lon=slice(lon_bounds[0],lon_bounds[1]))
    latus1 = us['lat'].to_numpy()
    lonus1 = us['lon'].to_numpy()
    
    ### Calculate months
    jun = us.where(((ds['time.month'] == 6)),drop=True)
    jul = us.where(((ds['time.month'] == 7)),drop=True)
    aug = us.where(((ds['time.month'] == 8)),drop=True)
    
    juntas = jun['t_ref'].to_numpy().reshape(len(years),len(junedays),latus1.shape[0],lonus1.shape[0])
    jultas = jul['t_ref'].to_numpy().reshape(len(years),len(julydays),latus1.shape[0],lonus1.shape[0])
    augtas = aug['t_ref'].to_numpy().reshape(len(years),len(augustdays),latus1.shape[0],lonus1.shape[0])
    
    ### Convert units if needed
    if variq == 'T2M':
        daysall[e,:,:,:,:] = np.concatenate([juntas,jultas,augtas]) - 273.15 # K to C
        print('Completed: Changed units (K to C)!')
    else:
        daysall[e,:,:,:,:] = np.concatenate([juntas,jultas,augtas],axis=1)
        
    print('>>>>>COMPLETED: Read in ---> raw_%s/atmos_daily.*.%s.nc' % (e+1,variq))
    
### Meshgrid and mask by CONUS
lonus2,latus2 = np.meshgrid(lonus1,latus1)

data_obsnan = np.full([1,latus1.shape[0],lonus1.shape[0]],np.nan)
datamask,data_obsnan = dSS.mask_CONUS(daysall,data_obsnan,resolution,lat_bounds,lon_bounds)

### Calculate baseline
minb = 1981
maxb = 2010
baseline = np.arange(minb,maxb+1,1)
baseq = np.where((years >= minb) & (years <= maxb))[0]

clim = datamask[:,baseq,:,:,:]
climdist = clim.reshape(ENS,len(baseline)*dayslength,latus1.shape[0],lonus1.shape[0])

### Calculate heat extremes
tx90 = np.nanpercentile(climdist,90,axis=1)
tx95 = np.nanpercentile(climdist,95,axis=1)

### Frequency of heat extremes
count = np.empty((ENS,len(years),latus1.shape[0],lonus1.shape[0]))
for ens in range(ENS):
    for i in range(latus1.shape[0]):
        for j in range(lonus1.shape[0]):
            for yr in range(years.shape[0]):
                summerens = datamask[ens,yr,:,i,j]
                
                if np.isfinite(np.nanmax(summerens)):
                    count[ens,yr,i,j] = (summerens > tx90[ens,i,j]).sum() 
                else:
                    count[ens,yr,i,j] = np.nan


executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
