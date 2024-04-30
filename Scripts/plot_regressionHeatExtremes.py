"""
Evaluate predictors of heat extremes

Author    : Zachary M. Labe
Date      : 10 April 2024
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import cmasher as cmr
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import sys
import scipy.stats as sts
from sklearn.linear_model import LinearRegression

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['TMAX']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
yearsh = np.arange(1921,2014+1,1)
years = np.arange(1921,2100+1)
years_os = np.arange(2011,2100+1)
years_os_10ye = np.arange(2031,2100+1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
dataset_obs = 'ERA5_MEDS'
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'US'

###############################################################################
###############################################################################
###############################################################################
### Read in climate models      
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons 
###############################################################################
###############################################################################
###############################################################################
def findNearestValueIndex(array,value):
    index = (np.abs(array-value)).argmin()
    return index
###############################################################################
###############################################################################
###############################################################################
### Get data
def getConusTimeSeries(variableRead,regionRead):
    lat_bounds,lon_bounds = UT.regions(regionRead)
    
    spear_mt,lats,lons = read_primary_dataset(variableRead,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_ht,lats,lons = read_primary_dataset(variableRead,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_osmt,lats,lons = read_primary_dataset(variableRead,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
    spear_osm_10yet,lats,lons = read_primary_dataset(variableRead,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
    lon2,lat2 = np.meshgrid(lons,lats)
    
    yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
    climoh_speart = np.nanmean(np.nanmean(spear_ht[:,yearq,:,:],axis=1),axis=0)
    
    spear_hALL = spear_ht - climoh_speart[np.newaxis,np.newaxis,:,:]
    spear_mALL = spear_mt - climoh_speart[np.newaxis,np.newaxis,:,:]
    spear_osmALL = spear_osmt - climoh_speart[np.newaxis,np.newaxis,:,:]
    spear_osm_10yeALL = spear_osm_10yet - climoh_speart[np.newaxis,np.newaxis,:,:]
    
    if regionRead == 'US':
        ### Mask over the USA
        spear_m,emptyobs = dSS.mask_CONUS(spear_mALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
        spear_h,emptyobs = dSS.mask_CONUS(spear_hALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
        spear_os,emptyobs = dSS.mask_CONUS(spear_osmALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
        spear_os_10ye,emptyobs = dSS.mask_CONUS(spear_osm_10yeALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
    elif variableRead == 'T2M':
        ### Mask out the ocean
        spear_m,emptyobs = dSS.remove_ocean(spear_mALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),lat_bounds,lon_bounds,'MEDS')
        spear_h,emptyobs = dSS.remove_ocean(spear_hALL,np.full((spear_hALL.shape[1],spear_hALL.shape[2],spear_hALL.shape[3]),np.nan),lat_bounds,lon_bounds,'MEDS')
        spear_os,emptyobs = dSS.remove_ocean(spear_osmALL,np.full((spear_osmALL.shape[1],spear_osmALL.shape[2],spear_osmALL.shape[3]),np.nan),lat_bounds,lon_bounds,'MEDS')
        spear_os_10ye,emptyobs = dSS.remove_ocean(spear_osm_10yeALL,np.full((spear_osm_10yeALL.shape[1],spear_osm_10yeALL.shape[2],spear_osm_10yeALL.shape[3]),np.nan),lat_bounds,lon_bounds,'MEDS')
    else:
        spear_h = spear_hALL 
        spear_m = spear_mALL
        spear_os = spear_osmALL
        spear_os_10ye = spear_osm_10yeALL
    
    ### Calculate global average in SPEAR_MED
    lon2,lat2 = np.meshgrid(lons,lats)
    spear_ah_globeht = UT.calc_weightedAve(spear_h,lat2)
    spear_am_globeht = UT.calc_weightedAve(spear_m,lat2)
    spear_osm_globeht = UT.calc_weightedAve(spear_os,lat2)
    spear_osm_10ye_globeht = UT.calc_weightedAve(spear_os_10ye,lat2)
    
    return spear_ah_globeht,spear_am_globeht,spear_osm_globeht,spear_osm_10ye_globeht

t2m_historical,t2m_future,t2m_os,t2m_os10ye = getConusTimeSeries('T2M','NHExtra')
precip_historical,precip_future,precip_os,precip_os10ye = getConusTimeSeries('PRECT','US')
rh_historical,rh_future,rh_os,rh_os10ye = getConusTimeSeries('rh_ref','US')
sst_historical,sst_future,sst_os,sst_os10ye = getConusTimeSeries('SST','NHExtra')

###############################################################################
###############################################################################
###############################################################################
### Read in heatwave data

### Read in SPEAR_MED
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
count90 = data.variables['count90'][:]
data.close()

### Read in SPEAR_MED_SSP534OS
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
filename_os = directorydatah + name_os
data_os = Dataset(filename_os)
count90_os = data_os.variables['count90'][:]
data_os.close()

### Read in SPEAR_MED_SSP534OS_10ye
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os10ye = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
filename_os10ye = directorydatah + name_os10ye
data_os10ye = Dataset(filename_os10ye)
count90_os10ye = data_os10ye.variables['count90'][:]
data_os10ye.close()

### Calculate spatial averages
lon2us,lat2us = np.meshgrid(lonus,latus)
avg_count90 = UT.calc_weightedAve(count90,lat2us)
avg_count90_historical = avg_count90[:,:t2m_historical.shape[0]]
avg_count90_future = avg_count90[:,-t2m_future.shape[0]:]
avg_count90_os = UT.calc_weightedAve(count90_os,lat2us)
avg_count90_os10ye = UT.calc_weightedAve(count90_os10ye,lat2us)

###############################################################################
###############################################################################
###############################################################################
### Calculate X training data
nTrain = 25
t2m_historical_TRAIN = t2m_historical[:nTrain,:]
t2m_future_TRAIN = t2m_future[:nTrain,:]
t2m_os_TRAIN = t2m_os[:nTrain,:]
t2m_os_10ye_TRAIN = t2m_os10ye[:nTrain,:]

precip_historical_TRAIN = precip_historical[:nTrain,:]
precip_future_TRAIN = precip_future[:nTrain,:]
precip_os_TRAIN = precip_os[:nTrain,:]
precip_os_10ye_TRAIN = precip_os10ye[:nTrain,:]

rh_historical_TRAIN = rh_historical[:nTrain,:]
rh_future_TRAIN = rh_future[:nTrain,:]
rh_os_TRAIN = rh_os[:nTrain,:]
rh_os_10ye_TRAIN = rh_os10ye[:nTrain,:]

sst_historical_TRAIN = sst_historical[:nTrain,:]
sst_future_TRAIN = sst_future[:nTrain,:]
sst_os_TRAIN = sst_os[:nTrain,:]
sst_os_10ye_TRAIN = sst_os10ye[:nTrain,:]

### Y train
count90_historical_TRAIN = avg_count90_historical[:nTrain,:]
count90_future_TRAIN = avg_count90_future[:nTrain,:]
count90_os_TRAIN = avg_count90_os[:nTrain,:]
count90_os10ye_TRAIN = avg_count90_os10ye[:nTrain,:]

###############################################################################
###############################################################################
###############################################################################
### Calculate Y testing data
nTest = 30 - nTrain
t2m_historical_TEST = t2m_historical[-nTest:,:]
t2m_future_TEST = t2m_future[-nTest:,:]
t2m_os_TEST = t2m_os[-nTest:,:]
t2m_os_10ye_TEST = t2m_os10ye[-nTest:,:]

precip_historical_TEST = precip_historical[-nTest:,:]
precip_future_TEST = precip_future[-nTest:,:]
precip_os_TEST = precip_os[-nTest:,:]
precip_os_10ye_TEST = precip_os10ye[-nTest:,:]

rh_historical_TEST = rh_historical[-nTest:,:]
rh_future_TEST = rh_future[-nTest:,:]
rh_os_TEST = rh_os[-nTest:,:]
rh_os_10ye_TEST = rh_os10ye[-nTest:,:]

sst_historical_TEST = sst_historical[-nTest:,:]
sst_future_TEST = sst_future[-nTest:,:]
sst_os_TEST = sst_os[-nTest:,:]
sst_os_10ye_TEST = sst_os10ye[-nTest:,:]

### Y TEST
count90_historical_TEST = avg_count90_historical[-nTest:,:]
count90_future_TEST = avg_count90_future[-nTest:,:]
count90_os_TEST = avg_count90_os[-nTest:,:]
count90_os10ye_TEST = avg_count90_os10ye[-nTest:,:]
