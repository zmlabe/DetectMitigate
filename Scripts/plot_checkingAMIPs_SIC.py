"""
Function reads in monthly data from ERA5
 
Notes
-----
    Author : Zachary Labe
    Date   : 15 July 2020
"""

### Import modules
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cmocean
import calc_Utilities as UT
import read_ERA5_monthlyBE_Resolution as ER
import scipy.stats as sts
import sys

### Parameters
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryERA = '/work/Zachary.Labe/Data/ERA5_05x05/'
directoryBES = '/work/Zachary.Labe/Data/BEST/'
directoryfigure = '/home/Zachary.Labe/Research/CheckingData/Figures/'
variq = 'SIC'
sliceperiod = 'none'
sliceyearera = np.arange(1950,2021+1,1)
sliceshape = 4
slicenan = 'nan'
addclimo = True

### Read in observations
lat1e,lon1e,era = ER.read_ERA5_monthlyBE_Resolution(variq,directoryERA,sliceperiod,
                                     sliceyearera,sliceshape,addclimo,
                                     slicenan)
era = np.asarray(era[:-1])
era[np.where(era <= 0)] = np.nan

era = np.flip(era,axis=2)
lat1e = np.flip(lat1e)

### Read in AMIP simulations
directoryAMIP = '/work/Zachary.Labe/Data/SPEAR/SPEAR_c192_pres_HadOISST_HIST_AllForc_Q50/'
temporalave = 'monthly/'
ens = np.arange(1,3+1,1)
years = np.arange(1921,2020+1,1)
tempAMIP = np.empty((len(ens),years.shape[0]*12,360,576))
for i in range(len(ens)):
    data = Dataset(directoryAMIP + temporalave + '%s_%s_1921-2020.nc' % (variq,str(ens[i]).zfill(2)))
    lat1m = data.variables['lat'][:]
    lon1m = data.variables['lon'][:]
    tempAMIP[i,:,:,:] = data.variables[variq][:]
    data.close()  
tempAMIP = tempAMIP.reshape(len(ens),years.shape[0],12,360,576)
tempAMIP[np.where(tempAMIP == 0.)] = np.nan

yearq50 = np.where(years >= 1950)[0]
tempAMIPn = tempAMIP[:,yearq50,:,:]

latqe = np.where(lat1e > 45)[0]
latqm = np.where(lat1m > 45)[0]
lat1e = lat1e[latqe]
lat1m = lat1m[latqm]

rean = era[:,:,latqe,:]
mode = tempAMIPn[:,:,:,latqm,:][0]

lon2e,lat2e = np.meshgrid(lon1e,lat1e)
lon2m,lat2m = np.meshgrid(lon1m,lat1m)
meanr = UT.calc_weightedAve(rean,lat2e)
meanm = UT.calc_weightedAve(mode,lat2m)
