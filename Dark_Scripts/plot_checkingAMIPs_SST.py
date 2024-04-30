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
variq = 'SST'
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
era[np.where(era < -999)] = np.nan

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
    tempAMIP[i,:,:,:] = data.variables[variq][:] - 273.15
    data.close()  
tempAMIP = tempAMIP.reshape(len(ens),years.shape[0],12,360,576)

if variq == 'SST':
    directorymask = '/work/Zachary.Labe/Data/masks/ocean_mask_SPEAR_MED.nc'
    data = Dataset(directorymask)
    mask = data.variables['ocean_mask'][:]
    data.close()
    
    mask[np.where(mask <= 0.5)] = 0.
    mask[np.where(mask > 0.5)] = 1.
    tempAMIP = tempAMIP * mask
    tempAMIP[np.where(tempAMIP == 0.)] = np.nan

yearq50 = np.where(years >= 1950)[0]
tempAMIP = tempAMIP[:,yearq50,:,:]

eraa = np.nanmean(era,axis=1)
tempAMIPa = np.nanmean(tempAMIP,axis=2)

lon2e,lat2e = np.meshgrid(lon1e,lat1e)
lon2m,lat2m = np.meshgrid(lon1m,lat1m)

meane = UT.calc_weightedAve(eraa,lat2e)
meanm = UT.calc_weightedAve(tempAMIPa,lat2m)

plt.figure()
plt.plot(meanm.transpose(),color='dimgrey',alpha=0.5,linewidth=1)
plt.plot(meane,color='k',linestyle='--',dashes=(1,0.3))

a = UT.calc_weightedAve(era,lat2e)
aa = UT.calc_weightedAve(tempAMIP,lat2m)[0]

plt.figure()
plt.plot(np.nanmean(a,axis=0))
plt.plot(np.nanmean(aa,axis=0))
