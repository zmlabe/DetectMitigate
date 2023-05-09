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
tempAMIP = tempAMIP[:,yearq50,:,:][0]

def calc_index(lat1e,lat1m,lon1e,lon1m,latmin,latmax,lonmin,lonmax,era,tempAMIP):
    latqe = np.where((lat1e >= latmin) & (lat1e <= latmax))[0]
    lonqe = np.where((lon1e >= lonmin) & (lon1e <= lonmax))[0]
    rea1 = era[:,:,latqe,:]
    rea2 = rea1[:,:,:,lonqe]
    lat1eN = lat1e[latqe]
    lon1eN = lon1e[lonqe]
    
    latqm = np.where((lat1m >= latmin) & (lat1m <= latmax))[-1]
    lonqm = np.where((lon1m >= lonmin) & (lon1m <= lonmax))[-1]
    mode1 = tempAMIP[:,:,latqm,:]
    mode2 = mode1[:,:,:,lonqm]
    lat1mN = lat1m[latqm]
    lon1mN = lon1m[lonqm]
                                                        
    lon2eN,lat2eN = np.meshgrid(lon1eN,lat1eN)
    lon2mN,lat2mN = np.meshgrid(lon1mN,lat1mN)
    
    ninoe = UT.calc_weightedAve(rea2,lat2eN)
    ninom = UT.calc_weightedAve(mode2,lat2mN)
    
    ninoea = sts.zscore(np.nanmean(ninoe,axis=1))
    ninoma = sts.zscore(np.nanmean(ninom,axis=1))
    
    return ninoe,ninoea,ninom,ninoma

ninoe,ninoea,ninom,ninoma = calc_index(lat1e,lat1m,lon1e,lon1m,-5,5,190,240,era,tempAMIP)
amoe,amoea,amom,amoma = calc_index(lat1e,lat1m,lon1e,lon1m,0,65,280,360,era,tempAMIP)

plt.figure()
plt.plot(sliceyearera[:-1],ninoea)
plt.plot(sliceyearera[:-1],ninoma)

corrn,pn = sts.pearsonr(ninom.ravel(),ninoe.ravel())

plt.figure()
plt.plot(sliceyearera[:-1],amoea)
plt.plot(sliceyearera[:-1],amoma)

corra,pa = sts.pearsonr(amom.ravel(),amoe.ravel())
