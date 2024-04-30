"""
Calculate Ural High Blocking Index

Reference : Peings et al. 2019
Author    : Zachary M. Labe
Date      : 27 June 022
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
import sys
import read_ERA5_monthly1x1 as ERA
import calc_DetrendData as DT

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/Attribution_SpringNA/Figures/' 
directorydata = '/work/Zachary.Labe/Data/' 

def calc_anomalies(years,data):
    """ 
    Calculate anomalies
    """
    
    ### Baseline - 1981-2010
    if data.ndim == 3:
        yearqold = np.where((years >= 1981) & (years <= 2010))[0]
        climold = np.nanmean(data[yearqold,:,:],axis=0)
        anoms = data - climold
    elif data.ndim == 4:
        yearqold = np.where((years >= 1981) & (years <= 2010))[0]
        climold = np.nanmean(data[:,yearqold,:,:],axis=1)
        anoms = data - climold[:,np.newaxis,:,:]
    
    return anoms

### Parameters
variq = 'Z500'
sliceperiod = 'JFM'
yearsall = np.arange(1979,2021+1,1)
sliceshape = 3
slicenan = 'nan'
addclimo = True
yearmin = 1979
yearmax = 2019

### Read data
latobs,lonobs,lev,varn = ERA.read_ERA5_monthly1x1(variq,directorydata,sliceperiod,
                                             yearsall,sliceshape,addclimo,slicenan,'surface')
lon2,lat2 = np.meshgrid(lonobs,latobs)

### Read only 1979-2021
yearq = np.where((yearsall >= yearmin) & (yearsall <= yearmax))[0]
years = yearsall[yearq]
var = varn[yearq,:,:]

### Calculate anomalies
anoms = calc_anomalies(years,var)

### Detrend data
vardt = DT.detrendDataR(anoms,'surface','monthly')
# vardt = anoms

### Calculate UBI
lonq1 = np.where((lonobs >=0) & (lonobs <=90))[0]
lonq2 = np.where((lonobs >= 330) & (lonobs <= 360))[0]
lonq = np.append(lonq1,lonq2)
latq = np.where((latobs >=45) & (latobs <=80))[0]
anomlon = vardt[:,:,lonq]
anomu = anomlon[:,latq,:]
lat2uq = lat2[latq,:]
lat2u = lat2uq[:,lonq]
ubi = UT.calc_weightedAve(anomu,lat2u)

### Save index
directoryoutput = '/work/Zachary.Labe/Data/ClimateIndices/UBI/'
np.savetxt(directoryoutput + 'UBI_%s_%s_%s-%s_detrended.txt' % (variq,sliceperiod,
                                                                yearmin,yearmax),
           np.c_[years,ubi])
print('\n========Calculated Ural Blocking Index=======\n')
