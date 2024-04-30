"""
Calculate NAO Index from two large boxes

Reference : McKenna et al. 2021, GRL
Author    : Zachary M. Labe
Date      : 4 July 2022
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
variq = 'SLP'
sliceperiod = 'March'
yearsall = np.arange(1979,2021+1,1)
sliceshape = 3
slicenan = 'nan'
addclimo = True
yearmin = 1979
yearmax = 2019

### Read data
lat,lon,lev,varn = ERA.read_ERA5_monthly1x1(variq,directorydata,sliceperiod,
                                             yearsall,sliceshape,addclimo,slicenan,'surface')
lon2,lat2 = np.meshgrid(lon,lat)

### Read only 1979-2019
yearq = np.where((yearsall >= yearmin) & (yearsall <= yearmax))[0]
years = yearsall[yearq]
var = varn[yearq,:,:]

### Calculate anomalies
anoms = calc_anomalies(years,var)

### Detrend data
data = DT.detrendDataR(anoms,'surface','monthly')
# data = anoms

### Calculate NAO
############################################################
### McKenna et al. 2021, GRL
lonq1_z1 = np.where((lon >=0) & (lon <=60))[0]
lonq2_z1 = np.where((lon >= 270) & (lon <= 360))[0]
lonq_z1 = np.append(lonq1_z1,lonq2_z1)
latq_z1 = np.where((lat >=20) & (lat <=55))[0]
z1_a = data[:,latq_z1,:]
z1_b = z1_a[:,:,lonq_z1]
lon2_z1,lat2_z1 = np.meshgrid(lon[lonq_z1],lat[latq_z1])
z1_m = UT.calc_weightedAve(z1_b,lat2_z1)  
############################################################
lonq1_z2 = np.where((lon >=0) & (lon <=60))[0]
lonq2_z2 = np.where((lon >= 270) & (lon <= 360))[0]
lonq_z2 = np.append(lonq1_z2,lonq2_z2)
latq_z2 = np.where((lat >=55) & (lat <=90))[0]
z2_a = data[:,latq_z2,:]
z2_b = z2_a[:,:,lonq_z2]
lon2_z2,lat2_z2 = np.meshgrid(lon[lonq_z2],lat[latq_z2])
z2_m = UT.calc_weightedAve(z2_b,lat2_z2)

############################################################
### Calculate NAO
nao_raw = z1_m - z2_m

### Save index
directoryoutput = '/work/Zachary.Labe/Data/ClimateIndices/NAO/'
np.savetxt(directoryoutput + 'NAOmodified_%s_%s_%s-%s_detrended.txt' % (variq,sliceperiod,
                                                                yearmin,yearmax),
           np.c_[years,nao_raw])
print('\n========Calculated NAO=======\n')
