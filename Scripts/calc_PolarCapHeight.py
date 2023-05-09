"""
Calculate polar cap height for 20CRv3

Author    : Zachary M. Labe
Date      : 16 August 2022
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
import sys
import read_20CRv3_monthly as CR
import calc_DetrendData as DT

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/Attribution_SpringNA/Figures/' 
directorydata = '/work/Zachary.Labe/Data/20CRv3/' 

### Create months to loop through
monthsall = ['January','February','March','April','May','JFM','FM','FMA','AMJ','JJA']

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
for i in range(len(monthsall)):
    variq = 'Z50'
    sliceperiod = monthsall[i]
    yearsall = np.arange(1921,2015+1,1)
    sliceshape = 3
    slicenan = 'nan'
    addclimo = True
    yearmin = 1921
    yearmax = 2015

    ### Read data
    latobs,lonobs,var = CR.read_20CRv3_monthly(variq,directorydata,sliceperiod,
                                    yearsall,sliceshape,addclimo,slicenan)
    lon2,lat2 = np.meshgrid(lonobs,latobs)
    
    ### Calculate anomalies
    anoms = np.asarray(calc_anomalies(yearsall,var))
    
    ### Detrend data
    vardt = DT.detrendDataR(anoms,'surface','monthly')
    # vardt = anoms
    
    ### Calculate SHI
    lonq = np.where((lonobs >=0) & (lonobs <=360))[0]
    latq = np.where((latobs >=65) & (latobs <=90))[0]
    anomlon = vardt[:,:,lonq]
    anoms = anomlon[:,latq,:]
    lat2sq = lat2[latq,:]
    lat2s = lat2sq[:,lonq]
    shi = UT.calc_weightedAve(anoms,lat2s)
    
    ### Save index
    directoryoutput = '/work/Zachary.Labe/Data/ClimateIndices/PCH%s/' % variq
    np.savetxt(directoryoutput + 'PCH%s-20CRv3_%s_%s_%s-%s_detrended.txt' % (variq,variq,sliceperiod,
                                                                    yearmin,yearmax),
               np.c_[yearsall,shi])
    print('\n========Calculated Polar Cap Height Index=======\n')
