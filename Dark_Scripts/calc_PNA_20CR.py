"""
Calculate PNA Index Modified Pointwise Method for the PNA Index for 20CRv3

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
    variq = 'Z500'
    sliceperiod = monthsall[i]
    yearsall = np.arange(1921,2015+1,1)
    sliceshape = 3
    slicenan = 'nan'
    addclimo = True
    yearmin = 1921
    yearmax = 2015
    
    ### Read data
    lat,lon,var = CR.read_20CRv3_monthly(variq,directorydata,sliceperiod,
                                    yearsall,sliceshape,addclimo,slicenan)
    lon2,lat2 = np.meshgrid(lon,lat)
    
    ### Calculate anomalies
    anoms = np.asarray(calc_anomalies(yearsall,var))
    
    ### Detrend data
    # data = DT.detrendDataR(anoms,'surface','monthly')
    data = anoms
    
    ### Calculate PNA
    ############################################################
    ### Modified Pointwise Method for the PNA Index from the CPC
    latq_z1 = np.where((lat >= 15) & (lat <=25))[0]
    lonq_z1 = np.where((lon >= 180) & (lon <= 220))[0]
    z1_a = data[:,latq_z1,:]
    z1_b = z1_a[:,:,lonq_z1]
    lon2_z1,lat2_z1 = np.meshgrid(lon[lonq_z1],lat[latq_z1])
    z1_m = UT.calc_weightedAve(z1_b,lat2_z1)  
    ############################################################
    latq_z2 = np.where((lat >= 40) & (lat <=50))[0]
    lonq_z2 = np.where((lon >= 180) & (lon <= 220))[0]
    z2_a = data[:,latq_z2,:]
    z2_b = z2_a[:,:,lonq_z2]
    lon2_z2,lat2_z2 = np.meshgrid(lon[lonq_z2],lat[latq_z2])
    z2_m = UT.calc_weightedAve(z2_b,lat2_z2)
    ############################################################
    latq_z3 = np.where((lat >= 45) & (lat <=60))[0]
    lonq_z3 = np.where((lon >= 235) & (lon <= 255))[0]
    z3_a = data[:,latq_z3,:]
    z3_b = z3_a[:,:,lonq_z3]
    lon2_z3,lat2_z3 = np.meshgrid(lon[lonq_z3],lat[latq_z3])
    z3_m = UT.calc_weightedAve(z3_b,lat2_z3)
    ############################################################
    latq_z4 = np.where((lat >= 25) & (lat <=35))[0]
    lonq_z4 = np.where((lon >= 270) & (lon <= 290))[0]
    z4_a = data[:,latq_z4,:]
    z4_b = z4_a[:,:,lonq_z4]
    lon2_z4,lat2_z4 = np.meshgrid(lon[lonq_z4],lat[latq_z4])
    z4_m = UT.calc_weightedAve(z4_b,lat2_z4)
        
    ############################################################
    ### Calculate PNA
    pna_raw = z1_m - z2_m + z3_m - z4_m
    
    ### Save index
    directoryoutput = '/work/Zachary.Labe/Data/ClimateIndices/PNA/'
    np.savetxt(directoryoutput + 'PNAmodified-20CRv3_%s_%s_%s-%s.txt' % (variq,sliceperiod,
                                                                    yearmin,yearmax),
               np.c_[yearsall,pna_raw])
    print('\n========Calculated PNA=======\n')
