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
import read_BEST as BE

### Parameters
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryERA = '/work/Zachary.Labe/Data/ERA5_05x05/'
directoryBES = '/work/Zachary.Labe/Data/BEST/'
directoryfigure = '/home/Zachary.Labe/Research/CheckingData/Figures/'
variq = 'T2M'
sliceperiod = 'annual'
sliceyearera = np.arange(1950,2021+1,1)
sliceyearbe = np.arange(1960,2020+1,1)
sliceshape = 3
slicenan = 'nan'
addclimo = True

### Read in observations
lat1e,lon1e,era = ER.read_ERA5_monthlyBE_Resolution(variq,directoryERA,sliceperiod,
                                     sliceyearera,sliceshape,addclimo,
                                     slicenan)
lat1,lon1,best = BE.read_BEST(directoryBES,sliceperiod,sliceyearbe,
                            sliceshape,False,slicenan)

### Read in AMIP simulations
directoryAMIP = '/work/Zachary.Labe/Data/SPEAR/SPEAR_c192_pres_HadOISST_HIST_AllForc_Q50/'
temporalave = 'monthly/'
ens = np.arange(1,3+1,1)
years = np.arange(1921,2020+1,1)
tempAMIP = np.empty((len(ens),years.shape[0]*12,360,576))
for i in range(len(ens)):
    data = Dataset(directoryAMIP + temporalave + '%s_%s_1921-2020.nc' % (variq,str(ens[i]).zfill(2)))
    latm1 = data.variables['lat'][:]
    lonm1 = data.variables['lon'][:]
    tempAMIP[i,:,:,:] = data.variables['T2M'][:] - 273.15
    data.close()  
tempAMIP = tempAMIP.reshape(len(ens),years.shape[0],12,360,576)

tempAMIPa = np.nanmean(tempAMIP,axis=2)
    
### Calculate global means
lonm2,latm2 = np.meshgrid(lonm1,latm1)
aveAMIP = UT.calc_weightedAve(tempAMIPa,latm2)

lon2e,lat2e = np.meshgrid(lon1e,lat1e)
# bestplot = best[39+71:,:,:]
eraplot = era[:,:,:]
# avebest = UT.calc_weightedAve(bestplot,lat2e)
aveera = UT.calc_weightedAve(eraplot,lat2e)
    
### Begin plot
fig = plt.figure()

for i in range(ens.shape[0]):
    plt.plot(years,aveAMIP[i],linewidth=2,clip_on=False,color='dimgrey',
             alpha=0.4)
# plt.plot(sliceyearbe,avebest,clip_on=False,linestyle='-',color='darkblue')
plt.plot(sliceyearera,aveera,clip_on=False,linestyle='--',dashes=(1,0.3),
         color='k',linewidth=3)

plt.xlim([1921,2020])
    
plt.savefig(directoryfigure + 'GMST_AMIPS_%s.png' % variq,dpi=300)
    
    
    
    
    
