"""
Script looks at AMIP data to check for issues
 
Notes
-----
    Author : Zachary Labe
    Date   : 26 May 2022
"""

### Import modules
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cmocean
import calc_Utilities as UT
import sys

### Parameters
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/CheckingData/Figures/'
variq = 'PRECT'
variqm = 'precip'

### Read in AMIP simulations
directoryAMIP = '/work/Zachary.Labe/Data/SPEAR/SPEAR_c192_pres_HadOISST_HIST_AllForc_Q50/'
temporalave = 'monthly/'
ens = np.arange(1,3+1,1)
mo = 12
years = np.arange(1921,2020+1,1)
tempAMIP = np.empty((len(ens),years.shape[0]*mo,360,576))
for i in range(len(ens)):
    data = Dataset(directoryAMIP + temporalave + '%s_%s_1921-2020.nc' % (variq,str(ens[i]).zfill(2)))
    latm1 = data.variables['lat'][:]
    lonm1 = data.variables['lon'][:]
    tempAMIP[i,:,:,:] = data.variables[variqm][:]
    data.close()
    print('>>> Reading AMIP ensemble number = %s!' % (i+1))

if variq == 'T2M':
    tempAMIP = tempAMIP - 273.15

if temporalave == 'monthly/':
    dat = tempAMIP.reshape(len(ens),years.shape[0],mo,latm1.shape[0],lonm1.shape[0])
    dat = np.asarray(dat)
else:
    dat = np.asarray(tempAMIP)
    
### Calculate averages
lon2,lat2 = np.meshgrid(lonm1,latm1)
ave = UT.calc_weightedAve(dat,lat2)
