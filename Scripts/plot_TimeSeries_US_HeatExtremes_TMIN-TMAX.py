"""
Calculate timeseries of US means of TMIN and compare to TMAX

Author    : Zachary M. Labe
Date      : 20 October 2023
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

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

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
### Epochs for +- years around selected GWL
lat_bounds,lon_bounds = UT.regions(reg_name)

### Read in SPEAR_MED - TMAX
variq = 'TMAX'
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
freq90 = data.variables['freq90'][:]
data.close()

### Read in SPEAR_MED_SSP534OS
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
filename_os = directorydatah + name_os
data_os = Dataset(filename_os)
freq90_os = data_os.variables['freq90'][:]
data_os.close()

### Read in SPEAR_MED_SSP534OS_10ye
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os10ye = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
filename_os10ye = directorydatah + name_os10ye
data_os10ye = Dataset(filename_os10ye)
freq90_os10ye = data_os10ye.variables['freq90'][:]
data_os10ye.close()

### Calculate spatial averages
lon2us,lat2us = np.meshgrid(lonus,latus)
avg_freq90 = UT.calc_weightedAve(freq90,lat2us)
avg_freq90_os = UT.calc_weightedAve(freq90_os,lat2us)
avg_freq90_os10ye = UT.calc_weightedAve(freq90_os10ye,lat2us)

### Calculate ensemble means
ave_avg = np.nanmean(avg_freq90,axis=0)
ave_os_avg = np.nanmean(avg_freq90_os,axis=0)
ave_os_10ye_avg = np.nanmean(avg_freq90_os10ye,axis=0)

###############################################################################
###############################################################################
############################################################################### 
### Read in SPEAR_MED - TMIN
variq = 'TMIN'
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
freq90_tmin = data.variables['freq90'][:]
data.close()

### Read in SPEAR_MED_SSP534OS
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
filename_os = directorydatah + name_os
data_os = Dataset(filename_os)
freq90_os_tmin = data_os.variables['freq90'][:]
data_os.close()

### Read in SPEAR_MED_SSP534OS_10ye
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os10ye = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
filename_os10ye = directorydatah + name_os10ye
data_os10ye = Dataset(filename_os10ye)
freq90_os10ye_tmin = data_os10ye.variables['freq90'][:]
data_os10ye.close()

### Calculate spatial averages
lon2us,lat2us = np.meshgrid(lonus,latus)
avg_freq90_tmin = UT.calc_weightedAve(freq90_tmin,lat2us)
avg_freq90_os_tmin = UT.calc_weightedAve(freq90_os_tmin,lat2us)
avg_freq90_os10ye_tmin = UT.calc_weightedAve(freq90_os10ye_tmin,lat2us)

### Calculate ensemble means
ave_avg_tmin = np.nanmean(avg_freq90_tmin,axis=0)
ave_os_avg_tmin = np.nanmean(avg_freq90_os_tmin,axis=0)
ave_os_10ye_avg_tmin = np.nanmean(avg_freq90_os10ye_tmin,axis=0)

###############################################################################
###############################################################################
###############################################################################               
### Plot Figure
### Adjust axes in time series plots 
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([]) 
        
fig = plt.figure()
ax = plt.subplot(111)

adjust_spines(ax, ['left', 'bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both',length=4.,width=2,which='major',color='dimgrey')
ax.tick_params(axis='x',labelsize=6,pad=1.5)
ax.tick_params(axis='y',labelsize=6,pad=1.5)
ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.8)

plt.plot(years,ave_avg*100.,linestyle='-',linewidth=1,color='maroon',zorder=3,label=r'\textbf{SPEAR_MED_SSP585}')    

plt.plot(years_os,ave_os_avg*100.,linestyle='-',linewidth=1,color='darkslategrey',zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS}')    

plt.plot(years_os_10ye,ave_os_10ye_avg*100.,linestyle='-',linewidth=1,color='teal',zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS_10ye}')    

plt.plot(years,ave_avg_tmin*100.,linestyle='--',dashes=(1,0.3),linewidth=1,color='maroon',zorder=3)    
plt.plot(years_os,ave_os_avg_tmin*100.,linestyle='--',dashes=(1,0.3),linewidth=1,color='darkslategrey',zorder=3)    
plt.plot(years_os_10ye,ave_os_10ye_avg_tmin*100.,linestyle='--',dashes=(1,0.3),linewidth=1,color='teal',zorder=3)    

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=0,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(0,101,10),2),np.round(np.arange(0,101,10),2))
plt.xlim([2015,2100])
plt.ylim([10,80])

plt.ylabel(r'\textbf{Frequency of Tx90--Tn90 over CONUS [Percent]}',fontsize=7,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_Tx90-Tn90_%s_%s.png' % (seasons[0],reg_name),dpi=300)
