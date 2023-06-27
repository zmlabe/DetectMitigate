"""
Calculate Southern Annular Mode for the OS runs

Author    : Zachary M. Labe
Date      : 26 June 2023
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
import read_SPEAR_MED as SPM
import read_SPEAR_MED_Scenario as SPSS
import read_SPEAR_MED_SSP534OS_10ye as SPSS10

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['SLP']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 9
yearsf = np.arange(2015,2100+1)
yearsh = np.arange(1921,2014+1,1)
yearsall = np.arange(1921,2100+1,1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
reg_name = 'Globe'

### Calculate linear trends
def calcTrend(data):
    if data.ndim == 3:
        slopes = np.empty((data.shape[1],data.shape[2]))
        x = np.arange(data.shape[0])
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                mask = np.isfinite(data[:,i,j])
                y = data[:,i,j]
                
                if np.sum(mask) == y.shape[0]:
                    xx = x
                    yy = y
                else:
                    xx = x[mask]
                    yy = y[mask]      
                if np.isfinite(np.nanmean(yy)):
                    slopes[i,j],intercepts, \
                    r_value,p_value,std_err = sts.linregress(xx,yy)
                else:
                    slopes[i,j] = np.nan
    elif data.ndim == 4:
        slopes = np.empty((data.shape[0],data.shape[2],data.shape[3]))
        x = np.arange(data.shape[1])
        for ens in range(data.shape[0]):
            print('Ensemble member completed: %s!' % (ens+1))
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    mask = np.isfinite(data[ens,:,i,j])
                    y = data[ens,:,i,j]
                    
                    if np.sum(mask) == y.shape[0]:
                        xx = x
                        yy = y
                    else:
                        xx = x[mask]
                        yy = y[mask]      
                    if np.isfinite(np.nanmean(yy)):
                        slopes[ens,i,j],intercepts, \
                        r_value,p_value,std_err = sts.linregress(xx,yy)
                    else:
                        slopes[ens,i,j] = np.nan
    
    dectrend = slopes * 10.   
    print('Completed: Finished calculating trends!')      
    return dectrend

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
### Get data
# lat1,lon1,spear = SPM.read_SPEAR_MED('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/',variq,'none',5,np.nan,30,'all')
# lat1,lon1,spear_os = SPSS.read_SPEAR_MED_Scenario('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/','SSP534OS',variq,'none',5,np.nan,30,'futureforcing')
# lat1,lon1,spear_os_10ye = SPSS10.read_SPEAR_MED_SSP534OS_10ye('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_SSP534OS_10ye/monthly/',variq,'none',5,np.nan,9,'futureforcing')

# ### Calculate annual means
# spear_yr = np.nanmean(spear,axis=2)
# spear_os_yr = np.nanmean(spear_os,axis=2)
# spear_os_10ye_yr = np.nanmean(spear_os_10ye,axis=2)

# ### Calculate SAM index 
# latq_40s = np.where((lat1 >= -40.25) & (lat1 <= -39.75))[0]
# latq_65s = np.where((lat1 >= -65.25) & (lat1 <= -64.75))[0]

### Calculate zonal means at -40 and -65
spear_40 = np.nanmean(spear_yr[:,:,latq_40s,:],axis=(2,3))
spear_os_40 = np.nanmean(spear_os_yr[:,:,latq_40s,:],axis=(2,3))
spear_os_10ye_40 = np.nanmean(spear_os_10ye_yr[:,:,latq_40s,:],axis=(2,3))

spear_65 = np.nanmean(spear_yr[:,:,latq_65s,:],axis=(2,3))
spear_os_65 = np.nanmean(spear_os_yr[:,:,latq_65s,:],axis=(2,3))
spear_os_10ye_65 = np.nanmean(spear_os_10ye_yr[:,:,latq_65s,:],axis=(2,3))

### Standardize Index
base = np.where((yearsall >= 1981) & (yearsall <= 2010))[0]
climo_40_sam = np.nanmean(spear_40[:,base],axis=1)
std_40_sam = np.nanstd(spear_40[:,base],axis=1)
climo_65_sam = np.nanmean(spear_65[:,base],axis=1)
std_65_sam = np.nanstd(spear_65[:,base],axis=1)

spear_40z = (spear_40-climo_40_sam[:,np.newaxis])/std_40_sam[:,np.newaxis]
spear_os_40z = (spear_os_40-climo_40_sam[:,np.newaxis])/std_40_sam[:,np.newaxis]
spear_os_10ye_40z = (spear_os_10ye_40-climo_40_sam[:9,np.newaxis])/std_40_sam[:9,np.newaxis]

spear_65z = (spear_65-climo_65_sam[:,np.newaxis])/std_65_sam[:,np.newaxis]
spear_os_65z = (spear_os_65-climo_65_sam[:,np.newaxis])/std_65_sam[:,np.newaxis]
spear_os_10ye_65z = (spear_os_10ye_65-climo_65_sam[:9,np.newaxis])/std_65_sam[:9,np.newaxis]

### Calculate inde
spear_samz = sts.zscore(spear_40z - spear_65z,axis=1)
spear_os_samz = sts.zscore(spear_os_40z - spear_os_65z,axis=1)
spear_os_10ye_samz = sts.zscore(spear_os_10ye_40z - spear_os_10ye_65z,axis=1)

### Calculate ensemble statistics
ensmean = np.nanmean(spear_samz,axis=0)
ensmax = np.nanmax(spear_samz,axis=0)
ensmin = np.nanmin(spear_samz,axis=0)

ensmean_os = np.nanmean(spear_os_samz,axis=0)
ensmax_os = np.nanmax(spear_os_samz,axis=0)
ensmin_os = np.nanmin(spear_os_samz,axis=0)

ensmean_os_10ye = np.nanmean(spear_os_10ye_samz,axis=0)
ensmax_os_10ye = np.nanmax(spear_os_10ye_samz,axis=0)
ensmin_os_10ye = np.nanmin(spear_os_10ye_samz,axis=0)

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

plt.fill_between(x=yearsall,y1=ensmin,y2=ensmax,facecolor='maroon',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsall[:],ensmean,linestyle='-',linewidth=1,
          zorder=1.5,color='maroon',label=r'\textbf{SPEAR_MED_SSP585}',clip_on=False)

plt.fill_between(x=yearsf,y1=ensmin_os,y2=ensmax_os,facecolor='darkblue',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsf[:],ensmean_os,linestyle='-',linewidth=1,
          zorder=1.5,color='darkblue',label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False)

plt.fill_between(x=yearsf,y1=ensmin_os_10ye,y2=ensmax_os_10ye,facecolor='aqua',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsf[:],ensmean_os_10ye,linestyle='-',linewidth=1,
          zorder=1.5,color='aqua',label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',clip_on=False)

leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
      bbox_to_anchor=(0.5,1.01),fancybox=True,ncol=3,frameon=False,
      handlelength=1,handletextpad=0.5)

plt.xticks(np.arange(1920,2101,20),np.arange(1920,2101,20))
plt.yticks(np.round(np.arange(-48,48.1,1),2),np.round(np.arange(-48,48.1,1),2))
plt.xlim([1920,2100])
plt.ylim([-4,4])

plt.ylabel(r'\textbf{Southern Annular Mode Index [normalized]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'SAMindex_OSComparison.png',dpi=300)
