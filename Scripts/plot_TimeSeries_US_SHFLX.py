"""
Calculate timeseries of CONUS mean of SHFLX

Author    : Zachary M. Labe
Date      : 2 May 2024
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

variablesall = ['SHFLX']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 9
years = np.arange(2015,2100+1)
yearsh = np.arange(1921,2014+1,1)

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
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_mALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_hALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osmALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10yeALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Mask over the USA
spear_m,maskobs = dSS.mask_CONUS(spear_mALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_h,maskobs = dSS.mask_CONUS(spear_hALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_osm,maskobs = dSS.mask_CONUS(spear_osmALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_osm_10ye,maskobs = dSS.mask_CONUS(spear_osm_10yeALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)

###############################################################################
###############################################################################
###############################################################################
### Calculate anomalies for historical baseline
yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)

spear_hm = spear_h - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_am = spear_m - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_aosm = spear_osm - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_aosm_10ye = spear_osm_10ye - climoh_spear[np.newaxis,np.newaxis,:,:]

### Calculate global means
ave_GLOBEhh = UT.calc_weightedAve(spear_hm,lat2)
ave_GLOBEh = UT.calc_weightedAve(spear_am,lat2)
ave_os_GLOBEh = UT.calc_weightedAve(spear_aosm,lat2)
ave_os_10ye_GLOBEh = UT.calc_weightedAve(spear_aosm_10ye,lat2)

### Calculate ensemble mean and spread
ave_GLOBE_avghh = np.nanmean(ave_GLOBEhh,axis=0)
ave_GLOBE_minhh = np.nanmin(ave_GLOBEhh,axis=0)
ave_GLOBE_maxhh = np.nanmax(ave_GLOBEhh,axis=0)

ave_GLOBE_avgh = np.nanmean(ave_GLOBEh,axis=0)
ave_GLOBE_minh = np.nanmin(ave_GLOBEh,axis=0)
ave_GLOBE_maxh = np.nanmax(ave_GLOBEh,axis=0)

ave_os_GLOBE_avgh = np.nanmean(ave_os_GLOBEh,axis=0)
ave_os_GLOBE_minh = np.nanmin(ave_os_GLOBEh,axis=0)
ave_os_GLOBE_maxh = np.nanmax(ave_os_GLOBEh,axis=0)

ave_os_10ye_GLOBE_avgh = np.nanmean(ave_os_10ye_GLOBEh,axis=0)
ave_os_10ye_GLOBE_minh = np.nanmin(ave_os_10ye_GLOBEh,axis=0)
ave_os_10ye_GLOBE_maxh = np.nanmax(ave_os_10ye_GLOBEh,axis=0)

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

### Plot historical baseline
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

plt.fill_between(x=yearsh,y1=ave_GLOBE_minhh,y2=ave_GLOBE_maxhh,facecolor='k',zorder=1,
         alpha=0.4,edgecolor='none',clip_on=False) 
plt.plot(yearsh,ave_GLOBE_avghh,linestyle='-',linewidth=2,color='k',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_Historical}')   

plt.fill_between(x=years,y1=ave_GLOBE_minh,y2=ave_GLOBE_maxh,facecolor='maroon',zorder=1,
         alpha=0.4,edgecolor='none',clip_on=False) 
plt.plot(years,ave_GLOBE_avgh,linestyle='-',linewidth=2,color='maroon',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP585}')    

plt.fill_between(x=years,y1=ave_os_GLOBE_minh,y2=ave_os_GLOBE_maxh,facecolor='darkslategrey',zorder=1,
         alpha=0.4,edgecolor='none',clip_on=False) 
plt.plot(years,ave_os_GLOBE_avgh,linestyle='-',linewidth=2,color='darkslategrey',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS}')    

plt.fill_between(x=years,y1=ave_os_10ye_GLOBE_minh,y2=ave_os_10ye_GLOBE_maxh,facecolor='teal',zorder=1,
         alpha=0.4,edgecolor='none',clip_on=False) 
plt.plot(years,ave_os_10ye_GLOBE_avgh,linestyle='-',linewidth=2,color='teal',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS_10ye}')    

leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-300,301,5),2),np.round(np.arange(-300,301,5),2))
plt.xlim([1921,2100])
plt.ylim([-15,20])

plt.ylabel(r'\textbf{SHFLX Anomaly [W m$^{-2}$; 1921-1950; %s; %s]}' % (seasons[0],reg_name),
            fontsize=7,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_CONUS_SHFLX_historicalbaseline_%s_%s.png' % (seasons[0],reg_name),dpi=300)

