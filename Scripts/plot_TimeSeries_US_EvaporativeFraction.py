"""
Calculate timeseries of CONUS mean of evaporative fraction (Latent/(Latent+Sensible))
or known as the ratio of the latent heat flux to the total heat flux (Rastogi et al. 2020, GRL)
                                                                                                                      
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

variablesall = ['EVAP','SHFLX']
variq = variablesall[0]
variq2 = variablesall[1]
numOfEns = 30
numOfEns_10ye = 30
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
reg_name = 'Ce_US'

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
### Get data for latent heat
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_mALLe,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_hALLe,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osmALLe,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10yeALLe,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Calculate latent heat (H = p*lv*E)
spear_mALL = 1000 * 2.45e6 * spear_mALLe * (1/86400) * (1/1000)
spear_hALL = 1000 * 2.45e6 * spear_hALLe * (1/86400) * (1/1000)
spear_osmALL = 1000 * 2.45e6 * spear_osmALLe * (1/86400) * (1/1000)
spear_osm_10yeALL = 1000 * 2.45e6 * spear_osm_10yeALLe * (1/86400) * (1/1000)

### Mask over the USA
spear_mLH,maskobs = dSS.mask_CONUS(spear_mALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_hLH,maskobs = dSS.mask_CONUS(spear_hALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_osmLH,maskobs = dSS.mask_CONUS(spear_osmALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_osm_10yeLH,maskobs = dSS.mask_CONUS(spear_osm_10yeALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)

###############################################################################
###############################################################################
###############################################################################
### Get data for sensible heat
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_mALLs,lats,lons = read_primary_dataset(variq2,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_hALLs,lats,lons = read_primary_dataset(variq2,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osmALLs,lats,lons = read_primary_dataset(variq2,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10yeALLs,lats,lons = read_primary_dataset(variq2,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Mask over the USA
spear_mSH,maskobs = dSS.mask_CONUS(spear_mALLs,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_hSH,maskobs = dSS.mask_CONUS(spear_hALLs,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_osmSH,maskobs = dSS.mask_CONUS(spear_osmALLs,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_osm_10yeSH,maskobs = dSS.mask_CONUS(spear_osm_10yeALLs,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)

###############################################################################
###############################################################################
###############################################################################
### Calculate Evaporative Fraction
spear_m = spear_mLH/(spear_mSH + spear_mLH)
spear_h = spear_hLH/(spear_hSH + spear_hLH)
spear_osm = spear_osmLH/(spear_osmSH + spear_osmLH)
spear_osm_10ye = spear_osm_10yeLH/(spear_osm_10yeSH + spear_osm_10yeLH)

###############################################################################
###############################################################################
###############################################################################
### Calculate anomalies for historical baseline
yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)

### Calculate CONUS means
ave_GLOBEhh = UT.calc_weightedAve(spear_h,lat2)
ave_GLOBEh = UT.calc_weightedAve(spear_m,lat2)
ave_os_GLOBEh = UT.calc_weightedAve(spear_osm,lat2)
ave_os_10ye_GLOBEh = UT.calc_weightedAve(spear_osm_10ye,lat2)

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

leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
      bbox_to_anchor=(0.5,-0.1),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-10,10.01,0.1),2),np.round(np.arange(-10,10.01,0.1),2))
plt.xlim([1921,2100])
plt.ylim([0.0,1])

plt.ylabel(r'\textbf{Evaporative Fraction [Unitless; %s; %s]}' % (seasons[0],reg_name),
            fontsize=7,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_CONUS_EF_historicalbaseline_%s_%s.png' % (seasons[0],reg_name),dpi=300)

