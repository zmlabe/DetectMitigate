"""
Calculate timeseries of CONUS mean of WA

Author    : Zachary M. Labe
Date      : 14 June 2023
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

variablesall = ['TMAXabs']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
years = np.arange(2015,2100+1)
years_os10ye = np.arange(2031,2100+1)
yearsh = np.arange(1921,2014+1,1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/MSFigures_Heat/'
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
### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_m = np.empty((numOfEns,years.shape[0],len(seasons),70,144))
spear_h = np.empty((numOfEns,yearsh.shape[0],len(seasons),70,144))
spear_osm = np.empty((numOfEns,years.shape[0],len(seasons),70,144))
spear_SSP245m = np.empty((numOfEns,years.shape[0],len(seasons),70,144))
spear_osm_10ye = np.empty((numOfEns_10ye,years.shape[0],len(seasons),70,144))
for mm in range(len(seasons)):
    monthlychoice = seasons[mm]
    spear_m[:,:,mm,:,:],lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_h[:,:,mm,:,:],lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_osm[:,:,mm,:,:],lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
    spear_SSP245m[:,:,mm,:,:],lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP245',lat_bounds,lon_bounds)
    spear_osm_10ye[:,:,mm,:,:],lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Meshgrid and mask by CONUS
lon2,lat2 = np.meshgrid(lons,lats)

### Calculate TxX
spear_mALL = np.nanmax(spear_m[:,:,:,:,:],axis=2)
spear_hALL = np.nanmax(spear_h[:,:,:,:,:],axis=2)
spear_osmALL = np.nanmax(spear_osm[:,:,:,:,:],axis=2)
spear_SSP245mALL = np.nanmax(spear_SSP245m[:,:,:,:,:],axis=2)
spear_osm_10yeALL = np.nanmax(spear_osm_10ye[:,:,:,:,:],axis=2)

### Mask over the USA
spear_m,maskobs = dSS.mask_CONUS(spear_mALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_h,maskobs = dSS.mask_CONUS(spear_hALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_osm,maskobs = dSS.mask_CONUS(spear_osmALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_SSP245m,maskobs = dSS.mask_CONUS(spear_SSP245mALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
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
spear_aSSP245m = spear_SSP245m - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_aosm_10ye = spear_osm_10ye - climoh_spear[np.newaxis,np.newaxis,:,:]

### Calculate global means
ave_GLOBEhh = UT.calc_weightedAve(spear_hm,lat2)
ave_GLOBEh = UT.calc_weightedAve(spear_am,lat2)
ave_os_GLOBEh = UT.calc_weightedAve(spear_aosm,lat2)
ave_SSP245_GLOBEh = UT.calc_weightedAve(spear_aSSP245m,lat2)
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

ave_SSP245_GLOBE_avgh = np.nanmean(ave_SSP245_GLOBEh,axis=0)
ave_SSP245_GLOBE_minh = np.nanmin(ave_SSP245_GLOBEh,axis=0)
ave_SSP245_GLOBE_maxh = np.nanmax(ave_SSP245_GLOBEh,axis=0)

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
         alpha=0.3,edgecolor='none',clip_on=False) 
plt.plot(yearsh,ave_GLOBE_avghh,linestyle='-',linewidth=3,color='k',
          clip_on=False,zorder=3,label=r'\textbf{Historical}')   

plt.fill_between(x=years,y1=ave_SSP245_GLOBE_minh,y2=ave_SSP245_GLOBE_maxh,facecolor='darkorange',zorder=1,
         alpha=0.3,edgecolor='none',clip_on=False) 
plt.plot(years,ave_SSP245_GLOBE_avgh,linestyle='--',linewidth=2,color='darkorange',
          clip_on=False,zorder=3,label=r'\textbf{SSP2-4.5}',dashes=(1,0.3)) 

plt.fill_between(x=years,y1=ave_GLOBE_minh,y2=ave_GLOBE_maxh,facecolor='maroon',zorder=1,
         alpha=0.3,edgecolor='none',clip_on=False) 
plt.plot(years,ave_GLOBE_avgh,linestyle='-',linewidth=3,color='maroon',
          clip_on=False,zorder=3,label=r'\textbf{SSP5-8.5}')    

plt.fill_between(x=years,y1=ave_os_GLOBE_minh,y2=ave_os_GLOBE_maxh,facecolor='darkslategrey',zorder=1,
         alpha=0.3,edgecolor='none',clip_on=False) 
plt.plot(years,ave_os_GLOBE_avgh,linestyle='-',linewidth=3,color='darkslategrey',
          clip_on=False,zorder=3,label=r'\textbf{SSP5-3.4OS}')    

plt.fill_between(x=years[-len(years_os10ye):],y1=ave_os_10ye_GLOBE_minh[-len(years_os10ye):],y2=ave_os_10ye_GLOBE_maxh[-len(years_os10ye):],facecolor='lightseagreen',zorder=1,
         alpha=0.3,edgecolor='none',clip_on=False) 
plt.plot(years[-len(years_os10ye):],ave_os_10ye_GLOBE_avgh[-len(years_os10ye):],linestyle='-',linewidth=2,color='lightseagreen',
          clip_on=False,zorder=3,label=r'\textbf{SSP5-3.4OS_10ye}')    

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.16,1.02),fancybox=True,ncol=1,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
### Plot maximum GMST for OS runs
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/MitigateHeat/'
osq = np.genfromtxt(directoryoutput + 'Max_GMST_SSP534OS_Annual.txt')[2]
os10yeq = np.genfromtxt(directoryoutput + 'Max_GMST_SSP534OS_10ye_Annual.txt')[2]
plt.axvline(x=years[int(osq)],color='darkslategrey',linewidth=2,linestyle='-',zorder=200)
plt.axvline(x=years[int(os10yeq)],color='lightseagreen',linewidth=2,linestyle='-',zorder=200)
plt.axvline(x=2040,color='darkslategrey',linewidth=1,linestyle=':',zorder=100)
plt.axvline(x=2031,color='lightseagreen',linewidth=1,linestyle=':',zorder=100)  

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-100,101,1),2),np.round(np.arange(-100,101,1),2))
plt.xlim([1920,2100])
plt.ylim([-3,10])

plt.text(1920,10.2,r'\textbf{[a]}',fontsize=10,color='k')

plt.ylabel(r'\textbf{TXx Anomaly [$^{\circ}$C]}',fontsize=7,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_CONUS_TXx_historicalbaseline_%s_%s.png' % (seasons[0],reg_name),dpi=300)

