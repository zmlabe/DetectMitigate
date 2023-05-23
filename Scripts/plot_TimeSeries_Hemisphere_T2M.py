"""
Calculate trend for OS and find relative trend ratio

Author    : Zachary M. Labe
Date      : 22 May 2023
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

variablesall = ['T2M']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 9
years = np.arange(2015,2100+1)

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
seasons = ['annual']
slicemonthnamen = ['ANNUAL']
monthlychoice = seasons[0]
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
### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_m,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osm,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Calculate anomalies
yearq = np.where((years >= 2015) & (years <= 2029))[0]
climo_spear = np.nanmean(spear_m[:,yearq,:,:],axis=1)
climo_osspear = np.nanmean(spear_osm[:,yearq,:,:],axis=1)
climo_os10yespear = np.nanmean(spear_osm_10ye[:,yearq,:,:],axis=1)

spear_am = spear_m - climo_spear[:,np.newaxis,:,:]
spear_aosm = spear_osm - climo_osspear[:,np.newaxis,:,:]
spear_aosm_10ye = spear_osm_10ye - climo_os10yespear[:,np.newaxis,:,:]

### Hemispheres
latq_NH = np.where((lats > 0))[0]
lats_NH = lats[latq_NH]
latq_SH = np.where((lats < 0))[0]
lats_SH = lats[latq_SH]

lon2_NH,lat2_NH = np.meshgrid(lons,lats_NH)
lon2_SH,lat2_SH = np.meshgrid(lons,lats_SH)

spear_m_NH = spear_am[:,:,latq_NH,:]
spear_osm_NH = spear_aosm[:,:,latq_NH,:]
spear_osm_10ye_NH = spear_aosm_10ye[:,:,latq_NH,:]

spear_m_SH = spear_am[:,:,latq_SH,:]
spear_osm_SH = spear_aosm[:,:,latq_SH,:]
spear_osm_10ye_SH = spear_aosm_10ye[:,:,latq_SH,:]

### Calculate global means
ave_NH = UT.calc_weightedAve(spear_m_NH,lat2_NH)
ave_os_NH = UT.calc_weightedAve(spear_osm_NH,lat2_NH)
ave_os_10ye_NH = UT.calc_weightedAve(spear_osm_10ye_NH,lat2_NH)

ave_SH = UT.calc_weightedAve(spear_m_SH,lat2_SH)
ave_os_SH = UT.calc_weightedAve(spear_osm_SH,lat2_SH)
ave_os_10ye_SH = UT.calc_weightedAve(spear_osm_10ye_SH,lat2_SH)

### Calculate ensemble mean and spread
ave_NH_avg = np.nanmean(ave_NH,axis=0)
ave_NH_min = np.nanmin(ave_NH,axis=0)
ave_NH_max = np.nanmax(ave_NH,axis=0)

ave_os_NH_avg = np.nanmean(ave_os_NH,axis=0)
ave_os_NH_min = np.nanmin(ave_os_NH,axis=0)
ave_os_NH_max = np.nanmax(ave_os_NH,axis=0)

ave_os_10ye_NH_avg = np.nanmean(ave_os_10ye_NH,axis=0)
ave_os_10ye_NH_min = np.nanmin(ave_os_10ye_NH,axis=0)
ave_os_10ye_NH_max = np.nanmax(ave_os_10ye_NH,axis=0)

ave_SH_avg = np.nanmean(ave_SH,axis=0)
ave_SH_min = np.nanmin(ave_SH,axis=0)
ave_SH_max = np.nanmax(ave_SH,axis=0)

ave_os_SH_avg = np.nanmean(ave_os_SH,axis=0)
ave_os_SH_min = np.nanmin(ave_os_SH,axis=0)
ave_os_SH_max = np.nanmax(ave_os_SH,axis=0)

ave_os_10ye_SH_avg = np.nanmean(ave_os_10ye_SH,axis=0)
ave_os_10ye_SH_min = np.nanmin(ave_os_10ye_SH,axis=0)
ave_os_10ye_SH_max = np.nanmax(ave_os_10ye_SH,axis=0)

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

plt.plot(years,ave_NH_avg,linestyle='-',linewidth=2,color='maroon',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP585}')    
plt.plot(years,ave_SH_avg,linestyle='--',linewidth=2,color='maroon',
          clip_on=False,zorder=3,dashes=(1,0.7))

plt.plot(years,ave_os_NH_avg,linestyle='-',linewidth=2,color='darkslategrey',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS}')    
plt.plot(years,ave_os_SH_avg,linestyle='--',linewidth=2,color='darkslategrey',
          clip_on=False,zorder=3,dashes=(1,0.7))

plt.plot(years,ave_os_10ye_NH_avg,linestyle='-',linewidth=2,color='teal',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS_10ye}')    
plt.plot(years,ave_os_10ye_SH_avg,linestyle='--',linewidth=2,color='teal',
          clip_on=False,zorder=3,dashes=(1,0.7))

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2))
plt.xlim([2015,2100])
plt.ylim([-1,5])

plt.ylabel(r'\textbf{Near-Surface Temperature Anomaly [$^{\circ}$C]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_Hemisphere_T2M.png',dpi=300)
