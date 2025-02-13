"""
Calculate land ocean time series for GMST

Author    : Zachary M. Labe
Date      : 13 November 2023
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import cmasher as cmr
import palettable.cubehelix as cm
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import sys
import scipy.stats as sts
import read_SPEAR_MED_LM42p2_test as LM

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

numOfEns = 30
numOfEns_10ye = 30
years = np.arange(2015,2100+1)
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
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
dataset_obs = 'ERA5_MEDS'
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'Globe'


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
### Get data - T2M
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_m,lats,lons = read_primary_dataset('T2M','SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_h,lats,lons = read_primary_dataset('T2M','SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osm,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10ye,lats,lons = read_primary_dataset('T2M','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Calculate anomalies
yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)

spear_am = spear_m - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_aosm = spear_osm - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_aosm_10ye = spear_osm_10ye - climoh_spear[np.newaxis,np.newaxis,:,:]

spear_am_LAND = spear_am.copy()
spear_aosm_LAND = spear_aosm.copy()
spear_aosm_10ye_LAND = spear_aosm_10ye.copy()

### Mask out the ocean
mask_spear,emptyobs = dSS.remove_ocean(spear_am_LAND,np.full((spear_am.shape[1],spear_am.shape[2],spear_am.shape[3]),np.nan),lat_bounds,lon_bounds,'MEDS')
mask_spear_aosm,emptyobs = dSS.remove_ocean(spear_aosm_LAND,np.full((spear_am.shape[1],spear_am.shape[2],spear_am.shape[3]),np.nan),lat_bounds,lon_bounds,'MEDS')
mask_spear_aosm_10ye,emptyobs = dSS.remove_ocean(spear_aosm_10ye_LAND,np.full((spear_am.shape[1],spear_am.shape[2],spear_am.shape[3]),np.nan),lat_bounds,lon_bounds,'MEDS')

mask_spear[np.where(mask_spear == 0.)] = np.nan
mask_spear_aosm[np.where(mask_spear_aosm == 0.)] = np.nan
mask_spear_aosm_10ye[np.where(mask_spear_aosm_10ye == 0.)] = np.nan

###############################################################################
###############################################################################
###############################################################################
### Get data - SST
spear_m_SST,lats,lons = read_primary_dataset('SST','SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_h_SST,lats,lons = read_primary_dataset('SST','SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osm_SST,lats,lons = read_primary_dataset('SST','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10ye_SST,lats,lons = read_primary_dataset('SST','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)

### Calculate anomalies
yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
climoh_spear_SST = np.nanmean(np.nanmean(spear_h_SST[:,yearq,:,:],axis=1),axis=0)

spear_am_SST = spear_m_SST - climoh_spear_SST[np.newaxis,np.newaxis,:,:]
spear_aosm_SST = spear_osm_SST - climoh_spear_SST[np.newaxis,np.newaxis,:,:]
spear_aosm_10ye_SST = spear_osm_10ye_SST - climoh_spear_SST[np.newaxis,np.newaxis,:,:]

### Calculate zonal means
globe_spear = np.nanmean(spear_am,axis=3)
globe_spear_os = np.nanmean(spear_aosm,axis=3)
globe_spear_os10ye = np.nanmean(spear_aosm_10ye,axis=3)

land_spear = np.nanmean(mask_spear,axis=3)
land_spear_os = np.nanmean(mask_spear_aosm,axis=3)
land_spear_os10ye = np.nanmean(mask_spear_aosm_10ye,axis=3)

ocean_spear = np.nanmean(spear_am_SST,axis=3)
ocean_spear_os = np.nanmean(spear_aosm_SST,axis=3)
ocean_spear_os10ye = np.nanmean(spear_aosm_10ye_SST,axis=3)

### Calculate ensemble means
globeMean_spear = np.nanmean(globe_spear,axis=0)
globeMean_spear_os = np.nanmean(globe_spear_os,axis=0)
globeMean_spear_os10ye = np.nanmean(globe_spear_os10ye,axis=0)

landMean_spear = np.nanmean(land_spear,axis=0)
landMean_spear_os = np.nanmean(land_spear_os,axis=0)
landMean_spear_os10ye = np.nanmean(land_spear_os10ye,axis=0)

oceanMean_spear = np.nanmean(ocean_spear,axis=0)
oceanMean_spear_os = np.nanmean(ocean_spear_os,axis=0)
oceanMean_spear_os10ye = np.nanmean(ocean_spear_os10ye,axis=0)

### Calculate ensemble median
globeMedian_spear = np.nanmedian(globe_spear,axis=0)
globeMedian_spear_os = np.nanmedian(globe_spear_os,axis=0)
globeMedian_spear_os10ye = np.nanmedian(globe_spear_os10ye,axis=0)

landMedian_spear = np.nanmedian(land_spear,axis=0)
landMedian_spear_os = np.nanmedian(land_spear_os,axis=0)
landMedian_spear_os10ye = np.nanmedian(land_spear_os10ye,axis=0)

oceanMedian_spear = np.nanmedian(ocean_spear,axis=0)
oceanMedian_spear_os = np.nanmedian(ocean_spear_os,axis=0)
oceanMedian_spear_os10ye = np.nanmedian(ocean_spear_os10ye,axis=0)

### Calculate ensemble standard deviation
globeSTD_spear = np.nanstd(globe_spear,axis=0)
globeSTD_spear_os = np.nanstd(globe_spear_os,axis=0)
globeSTD_spear_os10ye = np.nanstd(globe_spear_os10ye,axis=0)

landSTD_spear = np.nanstd(land_spear,axis=0)
landSTD_spear_os = np.nanstd(land_spear_os,axis=0)
landSTD_spear_os10ye = np.nanstd(land_spear_os10ye,axis=0)

oceanSTD_spear = np.nanstd(ocean_spear,axis=0)
oceanSTD_spear_os = np.nanstd(ocean_spear_os,axis=0)
oceanSTD_spear_os10ye = np.nanstd(ocean_spear_os10ye,axis=0)

### Calculate ensemble max
globeMax_spear = np.nanmax(globe_spear,axis=0)
globeMax_spear_os = np.nanmax(globe_spear_os,axis=0)
globeMax_spear_os10ye = np.nanmax(globe_spear_os10ye,axis=0)

landMax_spear = np.nanmax(land_spear,axis=0)
landMax_spear_os = np.nanmax(land_spear_os,axis=0)
landMax_spear_os10ye = np.nanmax(land_spear_os10ye,axis=0)

oceanMax_spear = np.nanmax(ocean_spear,axis=0)
oceanMax_spear_os = np.nanmax(ocean_spear_os,axis=0)
oceanMax_spear_os10ye = np.nanmax(ocean_spear_os10ye,axis=0)

### Calculate ensemble min
globeMin_spear = np.nanmin(globe_spear,axis=0)
globeMin_spear_os = np.nanmin(globe_spear_os,axis=0)
globeMin_spear_os10ye = np.nanmin(globe_spear_os10ye,axis=0)

landMin_spear = np.nanmin(land_spear,axis=0)
landMin_spear_os = np.nanmin(land_spear_os,axis=0)
landMin_spear_os10ye = np.nanmin(land_spear_os10ye,axis=0)

oceanMin_spear = np.nanmin(ocean_spear,axis=0)
oceanMin_spear_os = np.nanmin(ocean_spear_os,axis=0)
oceanMin_spear_os10ye = np.nanmin(ocean_spear_os10ye,axis=0)

### Calculate ensemble spread
globeSpread_spear = globeMax_spear - globeMin_spear
globeSpread_spear_os = globeMax_spear_os - globeMin_spear_os
globeSpread_spear_os10ye = globeMax_spear_os10ye - globeMin_spear_os10ye

landSpread_spear = landMax_spear - landMin_spear
landSpread_spear_os = landMax_spear_os - landMin_spear_os
landSpread_spear_os10ye = landMax_spear_os10ye - landMin_spear_os10ye

oceanSpread_spear = oceanMax_spear - oceanMin_spear
oceanSpread_spear_os = oceanMax_spear_os - oceanMin_spear_os
oceanSpread_spear_os10ye = oceanMax_spear_os10ye - oceanMin_spear_os10ye

###############################################################################
###############################################################################
###############################################################################
## Calculate 5-year averages
land_spear_5 = []
ocean_spear_5 = []

land_spear_os_5 = []
ocean_spear_os_5 = []

land_spear_os10ye_5 = []
ocean_spear_os10ye_5 = []
for y in range(0,len(landMean_spear)-1,5):
    print(y,y+5,years[y:y+5])
    
    land_spear_5q = np.nanmean(landMean_spear[y:y+5,:],axis=0)
    land_spear_5.append(land_spear_5q)
    
    land_spear_os_5q = np.nanmean(landMean_spear_os[y:y+5,:],axis=0)
    land_spear_os_5.append(land_spear_os_5q)
    
    land_spear_os10ye_5q = np.nanmean(landMean_spear_os10ye[y:y+5,:],axis=0)
    land_spear_os10ye_5.append(land_spear_os10ye_5q)
    
    ocean_spear_5q = np.nanmean(oceanMean_spear[y:y+5,:],axis=0)
    ocean_spear_5.append(ocean_spear_5q)
    
    ocean_spear_os_5q = np.nanmean(oceanMean_spear_os[y:y+5,:],axis=0)
    ocean_spear_os_5.append(ocean_spear_os_5q)
    
    ocean_spear_os10ye_5q = np.nanmean(oceanMean_spear_os10ye[y:y+5,:],axis=0)
    ocean_spear_os10ye_5.append(ocean_spear_os10ye_5q)
 
###############################################################################
###############################################################################
###############################################################################
### Calculate ratio 
ratio_spear = np.asarray(land_spear_5)/np.asarray(ocean_spear_5)
ratio_spear_os = np.asarray(land_spear_os_5)/np.asarray(ocean_spear_os_5)
ratio_spear_os10ye = np.asarray(land_spear_os10ye_5)/np.asarray(ocean_spear_os10ye_5)

### Calculate absolute value
ratio_spear[np.where(ratio_spear < 0)] = np.nan
ratio_spear_os[np.where(ratio_spear_os < 0)] = np.nan
ratio_spear_os10ye[np.where(ratio_spear_os10ye < 0)] = np.nan

fig = plt.figure()
ax = plt.subplot(111) 

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
        
adjust_spines(ax, ['left', 'bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(axis='both', direction='out',length=5.5,width=2,
                which='major',pad=3,labelcolor='darkgrey')

plt.vlines(x=0,ymin=-4,ymax=4.4,color='darkgrey',linestyle='--',linewidth=0.75)
plt.vlines(x=-67.6,ymin=-5.1,ymax=9,color='darkgrey',linestyle='--',linewidth=0.75)
plt.vlines(x=67.6,ymin=-5.1,ymax=9,color='darkgrey',linestyle='--',linewidth=0.75)

color=cmr.rainforest(np.linspace(0,0.95,ratio_spear.shape[0]))
for i,c in zip(range(ratio_spear.shape[0]),color):
    plt.plot(lats,ratio_spear[i,:],c=c,linewidth=0.75,zorder=5)

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
            bbox_to_anchor=(0.5225, 0.9),fancybox=True,ncol=12,frameon=False,
            handlelength=0,handletextpad=-1.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.ylabel(r'\textbf{Zonal Mean Ratio [Land/Ocean] -- SSP5-8.5}',
            fontsize=8,alpha=1,color='dimgrey')       

plt.xticks(np.arange(-90,91,15),map(str,np.arange(-90,91,15)),rotation=0,fontsize=9,color='k')
ylabels = map(str,np.arange(-10,11,1))
plt.yticks(np.arange(-10,11,1),ylabels,fontsize=7.5,color='k')
plt.ylim([0,5])
plt.xlim([-90,90])

t4 = plt.annotate(r'\textbf{EQUATOR}',
          textcoords='axes fraction',
          xy=(0,0), xytext=(0.435,0.02),
            fontsize=10,color='dimgrey',alpha=1)
t4 = plt.annotate(r'\textbf{ANTARCTIC}',
          textcoords='axes fraction',
          xy=(0,0), xytext=(0.0,0.02),
            fontsize=7,color='dimgrey',alpha=1)
t4 = plt.annotate(r'\textbf{ARCTIC}',
          textcoords='axes fraction',
          xy=(0,0), xytext=(1,0.02),
            fontsize=7,color='dimgrey',alpha=1,ha='right')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_ZonalRatio_LandOcean_SSP585.png',dpi=300)
###############################################################################
fig = plt.figure()
ax = plt.subplot(111) 

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
        
adjust_spines(ax, ['left', 'bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(axis='both', direction='out',length=5.5,width=2,
                which='major',pad=3,labelcolor='darkgrey')

plt.vlines(x=0,ymin=-4,ymax=4.4,color='darkgrey',linestyle='--',linewidth=0.75)
plt.vlines(x=-67.6,ymin=-5.1,ymax=9,color='darkgrey',linestyle='--',linewidth=0.75)
plt.vlines(x=67.6,ymin=-5.1,ymax=9,color='darkgrey',linestyle='--',linewidth=0.75)
    
color=cmr.rainforest(np.linspace(0,0.95,ratio_spear_os.shape[0]))
for i,c in zip(range(ratio_spear_os.shape[0]),color):
    plt.plot(lats,ratio_spear_os[i,:],c=c,linewidth=0.75,zorder=5)

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
            bbox_to_anchor=(0.5225, 0.9),fancybox=True,ncol=12,frameon=False,
            handlelength=0,handletextpad=-1.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.ylabel(r'\textbf{Zonal Mean Ratio [Land/Ocean] -- SSP5-3.4OS}',
            fontsize=8,alpha=1,color='dimgrey')       

plt.xticks(np.arange(-90,91,15),map(str,np.arange(-90,91,15)),rotation=0,fontsize=9,color='k')
ylabels = map(str,np.arange(-10,11,1))
plt.yticks(np.arange(-10,11,1),ylabels,fontsize=7.5,color='k')
plt.ylim([0,5])
plt.xlim([-90,90])

t4 = plt.annotate(r'\textbf{EQUATOR}',
          textcoords='axes fraction',
          xy=(0,0), xytext=(0.435,0.02),
            fontsize=10,color='dimgrey',alpha=1)
t4 = plt.annotate(r'\textbf{ANTARCTIC}',
          textcoords='axes fraction',
          xy=(0,0), xytext=(0.0,0.02),
            fontsize=7,color='dimgrey',alpha=1)
t4 = plt.annotate(r'\textbf{ARCTIC}',
          textcoords='axes fraction',
          xy=(0,0), xytext=(1,0.02),
            fontsize=7,color='dimgrey',alpha=1,ha='right')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_ZonalRatio_LandOcean_SSP534OS.png',dpi=300)
###############################################################################
fig = plt.figure()
ax = plt.subplot(111) 

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
        
adjust_spines(ax, ['left', 'bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(axis='both', direction='out',length=5.5,width=2,
                which='major',pad=3,labelcolor='darkgrey')

plt.vlines(x=0,ymin=-4,ymax=4.4,color='darkgrey',linestyle='--',linewidth=0.75)
plt.vlines(x=-67.6,ymin=-5.1,ymax=9,color='darkgrey',linestyle='--',linewidth=0.75)
plt.vlines(x=67.6,ymin=-5.1,ymax=9,color='darkgrey',linestyle='--',linewidth=0.75)
    
color=cmr.rainforest(np.linspace(0,0.95,ratio_spear_os10ye.shape[0]))
for i,c in zip(range(ratio_spear_os10ye.shape[0]),color):
    plt.plot(lats,ratio_spear_os10ye[i,:],c=c,linewidth=0.75,zorder=5,)

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
            bbox_to_anchor=(0.5225, 0.9),fancybox=True,ncol=12,frameon=False,
            handlelength=0,handletextpad=-1.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.ylabel(r'\textbf{Zonal Mean Ratio [Land/Ocean] -- SSP5-3.4OS_10ye}',
            fontsize=8,alpha=1,color='dimgrey')       

plt.xticks(np.arange(-90,91,15),map(str,np.arange(-90,91,15)),rotation=0,fontsize=9,color='k')
ylabels = map(str,np.arange(-10,11,1))
plt.yticks(np.arange(-10,11,1),ylabels,fontsize=7.5,color='k')
plt.ylim([0,5])
plt.xlim([-90,90])

t4 = plt.annotate(r'\textbf{EQUATOR}',
          textcoords='axes fraction',
          xy=(0,0), xytext=(0.435,0.02),
            fontsize=10,color='dimgrey',alpha=1)
t4 = plt.annotate(r'\textbf{ANTARCTIC}',
          textcoords='axes fraction',
          xy=(0,0), xytext=(0.0,0.02),
            fontsize=7,color='dimgrey',alpha=1)
t4 = plt.annotate(r'\textbf{ARCTIC}',
          textcoords='axes fraction',
          xy=(0,0), xytext=(1,0.02),
            fontsize=7,color='dimgrey',alpha=1,ha='right')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_ZonalRatio_LandOcean_SSP34OS10ye.png',dpi=300)

#############################################################################
#############################################################################
#############################################################################
fig = plt.figure()
ax = plt.subplot(111) 

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
        
adjust_spines(ax, ['left', 'bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(axis='both', direction='out',length=5.5,width=2,
                which='major',pad=3,labelcolor='darkgrey')

color=cmr.rainforest(np.linspace(0,0.9,ratio_spear.shape[0]))
for i,c in zip(range(ratio_spear.shape[0]),color):
    plt.plot(lats,ratio_spear[i,:],c=c,linewidth=1,zorder=5)

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
           bbox_to_anchor=(0.5225, 0.9),fancybox=True,ncol=12,frameon=False,
           handlelength=0,handletextpad=-1.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.ylabel(r'\textbf{Zonal Mean Ratio [Land/Ocean] -- SSP5-8.5}',
            fontsize=8,alpha=1,color='dimgrey')  
plt.xlabel(r'\textbf{Latitude [$^{\circ}$N]',fontsize=8,alpha=1,color='dimgrey')     

plt.xticks(np.arange(-90,91,15),map(str,np.arange(-90,91,15)),rotation=0,fontsize=9,color='k')
ylabels = map(str,np.arange(-10,11,1))
plt.yticks(np.arange(-10,11,1),ylabels,fontsize=7.5,color='k')
plt.ylim([0.5,2.5])
plt.xlim([0,50])

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_ZonalRatio_LandOcean_SSP585_NHextra.png',dpi=300)
###############################################################################
fig = plt.figure()
ax = plt.subplot(111) 

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
        
adjust_spines(ax, ['left', 'bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(axis='both', direction='out',length=5.5,width=2,
                which='major',pad=3,labelcolor='darkgrey')
    
color=cmr.rainforest(np.linspace(0,0.9,ratio_spear_os.shape[0]))
for i,c in zip(range(ratio_spear_os.shape[0]),color):
    plt.plot(lats,ratio_spear_os[i,:],c=c,linewidth=1,zorder=5)

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
           bbox_to_anchor=(0.5225, 0.9),fancybox=True,ncol=12,frameon=False,
           handlelength=0,handletextpad=-1.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.ylabel(r'\textbf{Zonal Mean Ratio [Land/Ocean] -- SSP5-3.4OS}',
            fontsize=8,alpha=1,color='dimgrey') 
plt.xlabel(r'\textbf{Latitude [$^{\circ}$N]',fontsize=8,alpha=1,color='dimgrey')         

plt.xticks(np.arange(-90,91,15),map(str,np.arange(-90,91,15)),rotation=0,fontsize=9,color='k')
ylabels = map(str,np.arange(-10,11,1))
plt.yticks(np.arange(-10,11,1),ylabels,fontsize=7.5,color='k')
plt.ylim([0.5,2.5])
plt.xlim([0,50])

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_ZonalRatio_LandOcean_SSP534OS_NHextra.png',dpi=300)
###############################################################################
fig = plt.figure()
ax = plt.subplot(111) 

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
        
adjust_spines(ax, ['left', 'bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(axis='both', direction='out',length=5.5,width=2,
                which='major',pad=3,labelcolor='darkgrey')
    
color=cmr.rainforest(np.linspace(0,0.9,ratio_spear_os10ye.shape[0]))
for i,c in zip(range(ratio_spear_os10ye.shape[0]),color):
    plt.plot(lats,ratio_spear_os10ye[i,:],c=c,linewidth=1,zorder=5,)

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
           bbox_to_anchor=(0.5225, 0.9),fancybox=True,ncol=12,frameon=False,
           handlelength=0,handletextpad=-1.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.ylabel(r'\textbf{Zonal Mean Ratio [Land/Ocean] -- SSP5-3.4OS_10ye}',
            fontsize=8,alpha=1,color='dimgrey')     
plt.xlabel(r'\textbf{Latitude [$^{\circ}$N]',fontsize=8,alpha=1,color='dimgrey')     

plt.xticks(np.arange(-90,91,15),map(str,np.arange(-90,91,15)),rotation=0,fontsize=9,color='k')
ylabels = map(str,np.arange(-10,11,1))
plt.yticks(np.arange(-10,11,1),ylabels,fontsize=7.5,color='k')
plt.ylim([0.5,2.5])
plt.xlim([0,50])

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_ZonalRatio_LandOcean_SSP34OS10ye_NHextra.png',dpi=300)
