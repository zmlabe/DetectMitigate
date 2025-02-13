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

### Calculate global means
globe_spear = UT.calc_weightedAve(spear_am,lat2)
globe_spear_os = UT.calc_weightedAve(spear_aosm,lat2)
globe_spear_os10ye = UT.calc_weightedAve(spear_aosm_10ye,lat2)

land_spear = UT.calc_weightedAve(mask_spear,lat2)
land_spear_os = UT.calc_weightedAve(mask_spear_aosm,lat2)
land_spear_os10ye = UT.calc_weightedAve(mask_spear_aosm_10ye,lat2)

ocean_spear = UT.calc_weightedAve(spear_am_SST,lat2)
ocean_spear_os = UT.calc_weightedAve(spear_aosm_SST,lat2)
ocean_spear_os10ye = UT.calc_weightedAve(spear_aosm_10ye_SST,lat2)

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
# plt.axhline(y=1.2,color='dimgrey',linestyle='-',linewidth=2,clip_on=False)

plt.plot(years,globeMean_spear,color='k',linewidth=2,linestyle='-',label=r'\textbf{Global Mean}')
plt.plot(years,globeMean_spear_os,color='k',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,globeMean_spear_os10ye,color = 'darkgrey',linewidth = 2)

plt.plot(years,landMean_spear,color='r',linewidth=2,linestyle='-',label=r'\textbf{Land Mean}')
plt.plot(years,landMean_spear_os,color='r',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,landMean_spear_os10ye,color = 'lightsalmon',linewidth = 2)

plt.plot(years,oceanMean_spear,color='darkblue',linewidth=2,linestyle='-',label=r'\textbf{Ocean Mean}')
plt.plot(years,oceanMean_spear_os,color='darkblue',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,oceanMean_spear_os10ye,color = 'deepskyblue',linewidth = 2)

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2))
plt.xlim([2030,2100])
plt.ylim([0,7])

plt.ylabel(r'\textbf{Near-Surface Temperature Anomaly [$^{\circ}$C; 1921-1950]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_GMST_GlobeLandOcean.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
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
# plt.axhline(y=1.2,color='dimgrey',linestyle='-',linewidth=2,clip_on=False)

plt.plot(years,landMean_spear,color='r',linewidth=4,linestyle='-',label=r'\textbf{Land Mean}',alpha=0.25)
plt.plot(years,landMean_spear_os,color='r',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,landMean_spear_os10ye,color = 'lightsalmon',linewidth = 2)

plt.plot(years,oceanMean_spear,color='darkblue',linewidth=4,linestyle='-',label=r'\textbf{Ocean Mean}',alpha=0.25)
plt.plot(years,oceanMean_spear_os,color='darkblue',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,oceanMean_spear_os10ye,color = 'deepskyblue',linewidth = 2)

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2))
plt.xlim([2030,2100])
plt.ylim([0,7])

plt.ylabel(r'\textbf{Near-Surface Temperature Anomaly [$^{\circ}$C; 1921-1950]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_GMST_LandOcean.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################               
### Calculate CV
globespear_cv = globeSTD_spear/globeMean_spear
globespear_os_cv = globeSTD_spear_os/globeMean_spear_os
globespear_os10ye_cv = globeSTD_spear_os10ye/globeMean_spear_os10ye

landspear_cv = landSTD_spear/landMean_spear
landspear_os_cv = landSTD_spear_os/landMean_spear_os
landspear_os10ye_cv = landSTD_spear_os10ye/landMean_spear_os10ye

oceanspear_cv = oceanSTD_spear/oceanMean_spear
oceanspear_os_cv = oceanSTD_spear_os/oceanMean_spear_os
oceanspear_os10ye_cv = oceanSTD_spear_os10ye/oceanMean_spear_os10ye

###############################################################################
###############################################################################
###############################################################################
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
# plt.axhline(y=1.2,color='dimgrey',linestyle='-',linewidth=2,clip_on=False)

plt.plot(years,globespear_cv,color='k',linewidth=2,linestyle='-',label=r'\textbf{Global Mean}')
plt.plot(years,globespear_os_cv,color='k',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,globespear_os10ye_cv,color = 'darkgrey',linewidth = 2)

plt.plot(years,landspear_cv,color='r',linewidth=2,linestyle='-',label=r'\textbf{Land Mean}')
plt.plot(years,landspear_os_cv,color='r',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,landspear_os10ye_cv,color = 'lightsalmon',linewidth = 2)

plt.plot(years,oceanspear_cv,color='darkblue',linewidth=2,linestyle='-',label=r'\textbf{Ocean Mean}')
plt.plot(years,oceanspear_os_cv,color='darkblue',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,oceanspear_os10ye_cv,color = 'deepskyblue',linewidth = 2)

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(0,1.01,0.02),2),np.round(np.arange(0,1.01,0.02),2))
plt.xlim([2015,2100])
plt.ylim([0,0.18])

plt.ylabel(r'\textbf{Coefficient of Variation [fraction]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_CV_GlobeLandOcean.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################               
### Calculate CV-modified
globespear_cvmod = globeSpread_spear/globeMean_spear
globespear_os_cvmod = globeSpread_spear_os/globeMean_spear_os
globespear_os10ye_cvmod = globeSpread_spear_os10ye/globeMean_spear_os10ye

landspear_cvmod = landSpread_spear/landMean_spear
landspear_os_cvmod = landSpread_spear_os/landMean_spear_os
landspear_os10ye_cvmod = landSpread_spear_os10ye/landMean_spear_os10ye

oceanspear_cvmod = oceanSpread_spear/oceanMean_spear
oceanspear_os_cvmod = oceanSpread_spear_os/oceanMean_spear_os
oceanspear_os10ye_cvmod = oceanSpread_spear_os10ye/oceanMean_spear_os10ye

###############################################################################
###############################################################################
###############################################################################
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
# plt.axhline(y=1.2,color='dimgrey',linestyle='-',linewidth=2,clip_on=False)

plt.plot(years,globespear_cvmod,color='k',linewidth=2,linestyle='-',label=r'\textbf{Global Mean}')
plt.plot(years,globespear_os_cvmod,color='k',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,globespear_os10ye_cvmod,color = 'darkgrey',linewidth = 2)

plt.plot(years,landspear_cvmod,color='r',linewidth=2,linestyle='-',label=r'\textbf{Land Mean}')
plt.plot(years,landspear_os_cvmod,color='r',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,landspear_os10ye_cvmod,color = 'lightsalmon',linewidth = 2)

plt.plot(years,oceanspear_cvmod,color='darkblue',linewidth=2,linestyle='-',label=r'\textbf{Ocean Mean}')
plt.plot(years,oceanspear_os_cvmod,color='darkblue',linewidth=2,linestyle='--',dashes=(1,0.3))
plt.plot(years,oceanspear_os10ye_cvmod,color = 'deepskyblue',linewidth = 2)

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(0,1.01,0.1),2),np.round(np.arange(0,1.01,0.1),2))
plt.xlim([2015,2100])
plt.ylim([0,0.8])

plt.ylabel(r'\textbf{Modified Coefficient of Variation [spread/mean]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_CVmod_GlobeLandOcean.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
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
# plt.axhline(y=1.2,color='dimgrey',linestyle='-',linewidth=2,clip_on=False)

plt.plot(years,landMean_spear-oceanMean_spear,color='r',linewidth=2,linestyle='-',label=r'\textbf{SSP5-8.5}')
plt.plot(years,landMean_spear_os-oceanMean_spear_os,color='darkblue',linewidth=2,label=r'\textbf{SSP5-3.4OS}')
plt.plot(years,landMean_spear_os10ye-oceanMean_spear_os10ye,color = 'deepskyblue',linewidth = 2,linestyle='--',dashes=(1,0.3),label=r'\textbf{SSP5-3.4OS_10ye}')

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-5,5.01,0.2),2),np.round(np.arange(-5,5.01,0.2),2))
plt.xlim([2015,2100])
plt.ylim([0,3])

plt.ylabel(r'\textbf{Land - Ocean [GMST $^{\circ}$C]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_difference_LandOcean.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
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
# plt.axhline(y=1.2,color='dimgrey',linestyle='-',linewidth=2,clip_on=False)

plt.plot(years,landMean_spear/oceanMean_spear,color='r',linewidth=2,linestyle='-',label=r'\textbf{SSP5-8.5}')
plt.plot(years,landMean_spear_os/oceanMean_spear_os,color='darkblue',linewidth=2,label=r'\textbf{SSP5-3.4OS}')
plt.plot(years,landMean_spear_os10ye/oceanMean_spear_os10ye,color = 'deepskyblue',linewidth = 2,linestyle='--',dashes=(1,0.3),label=r'\textbf{SSP5-3.4OS_10ye}')

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=3,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-5,5.01,0.1),2),np.round(np.arange(-5,5.01,0.1),2))
plt.xlim([2015,2100])
plt.ylim([1.6,1.9])

plt.ylabel(r'\textbf{Ratio [Land/Ocean]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_ratio_LandOcean.png',dpi=300)
