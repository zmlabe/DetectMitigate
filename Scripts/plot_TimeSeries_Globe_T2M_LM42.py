"""
Calculate GMST for lm42

Author    : Zachary M. Labe
Date      : 18 August 2023
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

variablesall = ['T2M']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 9
years = np.arange(2015,2100+1)
years_LM42 = np.arange(2015,2070+1)
yearsh = np.arange(1921,2014+1,1)
yearsh_LM42 = np.arange(1921,2070+1,1)

yearsall = np.arange(1921,2100+1,1)
yearsall_LM42 = np.arange(1921,2070+1,1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
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
def findNearestValueIndex(array,value):
    index = (np.abs(array-value)).argmin()
    return index
###############################################################################
###############################################################################
###############################################################################
### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_m,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_ssp245,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP245',lat_bounds,lon_bounds)
spear_h,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osm,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
spear_LM42,lats,lons = read_primary_dataset(variq,'SPEAR_MED_LM42p2_test',monthlychoice,'SSP585',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Read in all of LM4.2
lat,lon,spear_LM42_all = LM.read_SPEAR_MED_LM42p2_test('/work/Zachary.Labe/Data/',variq,monthlychoice,4,np.nan,3,'all')

### Calculate anomalies
yearq = np.where((years >= 2015) & (years <= 2029))[0]
climo_spear = np.nanmean(spear_m[:,yearq,:,:],axis=1)
climo_spear_ssp245 = np.nanmean(spear_ssp245[:,yearq,:,:],axis=1)
climo_osspear = np.nanmean(spear_osm[:,yearq,:,:],axis=1)
climo_os10yespear = np.nanmean(spear_osm_10ye[:,yearq,:,:],axis=1)
climo_LM42 = np.nanmean(spear_LM42[:,yearq,:,:],axis=1)

spear_am = spear_m - climo_spear[:,np.newaxis,:,:]
spear_am_ssp245 = spear_ssp245 - climo_spear_ssp245[:,np.newaxis,:,:]
spear_aosm = spear_osm - climo_osspear[:,np.newaxis,:,:]
spear_aosm_10ye = spear_osm_10ye - climo_os10yespear[:,np.newaxis,:,:]
spear_LM42m = spear_LM42 - climo_LM42[:,np.newaxis,:,:]

### Calculate global means
ave_GLOBE = UT.calc_weightedAve(spear_am,lat2)
ave_ssp245_GLOBE  = UT.calc_weightedAve(spear_am_ssp245 ,lat2)
ave_os_GLOBE = UT.calc_weightedAve(spear_aosm,lat2)
ave_os_10ye_GLOBE = UT.calc_weightedAve(spear_aosm_10ye,lat2)
ave_LM42_GLOBE = UT.calc_weightedAve(spear_LM42m,lat2)

### Calculate ensemble mean and spread
ave_GLOBE_avg = np.nanmean(ave_GLOBE,axis=0)
ave_GLOBE_min = np.nanmin(ave_GLOBE,axis=0)
ave_GLOBE_max = np.nanmax(ave_GLOBE,axis=0)

ave_GLOBE_ssp245_avg = np.nanmean(ave_ssp245_GLOBE,axis=0)
ave_GLOBE_ssp245_min = np.nanmin(ave_ssp245_GLOBE,axis=0)
ave_GLOBE_ssp245_max = np.nanmax(ave_ssp245_GLOBE,axis=0)

ave_os_GLOBE_avg = np.nanmean(ave_os_GLOBE,axis=0)
ave_os_GLOBE_min = np.nanmin(ave_os_GLOBE,axis=0)
ave_os_GLOBE_max = np.nanmax(ave_os_GLOBE,axis=0)

ave_os_10ye_GLOBE_avg = np.nanmean(ave_os_10ye_GLOBE,axis=0)
ave_os_10ye_GLOBE_min = np.nanmin(ave_os_10ye_GLOBE,axis=0)
ave_os_10ye_GLOBE_max = np.nanmax(ave_os_10ye_GLOBE,axis=0)

ave_LM42_GLOBE_avg = np.nanmean(ave_LM42_GLOBE,axis=0)
ave_LM42_GLOBE_min = np.nanmin(ave_LM42_GLOBE,axis=0)
ave_LM42_GLOBE_max = np.nanmax(ave_LM42_GLOBE,axis=0)

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

plt.plot(years,ave_GLOBE_avg,linestyle='-',linewidth=2,color='maroon',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP585}') 
plt.plot(years_LM42,ave_LM42_GLOBE_avg,linestyle='--',linewidth=1,color='maroon',dashes=(1,0.3),
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED-LM42_SSP585}')  
plt.plot(years,ave_GLOBE_ssp245_avg,linestyle='-',linewidth=1,color='salmon',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP245}')   

plt.plot(years,ave_os_GLOBE_avg,linestyle='-',linewidth=2,color='darkslategrey',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS}')     
plt.plot(years,ave_os_10ye_GLOBE_avg,linestyle='-',linewidth=2,color='teal',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS_10ye}')    

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2))
plt.xlim([2015,2100])
plt.ylim([-1,5])

plt.ylabel(r'\textbf{Near-Surface Temperature Anomaly [$^{\circ}$C; 2015-2029]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_Globe_T2M_LM42.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Calculate anomalies for historical baseline
yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
yearq_LM42 = np.where((yearsh_LM42 >= 1921) & (yearsh_LM42 <= 1950))[0]
climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)
climoh_LM42 = np.nanmean(np.nanmean(spear_LM42_all[:,yearq_LM42,:,:],axis=1),axis=0)

spear_am = spear_m - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_am_ssp245 = spear_ssp245 - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_aosm = spear_osm - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_aosm_10ye = spear_osm_10ye - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_LM42m = spear_LM42 - climoh_LM42[np.newaxis,np.newaxis,:,:]

### Calculate global means
ave_GLOBEh = UT.calc_weightedAve(spear_am,lat2)
ave_GLOBEh_ssp245 = UT.calc_weightedAve(spear_am_ssp245,lat2)
ave_os_GLOBEh = UT.calc_weightedAve(spear_aosm,lat2)
ave_os_10ye_GLOBEh = UT.calc_weightedAve(spear_aosm_10ye,lat2)
ave_LM42_GLOBEh = UT.calc_weightedAve(spear_LM42m,lat2)

### Calculate ensemble mean and spread
ave_GLOBE_avgh = np.nanmean(ave_GLOBEh,axis=0)
ave_GLOBE_minh = np.nanmin(ave_GLOBEh,axis=0)
ave_GLOBE_maxh = np.nanmax(ave_GLOBEh,axis=0)

ave_ssp245_GLOBE_avgh = np.nanmean(ave_GLOBEh_ssp245,axis=0)
ave_ssp245_GLOBE_minh = np.nanmin(ave_GLOBEh_ssp245,axis=0)
ave_ssp245_GLOBE_maxh = np.nanmax(ave_GLOBEh_ssp245,axis=0)

ave_os_GLOBE_avgh = np.nanmean(ave_os_GLOBEh,axis=0)
ave_os_GLOBE_minh = np.nanmin(ave_os_GLOBEh,axis=0)
ave_os_GLOBE_maxh = np.nanmax(ave_os_GLOBEh,axis=0)

ave_os_10ye_GLOBE_avgh = np.nanmean(ave_os_10ye_GLOBEh,axis=0)
ave_os_10ye_GLOBE_minh = np.nanmin(ave_os_10ye_GLOBEh,axis=0)
ave_os_10ye_GLOBE_maxh = np.nanmax(ave_os_10ye_GLOBEh,axis=0)

ave_LM42_GLOBE_avgh = np.nanmean(ave_LM42_GLOBEh,axis=0)
ave_LM42_GLOBE_minh = np.nanmin(ave_LM42_GLOBEh,axis=0)
ave_LM42_GLOBE_maxh = np.nanmax(ave_LM42_GLOBEh,axis=0)

### Calculate overshoot times
os_yr = np.where((years == 2040))[0][0]
os_10ye_yr = np.where((years == 2031))[0][0]

### Find year of selected GWL
ssp_GWL_15 = findNearestValueIndex(ave_GLOBE_avgh,1.5)
ssp_GWL_16 = findNearestValueIndex(ave_GLOBE_avgh,1.6)
ssp_GWL_17 = findNearestValueIndex(ave_GLOBE_avgh,1.7)
ssp_GWL_18 = findNearestValueIndex(ave_GLOBE_avgh,1.8)
ssp_GWL_21 = findNearestValueIndex(ave_GLOBE_avgh,2.1)

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
# plt.axhline(y=1.5,color='k',linestyle='--',linewidth=1,clip_on=False,
#             dashes=(1,0.3))
# plt.axhline(y=1.6,color='k',linestyle='--',linewidth=1,clip_on=False,
#             dashes=(1,0.3))
# plt.axhline(y=1.7,color='k',linestyle='--',linewidth=1,clip_on=False,
#             dashes=(1,0.3))
# plt.axhline(y=1.8,color='k',linestyle='--',linewidth=1,clip_on=False,
#             dashes=(1,0.3))
# plt.axhline(y=2.1,color='k',linestyle='--',linewidth=1,clip_on=False,
#             dashes=(1,0.3))

# plt.axvline(x=years[ssp_GWL_15],linestyle='--',linewidth=1,clip_on=False,
#             dashes=(1,0.3),color='k')
# plt.axvline(x=years[ssp_GWL_16],linestyle='--',linewidth=1,clip_on=False,
#             dashes=(1,0.3),color='k')
# plt.axvline(x=years[ssp_GWL_17],linestyle='--',linewidth=1,clip_on=False,
#             dashes=(1,0.3),color='k')
# plt.axvline(x=years[ssp_GWL_18],linestyle='--',linewidth=1,clip_on=False,
#             dashes=(1,0.3),color='k')

plt.plot(years,ave_GLOBE_avgh,linestyle='-',linewidth=2,color='maroon',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP585}')   
plt.plot(years_LM42,ave_LM42_GLOBE_avgh,linestyle='--',linewidth=1,color='maroon',dashes=(1,0.3),
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED-LM42_SSP585}')  
plt.plot(years,ave_ssp245_GLOBE_avgh,linestyle='-',linewidth=1,color='salmon',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP245}')  

plt.plot(years,ave_os_GLOBE_avgh,linestyle='-',linewidth=2,color='darkslategrey',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS}')    

plt.plot(years,ave_os_10ye_GLOBE_avgh,linestyle='-',linewidth=2,color='teal',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS_10ye}')   

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-18,18.1,0.5),2),np.round(np.arange(-18,18.1,0.5),2))
plt.xlim([2015,2100])
plt.ylim([0.5,5.5])

plt.ylabel(r'\textbf{Near-Surface Temperature Anomaly [$^{\circ}$C; 1921-1950]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_Globe_T2M_historicalbaseline_LM42.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Calculate anomalies for historical baseline
yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
yearq_LM42 = np.where((yearsh_LM42 >= 1921) & (yearsh_LM42 <= 1950))[0]
climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)
climoh_LM42 = np.nanmean(np.nanmean(spear_LM42_all[:,yearq_LM42,:,:],axis=1),axis=0)

### SPEAR_MED all data
spear_all = np.append(spear_h,spear_m,axis=1)

spear_am = spear_all - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_LM42m = spear_LM42_all - climoh_LM42[np.newaxis,np.newaxis,:,:]

### Calculate global means
ave_GLOBEh = UT.calc_weightedAve(spear_am,lat2)
ave_LM42_GLOBEh = UT.calc_weightedAve(spear_LM42m,lat2)

### Calculate ensemble mean and spread
ave_GLOBE_avgh = np.nanmean(ave_GLOBEh,axis=0)
ave_GLOBE_minh = np.nanmin(ave_GLOBEh,axis=0)
ave_GLOBE_maxh = np.nanmax(ave_GLOBEh,axis=0)

ave_LM42_GLOBE_avgh = np.nanmean(ave_LM42_GLOBEh,axis=0)
ave_LM42_GLOBE_minh = np.nanmin(ave_LM42_GLOBEh,axis=0)
ave_LM42_GLOBE_maxh = np.nanmax(ave_LM42_GLOBEh,axis=0)

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

plt.plot(yearsall,ave_GLOBE_avgh,linestyle='-',linewidth=2,color='maroon',
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED_SSP585}')   
plt.plot(yearsall_LM42,ave_LM42_GLOBE_avgh,linestyle='--',linewidth=1,color='maroon',dashes=(1,0.3),
          clip_on=False,zorder=3,label=r'\textbf{SPEAR_MED-LM42_SSP585}')  

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-18,18.1,0.5),2),np.round(np.arange(-18,18.1,0.5),2))
plt.xlim([1920,2100])
plt.ylim([-0.5,5.5])

plt.ylabel(r'\textbf{Near-Surface Temperature Anomaly [$^{\circ}$C; 1921-1950]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_Globe_T2M_historicalbaseline_ALL_LM42.png',dpi=300)

