"""
Plot time series of IPO Index

Author    : Zachary M. Labe
Date      : 9 October 2023
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
from scipy.signal import savgol_filter

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['SST']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
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
experimentnames = ['SSP5-8.5','SSP5-3.4OS','SSP5-3.4OS_10ye']
dataset_obs = 'ERA5_MEDS'
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'Globe'
detrend = False

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
def regressData(x,y,runnamesm):    
    print('\n>>> Using regressData function! \n')
    
    if y.ndim == 5: # 5D array
        slope = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        intercept = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        rvalue = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        pvalue = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        stderr = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        for model in range(y.shape[0]):
            print('Completed: Regression for %s!' % runnamesm[model])
            for ens in range(y.shape[1]):
                for i in range(y.shape[3]):
                    for j in range(y.shape[4]):
                        ### 1D time series for regression
                        xx = x
                        yy = y[model,ens,:,i,j]
                        
                        ### Mask data for nans
                        mask = ~np.isnan(xx) & ~np.isnan(yy)
                        varx = xx[mask]
                        vary = yy[mask]
                        
                        ### Calculate regressions
                        slope[model,ens,i,j],intercept[model,ens,i,j], \
                        rvalue[model,ens,i,j],pvalue[model,ens,i,j], \
                        stderr[model,ens,i,j] = sts.linregress(varx,vary)
                        
    if y.ndim == 4: # 4D array
        slope = np.empty((y.shape[0],y.shape[2],y.shape[3]))
        intercept = np.empty((y.shape[0],y.shape[2],y.shape[3],))
        rvalue = np.empty((y.shape[0],y.shape[2],y.shape[3]))
        pvalue = np.empty((y.shape[0],y.shape[2],y.shape[3],))
        stderr = np.empty((y.shape[0],y.shape[2],y.shape[3]))
        for model in range(y.shape[0]):
            print('Completed: Regression for %s!' % runnamesm[model])
            for i in range(y.shape[2]):
                for j in range(y.shape[3]):
                    ### 1D time series for regression
                    xx = x
                    yy = y[model,:,i,j]
                    
                    ### Mask data for nans
                    mask = ~np.isnan(xx) & ~np.isnan(yy)
                    varx = xx[mask]
                    vary = yy[mask]
                        
                    ### Calculate regressions
                    slope[model,i,j],intercept[model,i,j], \
                    rvalue[model,i,j],pvalue[model,i,j], \
                    stderr[model,i,j] = sts.linregress(varx,vary)
                    
    elif y.ndim == 3: #3D array
        slope = np.empty((y.shape[1],y.shape[2]))
        intercept = np.empty((y.shape[1],y.shape[2]))
        rvalue = np.empty((y.shape[1],y.shape[2]))
        pvalue = np.empty((y.shape[1],y.shape[2]))
        stderr = np.empty((y.shape[1],y.shape[2]))
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                ### 1D time series for regression
                xx = x
                yy = y[:,i,j]
                
                ### Mask data for nans
                mask = ~np.isnan(xx) & ~np.isnan(yy)
                varx = xx[mask]
                vary = yy[mask]
                        
                ### Calculate regressions
                slope[i,j],intercept[i,j],rvalue[i,j], \
                pvalue[i,j],stderr[i,j] = sts.linregress(varx,vary)
                        
    print('>>> Completed: Finished regressData function!')
    return slope,intercept,rvalue,pvalue,stderr
###############################################################################
###############################################################################
##############################################################################
### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_mr,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osmr,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10yer,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Remove ensemble mean
if detrend == True:
    spear_m_mean = np.nanmean(spear_mr[:,:,:,:],axis=0)
    spear_osm_mean = np.nanmean(spear_osmr[:,:,:,:],axis=0)
    spear_osm10ye_mean = np.nanmean(spear_osm_10yer[:,:,:,:],axis=0)
    
    spear_m = spear_mr - spear_m_mean
    spear_osm = spear_osmr - spear_osm_mean
    spear_osm_10ye = spear_osm_10yer - spear_osm10ye_mean
else:
    spear_m = spear_mr
    spear_osm = spear_osmr
    spear_osm_10ye = spear_osm_10yer

### Calculate anomalies
yearq = np.where((years >= 2015) & (years <= 2044))[0]
climo_spear = np.nanmean(spear_m[:,yearq,:,:],axis=1)
climo_osspear = np.nanmean(spear_osm[:,yearq,:,:],axis=1)
climo_os10yespear = np.nanmean(spear_osm_10ye[:,yearq,:,:],axis=1)

spear_am = spear_m - climo_spear[:,np.newaxis,:,:]
spear_aosm = spear_osm - climo_osspear[:,np.newaxis,:,:]
spear_aosm_10ye = spear_osm_10ye - climo_os10yespear[:,np.newaxis,:,:]

###############################################################################
### Calculate region 1
latq_z1 = np.where((lats >= 25) & (lats <= 45))[0]
lonq_z1 = np.where((lons >= 140) & (lons <= (180+(180-145))))[0]
lon2_z1,lat2_z1 = np.meshgrid(lons[lonq_z1],lats[latq_z1])

spear_am_z11 = spear_am[:,:,latq_z1,:]
spear_am_z1 = spear_am_z11[:,:,:,lonq_z1]

spear_aosm_z11 = spear_aosm[:,:,latq_z1,:]
spear_aosm_z1 = spear_aosm_z11[:,:,:,lonq_z1]

spear_aosm_10ye_z11 = spear_aosm_10ye[:,:,latq_z1,:]
spear_aosm_10ye_z1 = spear_aosm_10ye_z11[:,:,:,lonq_z1]

### Calculate index for z1
ave_spear_z1 = UT.calc_weightedAve(spear_am_z1,lat2_z1)
ave_spear_os_z1 = UT.calc_weightedAve(spear_aosm_z1,lat2_z1)
ave_spear_os10ye_z1 = UT.calc_weightedAve(spear_aosm_10ye_z1,lat2_z1)

###############################################################################
### Calculate region 2
latq_z2 = np.where((lats >= -10) & (lats <= 10))[0]
lonq_z2= np.where((lons >= 170) & (lons <= (180+(180-90))))[0]
lon2_z2,lat2_z2 = np.meshgrid(lons[lonq_z2],lats[latq_z2])

spear_am_z22 = spear_am[:,:,latq_z2,:]
spear_am_z2 = spear_am_z22[:,:,:,lonq_z2]

spear_aosm_z22 = spear_aosm[:,:,latq_z2,:]
spear_aosm_z2 = spear_aosm_z22[:,:,:,lonq_z2]

spear_aosm_10ye_z22 = spear_aosm_10ye[:,:,latq_z2,:]
spear_aosm_10ye_z2 = spear_aosm_10ye_z22[:,:,:,lonq_z2]

### Calculate index for z2
ave_spear_z2 = UT.calc_weightedAve(spear_am_z2,lat2_z2)
ave_spear_os_z2 = UT.calc_weightedAve(spear_aosm_z2,lat2_z2)
ave_spear_os10ye_z2 = UT.calc_weightedAve(spear_aosm_10ye_z2,lat2_z2)

###############################################################################
### Calculate region 3
latq_z3 = np.where((lats >= -50) & (lats <= -15))[0]
lonq_z3 = np.where((lons >= 150) & (lons <= (180+(180-160))))[0]
lon2_z3,lat2_z3 = np.meshgrid(lons[lonq_z3],lats[latq_z3])

spear_am_z33 = spear_am[:,:,latq_z3,:]
spear_am_z3 = spear_am_z33[:,:,:,lonq_z3]

spear_aosm_z33 = spear_aosm[:,:,latq_z3,:]
spear_aosm_z3 = spear_aosm_z33[:,:,:,lonq_z3]

spear_aosm_10ye_z33 = spear_aosm_10ye[:,:,latq_z3,:]
spear_aosm_10ye_z3 = spear_aosm_10ye_z33[:,:,:,lonq_z3]

### Calculate index for z3
ave_spear_z3 = UT.calc_weightedAve(spear_am_z3,lat2_z3)
ave_spear_os_z3 = UT.calc_weightedAve(spear_aosm_z3,lat2_z3)
ave_spear_os10ye_z3 = UT.calc_weightedAve(spear_aosm_10ye_z3,lat2_z3)

###############################################################################
### Calculate IPO
ave_spear = ave_spear_z2 - ((ave_spear_z1 + ave_spear_z3)/2)
ave_spear_os = ave_spear_os_z2 - ((ave_spear_os_z1 + ave_spear_os_z3)/2)
ave_spear_os10ye = ave_spear_os10ye_z2 - ((ave_spear_os10ye_z1 + ave_spear_os10ye_z3)/2)

### Calculate statistics for plot
max_sp = np.nanmax(ave_spear,axis=0)
min_sp = np.nanmin(ave_spear,axis=0)
mean_sp = np.nanmean(ave_spear,axis=0)

max_os = np.nanmax(ave_spear_os,axis=0)
min_os = np.nanmin(ave_spear_os,axis=0)
mean_os = np.nanmean(ave_spear_os,axis=0)

max_os_10ye = np.nanmax(ave_spear_os10ye,axis=0)
min_os_10ye = np.nanmin(ave_spear_os10ye,axis=0)
mean_os_10ye = np.nanmean(ave_spear_os10ye,axis=0)

minens = [min_sp,min_os,min_os_10ye]
maxens = [max_sp,max_os,max_os_10ye]
meanens = [mean_sp,mean_os,mean_os_10ye]
colors = ['maroon','deepskyblue','darkblue']

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
        
if detrend == True:
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
    
    for i in range(len(meanens)): 
        plt.fill_between(x=years,y1=minens[i],y2=maxens[i],facecolor=colors[i],zorder=1,
                 alpha=0.25,edgecolor='none',clip_on=False)
        plt.plot(years,meanens[i],linestyle='-',linewidth=3.5,color=colors[i],
                  label=r'\textbf{%s}' % experimentnames[i],zorder=1.5,clip_on=False,
                  alpha=0.75)
    
    leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
          bbox_to_anchor=(0.5,1.03),fancybox=True,ncol=3,frameon=False,
          handlelength=1,handletextpad=0.5)
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(-10,11,0.5),2),np.round(np.arange(-10,11,0.5),2))
    plt.xlim([2015,2100])
    plt.ylim([-3,3])
    
    plt.ylabel(r'\textbf{SST Anomaly [$^\circ$C]}',
               fontsize=10,color='k')
    plt.tight_layout()
    plt.savefig(directoryfigure + 'IPOIndex_Detrended_%s_EmissionScenarios_%s.png' % (variq,monthlychoice),dpi=300)
    
else:
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
    
    for i in range(len(meanens)): 
        plt.fill_between(x=years,y1=minens[i],y2=maxens[i],facecolor=colors[i],zorder=1,
                 alpha=0.25,edgecolor='none',clip_on=False)
        plt.plot(years,meanens[i],linestyle='-',linewidth=3.5,color=colors[i],
                  label=r'\textbf{%s}' % experimentnames[i],zorder=1.5,clip_on=False,
                  alpha=0.75)
    
    leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
          bbox_to_anchor=(0.5,1.03),fancybox=True,ncol=3,frameon=False,
          handlelength=1,handletextpad=0.5)
    
    plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
    plt.yticks(np.round(np.arange(-10,11,0.5),2),np.round(np.arange(-10,11,0.5),2))
    plt.xlim([2015,2100])
    plt.ylim([-3,3])
    
    plt.ylabel(r'\textbf{SST Anomaly [$^\circ$C]}',
               fontsize=10,color='k')
    plt.tight_layout()
    plt.savefig(directoryfigure + 'IPOIndex_Raw_%s_EmissionScenarios_%s.png' % (variq,monthlychoice),dpi=300)

