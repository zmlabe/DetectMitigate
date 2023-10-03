"""
Calculate sahel precipitation for the OS runs

Author    : Zachary M. Labe
Date      : 16 June 2023
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

variablesall = ['PRECT']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
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
selectGWL = 1.5
selectGWLn = '%s' % (int(selectGWL*10))
yrplus = 3

lat1,lon1,spear = SPM.read_SPEAR_MED('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/',variq,'none',5,np.nan,30,'all')
lat1,lon1,spear_os = SPSS.read_SPEAR_MED_Scenario('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/','SSP534OS',variq,'none',5,np.nan,30,'futureforcing')
lat1,lon1,spear_os_10ye = SPSS10.read_SPEAR_MED_SSP534OS_10ye('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_SSP534OS_10ye/monthly/',variq,'none',5,np.nan,30,'futureforcing')

### Calculate dry season (December-January-February)
spear_djf = np.empty((spear.shape[0],spear.shape[1]-1,lat1.shape[0],lon1.shape[0]))
spear_os_djf = np.empty((spear_os.shape[0],spear_os.shape[1]-1,lat1.shape[0],lon1.shape[0]))
for ens in range(len(spear)):
    spear_djf[ens,:,:,:] = UT.calcDecJanFeb(spear[ens],lat1,lon1,'surface',1)
    spear_os_djf[ens,:,:,:] = UT.calcDecJanFeb(spear_os[ens],lat1,lon1,'surface',1)

spear_os_10ye_djf = np.empty((spear_os_10ye.shape[0],spear_os_10ye.shape[1]-1,lat1.shape[0],lon1.shape[0]))
for ens in range(len(spear_os_10ye)):
    spear_os_10ye_djf[ens,:,:,:] = UT.calcDecJanFeb(spear_os_10ye[ens],lat1,lon1,'surface',1)

### Calculate wet season (July-August-September)
spear_jas = np.nanmean(spear[:,:,6:9,:,:],axis=2)
spear_os_jas = np.nanmean(spear_os[:,:,6:9,:,:],axis=2)
spear_os_10ye_jas = np.nanmean(spear_os_10ye[:,:,6:9,:,:],axis=2)

###############################################################################
### Central Sahel
latqe = np.where((lat1 >= 10) & (lat1 <= 20))[0]
lonqe1 = np.where((lon1 >= 355))[0]
lonqe2 = np.where((lon1 >= 0) & (lon1 <= 20))[0]
lonqe = np.append(lonqe1,lonqe2)
late = lat1[latqe]
lone = lon1[lonqe]
lon2e,lat2e = np.meshgrid(lone,late)

spear_djf_cent1 = spear_djf[:,:,latqe,:]
spear_djf_cent = spear_djf_cent1[:,:,:,lonqe]

spear_os_djf_cent1 = spear_os_djf[:,:,latqe,:]
spear_os_djf_cent = spear_os_djf_cent1[:,:,:,lonqe]

spear_os_10ye_djf_cent1 = spear_os_10ye_djf[:,:,latqe,:]
spear_os_10ye_djf_cent = spear_os_10ye_djf_cent1[:,:,:,lonqe]

spear_djf_cent_ave = UT.calc_weightedAve(spear_djf_cent,lat2e)
spear_os_djf_cent_ave = UT.calc_weightedAve(spear_os_djf_cent,lat2e)
spear_os_10ye_djf_cent_ave = UT.calc_weightedAve(spear_os_10ye_djf_cent,lat2e)

spear_jas_cent1 = spear_jas[:,:,latqe,:]
spear_jas_cent = spear_jas_cent1[:,:,:,lonqe]

spear_os_jas_cent1 = spear_os_jas[:,:,latqe,:]
spear_os_jas_cent = spear_os_jas_cent1[:,:,:,lonqe]

spear_os_10ye_jas_cent1 = spear_os_10ye_jas[:,:,latqe,:]
spear_os_10ye_jas_cent = spear_os_10ye_jas_cent1[:,:,:,lonqe]

spear_jas_cent_ave = UT.calc_weightedAve(spear_jas_cent,lat2e)
spear_os_jas_cent_ave = UT.calc_weightedAve(spear_os_jas_cent,lat2e)
spear_os_10ye_jas_cent_ave = UT.calc_weightedAve(spear_os_10ye_jas_cent,lat2e)

###############################################################################
### Western Sahel
latqw = np.where((lat1 >= 10) & (lat1 <= 20))[0]
lonqw = np.where((lon1 >= 340) & (lon1 <= 355))[0]
latw = lat1[latqw]
lonw = lon1[lonqw]
lon2w,lat2w = np.meshgrid(lonw,latw)

spear_djf_west1 = spear_djf[:,:,latqw,:]
spear_djf_west = spear_djf_west1[:,:,:,lonqw]

spear_os_djf_west1 = spear_os_djf[:,:,latqw,:]
spear_os_djf_west = spear_os_djf_west1[:,:,:,lonqw]

spear_os_10ye_djf_west1 = spear_os_10ye_djf[:,:,latqw,:]
spear_os_10ye_djf_west = spear_os_10ye_djf_west1[:,:,:,lonqw]

spear_djf_west_ave = UT.calc_weightedAve(spear_djf_west,lat2w)
spear_os_djf_west_ave = UT.calc_weightedAve(spear_os_djf_west,lat2w)
spear_os_10ye_djf_west_ave = UT.calc_weightedAve(spear_os_10ye_djf_west,lat2w)

spear_jas_west1 = spear_jas[:,:,latqw,:]
spear_jas_west = spear_jas_west1[:,:,:,lonqw]

spear_os_jas_west1 = spear_os_jas[:,:,latqw,:]
spear_os_jas_west = spear_os_jas_west1[:,:,:,lonqw]

spear_os_10ye_jas_west1 = spear_os_10ye_jas[:,:,latqw,:]
spear_os_10ye_jas_west = spear_os_10ye_jas_west1[:,:,:,lonqw]

spear_jas_west_ave = UT.calc_weightedAve(spear_jas_west,lat2w)
spear_os_jas_west_ave = UT.calc_weightedAve(spear_os_jas_west,lat2w)
spear_os_10ye_jas_west_ave = UT.calc_weightedAve(spear_os_10ye_jas_west,lat2w)

###############################################################################
### Calculate ensemble mean
spear_djf_cent_ave_mean = np.nanmean(spear_djf_cent_ave[:,:],axis=0)
spear_jas_cent_ave_mean = np.nanmean(spear_jas_cent_ave[:,:],axis=0)
spear_djf_west_ave_mean = np.nanmean(spear_djf_west_ave[:,:],axis=0)
spear_jas_west_ave_mean = np.nanmean(spear_jas_west_ave[:,:],axis=0)

spear_os_djf_cent_ave_mean = np.nanmean(spear_os_djf_cent_ave[:,:],axis=0)
spear_os_jas_cent_ave_mean = np.nanmean(spear_os_jas_cent_ave[:,:],axis=0)
spear_os_djf_west_ave_mean = np.nanmean(spear_os_djf_west_ave[:,:],axis=0)
spear_os_jas_west_ave_mean = np.nanmean(spear_os_jas_west_ave[:,:],axis=0)

spear_os_10ye_djf_cent_ave_mean = np.nanmean(spear_os_10ye_djf_cent_ave[:,:],axis=0)
spear_os_10ye_jas_cent_ave_mean = np.nanmean(spear_os_10ye_jas_cent_ave[:,:],axis=0)
spear_os_10ye_djf_west_ave_mean = np.nanmean(spear_os_10ye_djf_west_ave[:,:],axis=0)
spear_os_10ye_jas_west_ave_mean = np.nanmean(spear_os_10ye_jas_west_ave[:,:],axis=0)

###############################################################################
### Calculate ensemble max
spear_djf_cent_ave_max = np.nanmax(spear_djf_cent_ave[:,:],axis=0)
spear_jas_cent_ave_max = np.nanmax(spear_jas_cent_ave[:,:],axis=0)
spear_djf_west_ave_max = np.nanmax(spear_djf_west_ave[:,:],axis=0)
spear_jas_west_ave_max = np.nanmax(spear_jas_west_ave[:,:],axis=0)

spear_os_djf_cent_ave_max = np.nanmax(spear_os_djf_cent_ave[:,:],axis=0)
spear_os_jas_cent_ave_max = np.nanmax(spear_os_jas_cent_ave[:,:],axis=0)
spear_os_djf_west_ave_max = np.nanmax(spear_os_djf_west_ave[:,:],axis=0)
spear_os_jas_west_ave_max = np.nanmax(spear_os_jas_west_ave[:,:],axis=0)

spear_os_10ye_djf_cent_ave_max = np.nanmax(spear_os_10ye_djf_cent_ave[:,:],axis=0)
spear_os_10ye_jas_cent_ave_max = np.nanmax(spear_os_10ye_jas_cent_ave[:,:],axis=0)
spear_os_10ye_djf_west_ave_max = np.nanmax(spear_os_10ye_djf_west_ave[:,:],axis=0)
spear_os_10ye_jas_west_ave_max = np.nanmax(spear_os_10ye_jas_west_ave[:,:],axis=0)

###############################################################################
### Calculate ensemble min
spear_djf_cent_ave_min = np.nanmin(spear_djf_cent_ave[:,:],axis=0)
spear_jas_cent_ave_min = np.nanmin(spear_jas_cent_ave[:,:],axis=0)
spear_djf_west_ave_min = np.nanmin(spear_djf_west_ave[:,:],axis=0)
spear_jas_west_ave_min = np.nanmin(spear_jas_west_ave[:,:],axis=0)

spear_os_djf_cent_ave_min = np.nanmin(spear_os_djf_cent_ave[:,:],axis=0)
spear_os_jas_cent_ave_min = np.nanmin(spear_os_jas_cent_ave[:,:],axis=0)
spear_os_djf_west_ave_min = np.nanmin(spear_os_djf_west_ave[:,:],axis=0)
spear_os_jas_west_ave_min = np.nanmin(spear_os_jas_west_ave[:,:],axis=0)

spear_os_10ye_djf_cent_ave_min = np.nanmin(spear_os_10ye_djf_cent_ave[:,:],axis=0)
spear_os_10ye_jas_cent_ave_min = np.nanmin(spear_os_10ye_jas_cent_ave[:,:],axis=0)
spear_os_10ye_djf_west_ave_min = np.nanmin(spear_os_10ye_djf_west_ave[:,:],axis=0)
spear_os_10ye_jas_west_ave_min = np.nanmin(spear_os_10ye_jas_west_ave[:,:],axis=0)

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
        
fig = plt.figure(figsize=(9,7))
ax = plt.subplot(221)

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

plt.fill_between(x=yearsall[1:],y1=spear_djf_west_ave_min,y2=spear_djf_west_ave_max,facecolor='maroon',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsall[1:],spear_djf_west_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='maroon',label=r'\textbf{SPEAR_MED_SSP585}',clip_on=False)

plt.fill_between(x=yearsf[1:],y1=spear_os_djf_west_ave_min,y2=spear_os_djf_west_ave_max,facecolor='darkblue',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsf[1:],spear_os_djf_west_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='darkblue',label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False)

plt.fill_between(x=yearsf[1:],y1=spear_os_10ye_djf_west_ave_min,y2=spear_os_10ye_djf_west_ave_max,facecolor='aqua',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsf[1:],spear_os_10ye_djf_west_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='aqua',label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',clip_on=False)

leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
      bbox_to_anchor=(1.1,-1.35),fancybox=True,ncol=3,frameon=False,
      handlelength=1,handletextpad=0.5)

plt.xticks(np.arange(1920,2101,20),np.arange(1920,2101,20))
plt.yticks(np.round(np.arange(-48,48.1,0.1),2),np.round(np.arange(-48,48.1,0.1),2))
plt.xlim([1920,2100])
plt.ylim([0,0.5])

plt.ylabel(r'\textbf{PRECT [mm/day]}',
            fontsize=10,color='k')

plt.text(1923,0.42,r'\textbf{DJF}',fontsize=19,color='k')
plt.text(2097,0.5,r'\textbf{[a]}',fontsize=8,color='k')
plt.title(r'\textbf{WESTERN SAHEL}',fontsize=15,color='dimgrey')

###############################################################################
ax = plt.subplot(222)

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

plt.fill_between(x=yearsall[1:],y1=spear_djf_cent_ave_min,y2=spear_djf_cent_ave_max,facecolor='maroon',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsall[1:],spear_djf_cent_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='maroon',label=r'\textbf{SPEAR_MED_SSP585}',clip_on=False)

plt.fill_between(x=yearsf[1:],y1=spear_os_djf_cent_ave_min,y2=spear_os_djf_cent_ave_max,facecolor='darkblue',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsf[1:],spear_os_djf_cent_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='darkblue',label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False)

plt.fill_between(x=yearsf[1:],y1=spear_os_10ye_djf_cent_ave_min,y2=spear_os_10ye_djf_cent_ave_max,facecolor='aqua',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsf[1:],spear_os_10ye_djf_cent_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='aqua',label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',clip_on=False)

plt.xticks(np.arange(1920,2101,20),np.arange(1920,2101,20))
plt.yticks(np.round(np.arange(-48,48.1,0.1),2),np.round(np.arange(-48,48.1,0.1),2))
plt.xlim([1920,2100])
plt.ylim([0,0.5])

plt.text(1923,0.42,r'\textbf{DJF}',fontsize=19,color='k')
plt.text(2097,0.5,r'\textbf{[b]}',fontsize=8,color='k')
plt.title(r'\textbf{CENTRAL SAHEL}',fontsize=15,color='dimgrey')

###############################################################################
ax = plt.subplot(223)

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

plt.fill_between(x=yearsall,y1=spear_jas_west_ave_min,y2=spear_jas_west_ave_max,facecolor='maroon',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsall[:],spear_jas_west_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='maroon',label=r'\textbf{SPEAR_MED_SSP585}',clip_on=False)

plt.fill_between(x=yearsf,y1=spear_os_jas_west_ave_min,y2=spear_os_jas_west_ave_max,facecolor='darkblue',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsf[:],spear_os_jas_west_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='darkblue',label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False)

plt.fill_between(x=yearsf,y1=spear_os_10ye_jas_west_ave_min,y2=spear_os_10ye_jas_west_ave_max,facecolor='aqua',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsf[:],spear_os_10ye_jas_west_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='aqua',label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',clip_on=False)
plt.xticks(np.arange(1920,2101,20),np.arange(1920,2101,20))
plt.yticks(np.round(np.arange(-48,48.1,1),2),np.round(np.arange(-48,48.1,1),2))
plt.xlim([1920,2100])
plt.ylim([1,8])

plt.ylabel(r'\textbf{PRECT [mm/day]}',
            fontsize=10,color='k')
plt.text(2097,8,r'\textbf{[c]}',fontsize=8,color='k')
plt.text(1923,7.22,r'\textbf{JAS}',fontsize=19,color='k')

###############################################################################
ax = plt.subplot(224)

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

plt.fill_between(x=yearsall,y1=spear_jas_cent_ave_min,y2=spear_jas_cent_ave_max,facecolor='maroon',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsall[:],spear_jas_cent_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='maroon',label=r'\textbf{SPEAR_MED_SSP585}',clip_on=False)

plt.fill_between(x=yearsf,y1=spear_os_jas_cent_ave_min,y2=spear_os_jas_cent_ave_max,facecolor='darkblue',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsf[:],spear_os_jas_cent_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='darkblue',label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False)

plt.fill_between(x=yearsf,y1=spear_os_10ye_jas_cent_ave_min,y2=spear_os_10ye_jas_cent_ave_max,facecolor='aqua',zorder=1,
          alpha=0.25,edgecolor='none',clip_on=False)
plt.plot(yearsf[:],spear_os_10ye_jas_cent_ave_mean,linestyle='-',linewidth=1,
          zorder=1.5,color='aqua',label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',clip_on=False)

plt.xticks(np.arange(1920,2101,20),np.arange(1920,2101,20))
plt.yticks(np.round(np.arange(-48,48.1,1),2),np.round(np.arange(-48,48.1,1),2))
plt.xlim([1920,2100])
plt.ylim([1,8])

plt.text(2097,8,r'\textbf{[d]}',fontsize=8,color='k')
plt.text(1923,7.22,r'\textbf{JAS}',fontsize=19,color='k')

# plt.tight_layout()
plt.savefig(directoryfigure + 'SahelSeasons_%s_EmissionScenarios.png' % (variq),dpi=300)
