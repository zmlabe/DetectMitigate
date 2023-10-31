"""
Plot trends in last 30 years for SSPs and natural forcings

Author     : Zachary M. Labe
Date       : 25 October 2023
Version    : 1 
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import cmocean
import cmasher as cmr
import numpy as np
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import read_BEST as B
from scipy.interpolate import griddata as g
import scipy.stats as sts
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Directories
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
monthq = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
modelGCMs = ['SPEAR_MED_NATURAL','SPEAR_MED',
             'SPEAR_MED_Scenario','SPEAR_MED_Scenario']
dataset_obs = 'ERA5_MEDS'
lenOfPicks = len(modelGCMs)
monthlychoice = 'annual'
allvariables = ['Temperature','Temperature','Temperature','Temperature','Precipitation','Precipitation','Precipitation','Precipitation']
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
timeper = ['futureforcing','futureforcing','futureforcing','futureforcing']
scenarioall = ['natural','SSP585','SSP119','SSP245']
scenarioallnames = ['Natural','SSP5-8.5','SSP1-1.9','SSP2-4.5',
                    'Natural','SSP5-8.5','SSP1-1.9','SSP2-4.5']
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
baseline = np.arange(1951,1980+1,1)
###############################################################################
###############################################################################
window = 0
yearsall = [np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1)]
years = yearsall[0]
###############################################################################
###############################################################################
numOfEns = 30
numOfEns_10ye = 30
lentime = len(yearsall)
###############################################################################
###############################################################################
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
###############################################################################
### Read in model and observational/reanalysis data
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons 
def setcolor(x, color):
      for m in x:
          for t in x[m][1]:
              t.set_color(color) 

### Loop in all climate models for T2M
data_all = []
for no in range(len(modelGCMs)):
    dataset = modelGCMs[no]
    scenario = scenarioall[no]
    data_allq,lats,lons = read_primary_dataset('T2M',dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
    data_all.append(data_allq)
data = np.asarray(data_all)

trend_natural = UT.linearTrend(data[0],years,'surface',2071,2100)*10.
trend_ssp585 = UT.linearTrend(data[1],years,'surface',2071,2100)*10.
trend_ssp119 = UT.linearTrend(data[2],years,'surface',2071,2100)*10.
trend_ssp245 = UT.linearTrend(data[3],years,'surface',2071,2100)*10.

### Calculate ensemble mean
meantrend_natural = np.nanmean(trend_natural[:,:,:],axis=0)
meantrend_ssp585 = np.nanmean(trend_ssp585[:,:,:],axis=0)
meantrend_ssp119 = np.nanmean(trend_ssp119[:,:,:],axis=0)
meantrend_ssp245 = np.nanmean(trend_ssp245[:,:,:],axis=0)

###############################################################################
###############################################################################
###############################################################################
### Loop in all climate models for PRECT
data_allp = []
for no in range(len(modelGCMs)):
    dataset = modelGCMs[no]
    scenario = scenarioall[no]
    data_allqp,lats,lons = read_primary_dataset('PRECT',dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
    data_allp.append(data_allqp)
datap = np.asarray(data_allp)

trendp_natural = UT.linearTrend(datap[0],years,'surface',2071,2100)*10.
trendp_ssp585 = UT.linearTrend(datap[1],years,'surface',2071,2100)*10.
trendp_ssp119 = UT.linearTrend(datap[2],years,'surface',2071,2100)*10.
trendp_ssp245 = UT.linearTrend(datap[3],years,'surface',2071,2100)*10.

### Calculate ensemble mean
meantrendp_natural = np.nanmean(trendp_natural[:,:,:],axis=0)
meantrendp_ssp585 = np.nanmean(trendp_ssp585[:,:,:],axis=0)
meantrendp_ssp119 = np.nanmean(trendp_ssp119[:,:,:],axis=0)
meantrendp_ssp245 = np.nanmean(trendp_ssp245[:,:,:],axis=0)

###############################################################################
###############################################################################
###############################################################################
limitt = np.arange(-0.5,0.51,0.02)
barlimt = np.round(np.arange(-0.5,0.51,0.25),2)
labelt = r'\textbf{$^{\circ}$C/decade}'

limitp = np.arange(-0.2,0.201,0.01)
barlimp = np.round(np.arange(-0.2,0.201,0.1),2)
labelp = r'\textbf{mm/day/decade}'

### Collect all data
alldata = [meantrend_natural,meantrend_ssp585,meantrend_ssp119,meantrend_ssp245,
           meantrendp_natural,meantrendp_ssp585,meantrendp_ssp119,meantrendp_ssp245]

fig = plt.figure(figsize=(10,3.5))
for mo in range(len(alldata)):                                                                                                                         
    ax = plt.subplot(2,4,mo+1)
    
    if mo < 4:
        limit = limitt
        barlim = barlimt
        label = labelt
    else:
        limit = limitp
        barlim = barlimp
        label = labelp        
    
    var = alldata[mo]

    m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
    m.drawcoastlines(color='dimgrey',linewidth=0.6)
        
    lon2,lat2 = np.meshgrid(lons,lats)
    
    circle = m.drawmapboundary(fill_color='k',color='k',linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    
    if mo < 4:
        cs1.set_cmap(cmocean.cm.balance)
    else:
        cs1.set_cmap(cmr.seasons_r)
    
    ax.annotate(r'\textbf{[%s]}' % (letters[mo]),xy=(0,0),xytext=(0.06,0.92),
              textcoords='axes fraction',color='k',fontsize=10,
              rotation=0,ha='center',va='center')
    
    if any([mo==0,mo==4]):
        ax.annotate(r'\textbf{%s}' % (allvariables[mo]),xy=(0,0),xytext=(-0.06,0.50),
                  textcoords='axes fraction',color='k',fontsize=15,
                  rotation=90,ha='center',va='center')
    if mo < 4:
        plt.title(r'\textbf{%s}' % scenarioallnames[mo],fontsize=20,color='dimgrey')
    
    if mo == 3:
        cbar_ax1 = fig.add_axes([0.92,0.59,0.013,0.25])                 
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='vertical',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=10,color='k',labelpad=4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='y', size=.01,labelsize=7)
        cbar1.outline.set_edgecolor('dimgrey')
    elif mo == 7:
        cbar_ax1 = fig.add_axes([0.92,0.14,0.013,0.25])                 
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='vertical',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=10,color='k',labelpad=5.5)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='y', size=.01,labelsize=7)
        cbar1.outline.set_edgecolor('dimgrey')
        
plt.tight_layout()
plt.subplots_adjust(hspace=0.0,wspace=0.02,right=0.9)
    
plt.savefig(directoryfigure + 'MSFigure_ANN_Trends_SSPs.png',dpi=600)
