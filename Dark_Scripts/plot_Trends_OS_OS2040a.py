"""
Plot trends in last 30 years for OS

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
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
monthq = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_SSP370_OS2040a']
dataset_obs = 'ERA5_MEDS'
lenOfPicks = len(modelGCMs)
monthlychoice = 'annual'
allvariables = ['Temperature','Temperature','Precipitation','Precipitation']
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
timeper = ['futureforcing','futureforcing']
scenarioall = ['SSP534OS','SSP370_OS2040a']
scenarioallnames = ['SPEAR_MED_SSP534OS','SPEAR_MED_SSP370_OS2040a']
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

trend_os = UT.linearTrend(data[0],years,'surface',2041,2100)*10.
trend_os10ye = UT.linearTrend(data[1],years,'surface',2041,2100)*10.

### Calculate ensemble mean
meantrend_os = np.nanmean(trend_os[:,:,:],axis=0)
meantrend_os10ye = np.nanmean(trend_os10ye[:,:,:],axis=0)

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

trendp_os = UT.linearTrend(datap[0],years,'surface',2071,2100)*10.
trendp_os10ye = UT.linearTrend(datap[1],years,'surface',2071,2100)*10.

### Calculate ensemble mean
meantrendp_os = np.nanmean(trendp_os[:,:,:],axis=0)
meantrendp_os10ye = np.nanmean(trendp_os10ye[:,:,:],axis=0)

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
alldata = [meantrend_os,meantrend_os10ye,
           meantrendp_os,meantrendp_os10ye]

fig = plt.figure(figsize=(10,5))
for mo in range(len(alldata)):                                                                                                                         
    ax = plt.subplot(2,2,mo+1)
    
    if mo < 2:
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
    
    if mo < 2:
        cs1.set_cmap(cmocean.cm.balance)
    else:
        cs1.set_cmap(cmr.seasons_r)
    
    ax.annotate(r'\textbf{[%s]}' % (letters[mo]),xy=(0,0),xytext=(0.06,0.90),
              textcoords='axes fraction',color='k',fontsize=10,
              rotation=0,ha='center',va='center')
    
    if any([mo==0,mo==2]):
        ax.annotate(r'\textbf{%s}' % (allvariables[mo]),xy=(0,0),xytext=(-0.05,0.50),
                  textcoords='axes fraction',color='k',fontsize=15,
                  rotation=90,ha='center',va='center')
    if mo < 2:
        plt.title(r'\textbf{%s}' % scenarioallnames[mo],fontsize=20,color='dimgrey')
    
    if mo == 1:
        cbar_ax1 = fig.add_axes([0.92,0.58,0.013,0.25])                 
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='vertical',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=10,color='k',labelpad=4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='y', size=.01,labelsize=7)
        cbar1.outline.set_edgecolor('dimgrey')
    elif mo == 3:
        cbar_ax1 = fig.add_axes([0.92,0.135,0.013,0.25])                 
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='vertical',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=10,color='k',labelpad=5.5)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='y', size=.01,labelsize=7)
        cbar1.outline.set_edgecolor('dimgrey')
        
plt.tight_layout()
plt.subplots_adjust(hspace=0.02,wspace=0.02,right=0.9)
    
plt.savefig(directoryfigure + 'Trends_OS2040a.png',dpi=600)
