"""
Plots timeseries of precipitation over Amazon

Author     : Zachary M. Labe
Date       : 8 May 2023
Version    : overshoot runs
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
from netCDF4 import Dataset
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
import cmasher as cmr
import calc_dataFunctions as df
import calc_Utilities as UT
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Directories
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
dataset_obs = 'ERA5_MEDS'
lenOfPicks = len(modelGCMs)
allDataLabels = modelGCMs
monthlychoice = 'annual'
variq = 'PRECT'
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
timeper = ['futureforcing','futureforcing']
scenarioall = ['SSP543OS','SSP543OS_10ye']
num_of_class = len(scenarioall)
shuffletype = 'GAUSS'
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
ensTypeExperi = 'ENS'
shuffletype = 'RANDGAUSS'
if window == 0:
    rm_standard_dev = False
    ravel_modelens = False
    ravelmodeltime = False
else:
    rm_standard_dev = True
    ravelmodeltime = False
    ravel_modelens = True
yearsall = [np.arange(2015+window,2100+1,1),
            np.arange(2015+window,2100+1,1)]

if variq == 'PRECT':
    yearsobs = np.arange(1950+window,2021+1,1)
else:
    yearsobs = np.arange(1979+window,2021+1,1)
lentime = len(yearsall)
###############################################################################
###############################################################################
numOfEns = 30
numOfEns_10ye = 9
dataset_inference = True
###############################################################################
###############################################################################
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model and observational/reanalysis data
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons    

data_os,lat1,lon1 = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
data_os_10ye,lat1,lon1 = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)

###############################################################################
###############################################################################
###############################################################################
### Calculate mean PRECT over Amazon
latmin = -12
latmax = 8
lonmin = 275
lonmax = 325
latq = np.where((lat1 >= latmin) & (lat1 <= latmax))[0]
lonq = np.where((lon1 >= lonmin) & (lon1 <= lonmax))[0]
latNA1 = lat1[latq]
lonNA1 = lon1[lonq]
lonNA2,latNA2 = np.meshgrid(lonNA1,latNA1)

NA_data_os1 = data_os[:,:,latq,:]
NA_data_os = NA_data_os1[:,:,:,lonq]
ave_NA_data_os = UT.calc_weightedAve(NA_data_os,latNA2)

data_os_10ye1 = data_os_10ye[:,:,latq,:]
NA_data_os_10ye = data_os_10ye1[:,:,:,lonq]
ave_NA_data_os_10ye = UT.calc_weightedAve(NA_data_os_10ye,latNA2)

### Calculate statistics for plot
max_os = np.nanmax(ave_NA_data_os,axis=0)
min_os = np.nanmin(ave_NA_data_os,axis=0)
mean_os = np.nanmean(ave_NA_data_os,axis=0)

max_os_10ye = np.nanmax(ave_NA_data_os_10ye,axis=0)
min_os_10ye = np.nanmin(ave_NA_data_os_10ye,axis=0)
mean_os_10ye = np.nanmean(ave_NA_data_os_10ye,axis=0)

minens = [min_os,min_os_10ye]
maxens = [max_os,max_os_10ye]
meanens = [mean_os,mean_os_10ye]
colors = ['teal','maroon']

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
        
plt.figure()
m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.27)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
x,y = m(lonNA2,latNA2)
cs1 = m.contourf(x,y,latNA2,np.arange(latNA2.min(),latNA2.max()+1,latNA2.max()-latNA2.min()-1),extend='both',
                  colors='r',zorder=11)
        
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
    plt.fill_between(x=yearsall[i],y1=minens[i],y2=maxens[i],facecolor=colors[i],zorder=1,
              alpha=0.25,edgecolor='none',clip_on=False)
    plt.plot(yearsall[i],meanens[i],linestyle='-',linewidth=3.5,color=colors[i],
              label=r'\textbf{%s}' % scenarioall[i],zorder=1.5,clip_on=False,
              alpha=0.75)
    
plt.plot(yearsall[0],ave_NA_data_os[-1],linestyle='--',linewidth=0.8,color='teal',
          clip_on=False,zorder=3,dashes=(1,0.3))
plt.plot(yearsall[1],ave_NA_data_os_10ye[-1],linestyle='--',linewidth=0.8,color='maroon',
          clip_on=False,zorder=3,dashes=(1,0.3))

leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
      bbox_to_anchor=(0.5,1.03),fancybox=True,ncol=3,frameon=False,
      handlelength=1,handletextpad=0.5)

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-48,48.1,0.1),2),np.round(np.arange(-48,48.1,0.1),2))
plt.xlim([2015,2100])
plt.ylim([4.2,6.2])

plt.ylabel(r'\textbf{Amazon PRECT [mm/day]}',
            fontsize=10,color='k')
plt.title(r'\textbf{Amazon %s' % variq,fontsize=15,color='dimgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'Amazon_%s_EmissionScenarios_%s.png' % (variq,monthlychoice),dpi=300)
