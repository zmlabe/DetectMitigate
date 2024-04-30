"""
Plot GMST for the different SPEAR emission scenarios

Author     : Zachary M. Labe
Date       : 5 January 2022
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

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Directories
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_NATURAL','SPEAR_MED',
             'SPEAR_MED_Scenario','SPEAR_MED_Scenario','SPEAR_MED_Scenario',
             'SPEAR_MED_SSP534OS_10ye']
dataset_obs = 'ERA5_MEDS'
lenOfPicks = len(modelGCMs)
monthlychoice = 'annual'
variq = 'T2M'
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
timeper = ['historicalforcing','futureforcing','futureforcing','futureforcing','futureforcing','futureforcing','futureforcing']
scenarioall = ['historical','natural','SSP585','SSP119','SSP245','SSP534OS','SSP534OS_10ye']
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
yearsall = [np.arange(1929+window,2014+1,1),np.arange(2015+window,2100+1,1),
            np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1),
            np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1),
            np.arange(2015+window,2100+1,1)]
yearsobs = np.arange(1929+window,2021+1,1)
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
###############################################################################
### Read in model and observational/reanalysis data
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons 

def regrid(lat11,lon11,lat21,lon21,var):
    """
    Interpolated on selected grid. Reads ERA5 in as 4d with 
    [year,month,lat,lon]
    """
    
    lon1,lat1 = np.meshgrid(lon11,lat11)
    lon2,lat2 = np.meshgrid(lon21,lat21)
    
    varn_re = np.reshape(var,((lat1.shape[0]*lon1.shape[1])))   
    
    print('Completed: Start regridding process:')
    z = g((np.ravel(lat1),np.ravel(lon1)),varn_re,(lat2,lon2),method='linear')
    print('Completed: Regridding---')
    return z

### Loop in all climate models
data_all = []
for no in range(len(modelGCMs)):
    dataset = modelGCMs[no]
    scenario = scenarioall[no]
    data_allq,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
    data_all.append(data_allq)
data = np.asarray(data_all)

### Calculate historical baseline for calculating anomalies (and ensemble mean)
historical = data[0]
historicalyrs = yearsall[0]

yearhq = np.where((historicalyrs >= baseline.min()) & (historicalyrs <= baseline.max()))[0]
historicalc = np.nanmean(np.nanmean(historical[:,yearhq,:,:],axis=1),axis=0)

### Calculate anomalies
data_anom = []
for no in range(len(modelGCMs)):
    anomq = data[no] - historicalc[np.newaxis,np.newaxis,:,:]
    data_anom.append(anomq)

### Calculate global average
lon2,lat2 = np.meshgrid(lons,lats)
aveall = []
maxens = []
minens = []
meanens = []
medianens = []
for no in range(len(modelGCMs)):
    aveallq = UT.calc_weightedAve(data_anom[no],lat2)

    maxensq = np.nanmax(aveallq,axis=0)
    minensq = np.nanmin(aveallq,axis=0)
    meanensq = np.nanmean(aveallq,axis=0)
    medianensq = np.nanmedian(aveallq,axis=0)
    
    aveall.append(aveallq)
    maxens.append(maxensq)
    minens.append(minensq)
    meanens.append(meanensq)
    medianens.append(medianensq)

### Read in observations from GISTEMPv4 and regrid onto SPEAR
latobs1,lonobs1,varobs = B.read_BEST('/work/Zachary.Labe/Data/BEST/',
                                      monthlychoice,yearsobs,3,False,np.nan)
lonobs2,latobs2 = np.meshgrid(lonobs1,latobs1)

newobs = np.empty((varobs.shape[0],lats.shape[0],lons.shape[0]))
for i in range(varobs.shape[0]):
    newobs[i,:,:] = regrid(latobs1,lonobs1,lats,lons,varobs[i,:,:])  

yearhqo = np.where((yearsobs >= baseline.min()) & (yearsobs <= baseline.max()))[0]
climobs = np.nanmean(newobs[yearhqo,:,:],axis=0)
anomobs = newobs - climobs
aveobs = UT.calc_weightedAve(anomobs,lat2)

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

color = cmr.rainforest(np.linspace(0.00,0.9,len(aveall)))
for i,c in zip(range(len(aveall)),color): 
    plt.fill_between(x=yearsall[i],y1=minens[i],y2=maxens[i],facecolor=c,zorder=1,
             alpha=0.4,edgecolor='none',clip_on=False)
    plt.plot(yearsall[i],meanens[i],linestyle='-',linewidth=2,color=c,
             label=r'\textbf{%s}' % scenarioall[i],zorder=2,clip_on=False)

plt.plot(yearsobs,aveobs,linestyle='--',linewidth=0.8,color='k',
         dashes=(1,0.5),clip_on=False,zorder=30,label=r'\textbf{BEST}')

leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
      bbox_to_anchor=(0.5,1.05),fancybox=True,ncol=3,frameon=False,
      handlelength=1,handletextpad=0.5)

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2))
plt.xlim([1930,2100])
plt.ylim([-1,6])

plt.ylabel(r'\textbf{T2M Anomaly [$^{\circ}$C] Relative to 1951-1980}',
           fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'GMST_EmissionScenarios_%s.png' % monthlychoice,dpi=300)
