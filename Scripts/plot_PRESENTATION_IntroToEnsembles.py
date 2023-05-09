"""
Plot GMST for the different SPEAR emission scenarios for ANN architecture
in presentation

Author     : Zachary M. Labe
Date       : 23 February 2023
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
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

### Directories
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/ANNarchitecture/Presentation/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_Scenario']
lenOfPicks = len(modelGCMs)
monthlychoice = 'annual'
variq = 'T2M'
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
timeper = ['historicalforcing','futureforcing']
scenarioall = ['historical','SSP245']
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
yearsall = [np.arange(1929+window,2014+1,1),np.arange(2015+window,2100+1,1)]
yearsobs = np.arange(1929+window,2022+1,1)
years = np.arange(1929,2100+1,1)
###############################################################################
###############################################################################
numOfEns = 30
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
data = data_all

### Calculate historical baseline for calculating anomalies (and ensemble mean)
historical = data[0]
historicalyrs = yearsall[0]

yearhq = np.where((historicalyrs >= baseline.min()) & (historicalyrs <= baseline.max()))[0]
historicalc = np.nanmean(np.nanmean(historical[:,yearhq,:,:],axis=1),axis=0)

### Calculate anomalies
data_anoma = []
for no in range(len(modelGCMs)):
    anomq = data[no] - historicalc[np.newaxis,np.newaxis,:,:]
    data_anoma.append(anomq)

### Create long timeseries for SPEAR_MED
data_anom = [np.append(data_anoma[0],data_anoma[1],axis=1)]

### Calculate global average
lon2,lat2 = np.meshgrid(lons,lats)
aveall = []
maxens = []
minens = []
meanens = []
medianens = []
for no in range(1):
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

### Read in observations from BEST and regrid onto SPEAR
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
            spine.set_position(('outward', 7))
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
        
fig = plt.figure(figsize=(7,5))
ax = plt.subplot(111)

adjust_spines(ax, ['left', 'bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both',length=4.,width=2,which='major',color='darkgrey')
ax.tick_params(axis='x',labelsize=8,pad=1.5)
ax.tick_params(axis='y',labelsize=8,pad=1.5)

ensdata = aveall[0]

for i in  range(ensdata.shape[0]):
# for i in  range(0,3):
    plt.plot(years,ensdata[i],linestyle='-',linewidth=0.8,color='deepskyblue',
              alpha=0.5)
    
    # if i >= 28:
    #     plt.plot(years,ensdata[i],linestyle='-',linewidth=1.5,color='r',
    #               alpha=1)
    # if (i >= 22) & (i < 26):
    #     plt.plot(years,ensdata[i],linestyle='-',linewidth=0.8,color='w',
    #               alpha=1)

    # if i < 2:
    #     plt.plot(years,ensdata[i],linestyle='-',linewidth=0.8,color='deepskyblue',
    #           alpha=0.5)
    # else:
    #     plt.plot(years,ensdata[i],linestyle='-',linewidth=1.7,color='deepskyblue',
    #               alpha=1)
    # plt.plot(years,meanens[0],linestyle='-',linewidth=0.8,color='r',
    #           alpha=0.5,dashes=(1,3))
    # plt.plot(years,np.nanmean(ensdata-meanens[0],axis=0),linestyle='-',linewidth=1.7,color='r',
    #           alpha=1)
    # plt.plot(years,maxens[0],linestyle='-',linewidth=1.7,color='deepskyblue',
    #           alpha=1)
    # plt.plot(years,minens[0],linestyle='-',linewidth=1.7,color='deepskyblue',
    #           alpha=1)

# plt.plot(yearsobs[1:],aveobs[1:],linestyle='--',linewidth=3,color='gold',
#           dashes=(1,0.2),clip_on=False,zorder=30,label=r'\textbf{BEST}')

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),color='w')
plt.yticks(np.round(np.arange(-18,18.1,0.5),2),np.round(np.arange(-18,18.1,0.5),2),color='w')
plt.xlim([1930,2050])
plt.ylim([-0.5,2.5])

plt.ylabel(r'\textbf{Temperature Anomaly [$^{\circ}$C] Relative to 1951-1980}',
           fontsize=7,color='darkgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'GMST_IntroToEnsemble_s%s_10.png' % monthlychoice,
            dpi=300,transparent=False)

###############################################################################
###############################################################################
###############################################################################  
