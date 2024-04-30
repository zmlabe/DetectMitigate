"""
Calculate trend for SPEAR_MED and find relationship with LRP

Author    : Zachary M. Labe
Date      : 25 April 2023
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
from scipy.ndimage import gaussian_filter

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Model paramaters
fac = 0.80 # 0.70 training, 0.13 validation, 0.07 for testing
random_segment_seed = 71541
random_network_seed = 87750
hidden = [30,30,30]
n_epochs = 1500
batch_size = 128
lr_here = 0.0001
ridgePenalty = 0.1
actFun = 'relu'

### LRP smoothing
sigmafactor = 1.5

variablesall = ['T2M']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 9
###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/LRP/'
directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/'
directorydata2 = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/LRP/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
scenarioall = ['SSP245','SSP585']
dataset_obs = 'ERA5_MEDS'
seasons = ['annual']
slicemonthnamen = ['ANNUAL']
monthlychoice = seasons[0]
reg_name = 'Globe'
###############################################################################
###############################################################################
lenOfPicks = len(scenarioall)
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
rm_merid_mean = False
rm_annual_mean = True
###############################################################################
###############################################################################
rm_ensemble_mean = False
rm_observational_mean = False
###############################################################################
###############################################################################
calculate_anomalies = False
if calculate_anomalies == True:
    baseline = np.arange(1951,1980+1,1)
###############################################################################
###############################################################################
window = 0
ensTypeExperi = 'ENS'
###############################################################################
###############################################################################
if ensTypeExperi == 'ENS':
    if window == 0:
        rm_standard_dev = False
        ravel_modelens = False
        ravelmodeltime = False
    else:
        rm_standard_dev = True
        ravelmodeltime = False
        ravel_modelens = True
yearsall = [np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1)]
lentime = len(yearsall)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
num_of_class = lenOfPicks
###############################################################################
###############################################################################
lrpRule = 'integratedgradient'
normLRP = True
    
### Select how to save files
savename = 'ANNv2_EmissionScenarioBinary_ssp_245-585_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
print('*Filename == < %s >' % savename) 

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
def readData(directory,typemodel,savename):
    """
    Read in LRP maps
    """
    name = 'LRPMap_IG' + typemodel + '_' + savename + '.nc'
    filename = directory + name
    data = Dataset(filename)
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    lrp = data.variables['LRP'][:]
    data.close() 
    return lrp,lat,lon 

###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Create sample class labels for each model for my own testing
### Appends a twin set of classes for the random noise class 
classesl = np.empty((lenOfPicks,numOfEns,len(yearsall[0])))
for i in range(lenOfPicks):
    classesl[i,:,:] = np.full((numOfEns,len(yearsall[0])),i)  
    
if ensTypeExperi == 'ENS':
    classeslnew = np.swapaxes(classesl,0,1)
###############################################################################
###############################################################################
###############################################################################
###############################################################################       
### Read in data
def readData(directory,typemodel,savename):
    """
    Read in LRP maps
    """
    
    name = 'LRPMap_IG' + typemodel + '_' + savename + '.nc'
    filename = directory + name
    data = Dataset(filename)
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    lrp = data.variables['LRP'][:]
    data.close()
    
    return lrp,lat,lon

### Read in training and testing predictions and labels
predos = np.int_(np.genfromtxt(directorydata + 'overshootPredictedLabels_' + savename + '.txt'))[-1,:]
predos_10ye = np.int_(np.genfromtxt(directorydata + 'overshoot_10yePredictedLabels_' + savename + '.txt'))[-1,:]

### Read in LRP maps
lrp_os,lat1,lon1 = readData(directorydata2,'OS',savename)
lrp_os_10ye,lat1,lon1 = readData(directorydata2,'OS_10ye',savename)
lon2,lat2 = np.meshgrid(lon1,lat1)

### Years for each simulation
years_med = np.arange(2015,2100+1,1)

### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_os,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_os_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Slice data based on time period
# timeq_os = 2040
# timeq_os_10ye = 2031
# where_osq = np.where((years_med >= timeq_os))[0]
# where_osq_10ye = np.where((years_med >= timeq_os_10ye))[0]
yrmin_os = 2015
yrmax_os = 2100
yrmin_os_10ye = 2015
yrmax_os_10ye = 2100
where_osq = np.where((years_med >= yrmin_os) & (years_med <= yrmax_os))[0]
where_osq_10ye = np.where((years_med >= yrmin_os_10ye) & (years_med <= yrmax_os_10ye))[0]

spear_os = spear_os[:,where_osq,:,:]
spear_os_10ye = spear_os_10ye[:,where_osq_10ye,:,:]
lrp_os = lrp_os[:,where_osq,:,:]
lrp_os_10ye = lrp_os_10ye[:,where_osq_10ye,:,:]

### Calculate ensemble means
mean_spear_os = np.nanmean(spear_os,axis=0)
mean_spear_os_10ye = np.nanmean(spear_os_10ye,axis=0)

### Calculate trends 
spear_trend_os_all = calcTrend(spear_os)
spear_trend_os_10ye_all = calcTrend(spear_os_10ye)

### Calculate ensemble mean of trends
spear_trend_os = np.nanmean(spear_trend_os_all,axis=0)
spear_trend_os_10ye = np.nanmean(spear_trend_os_10ye_all,axis=0)

runs = [spear_trend_os,spear_trend_os_10ye]
modelname = ['SPEAR_MED']
modelnameyears = ['2015-2100']

### Calculate ensemble mean 
lrp_os_ig = np.nanmean(lrp_os,axis=0)
lrp_os_10ye_ig = np.nanmean(lrp_os_10ye,axis=0)

### Take means across all years
lrp_os_ig_mean = np.nanmean(lrp_os_ig[:,:,:],axis=0)
lrp_os_10ye_ig_mean = np.nanmean(lrp_os_10ye_ig[:,:,:],axis=0)

lrp_os_10ye_ig_meanS = gaussian_filter(lrp_os_ig_mean,sigma=sigmafactor,order=0)
lrp_os_10ye_ig_meanS = gaussian_filter(lrp_os_10ye_ig_mean,sigma=sigmafactor,order=0)
    
###########################################################################
###########################################################################
###########################################################################
### LRP scatter
lrp_scatter_os = lrp_os_ig_mean.ravel()
trend_scatter_os = spear_trend_os.ravel()
lrp_scatter_os_10ye = lrp_os_10ye_ig_mean.ravel()
trend_scatter_os_10ye = spear_trend_os_10ye.ravel()

sigq_os = 0.05
significance_os = np.where(lrp_scatter_os <= sigq_os)[0]
lrp_scatter_os[significance_os] = np.nan

mask_os = ~np.isnan(trend_scatter_os) & ~np.isnan(lrp_scatter_os)
slope_os, intercept_os, r_value_os, p_value_os, std_err_os = sts.linregress(trend_scatter_os[mask_os],lrp_scatter_os[mask_os])

sigq_os_10ye = 0.05
significance_os_10ye = np.where(lrp_scatter_os_10ye <= sigq_os_10ye)[0]
lrp_scatter_os_10ye[significance_os_10ye] = np.nan

mask_os_10ye = ~np.isnan(trend_scatter_os_10ye) & ~np.isnan(lrp_scatter_os_10ye)
slope_os_10ye, intercept_os_10ye, r_value_os_10ye, p_value_os_10ye, std_err_os_10ye = sts.linregress(trend_scatter_os_10ye[mask_os_10ye],lrp_scatter_os_10ye[mask_os_10ye])

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

plt.scatter(trend_scatter_os,lrp_scatter_os,marker='o',s=30,color='maroon',
            alpha=0.4,edgecolors='maroon',linewidth=0,clip_on=False)

plt.xticks(np.arange(-4,4,0.2),map(str,np.round(np.arange(-4,4,0.2),2)),fontsize=10)
plt.yticks(np.arange(-4,4,0.1),map(str,np.round(np.arange(-4,4,0.1),2)),fontsize=10)
plt.xlim([-1.6,1.6])
plt.ylim([0,0.4])

plt.title('XAI for %s-%s using %s [R=%s]' % (yrmin_os,yrmax_os,variq,np.round(r_value_os,2)),color='k',fontsize=15)

plt.xlabel(r'\textbf{Trend for SPEAR_MED_SSP534OS}',fontsize=11,color='dimgrey')
plt.ylabel(r'\textbf{ANN Relevance for SPEAR_MED_SSP534OS}',fontsize=11,color='dimgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'SCATTER_LRP-trend_SPEAR-MED-SSP534OS_v1_%s-%s.png' % (yrmin_os,yrmax_os),dpi=300)

############################################################################### 

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

plt.scatter(trend_scatter_os_10ye,lrp_scatter_os_10ye,marker='o',s=30,color='teal',
            alpha=0.4,edgecolors='teal',linewidth=0,clip_on=False)

plt.xticks(np.arange(-4,4,0.2),map(str,np.round(np.arange(-4,4,0.2),2)),fontsize=10)
plt.yticks(np.arange(-4,4,0.1),map(str,np.round(np.arange(-4,4,0.1),2)),fontsize=10)
plt.xlim([-1.6,1.6])
plt.ylim([0,0.4])

plt.title('XAI for %s-%s using %s  [R=%s]' % (yrmin_os_10ye,yrmax_os_10ye,variq,np.round(r_value_os_10ye,2)),color='k',fontsize=15)

plt.xlabel(r'\textbf{Trend for SPEAR_MED_SSP534OS_10ye}',fontsize=11,color='dimgrey')
plt.ylabel(r'\textbf{ANN Relevance for SPEAR_MED_SSP534OS_10ye}',fontsize=11,color='dimgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'SCATTER_LRP-trend_SPEAR-MED-SSP534OS_10ye_v1_%s-%s.png' % (yrmin_os_10ye,yrmax_os_10ye),dpi=300)
