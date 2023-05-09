"""
First version of classification problem for emission scenario using the 
overshoot scenarios

Author     : Zachary M. Labe
Date       : 13 February 2023
Version    : 1 binary for ssp245 or ssp119
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
from sklearn.metrics import accuracy_score
import cmasher as cmr
from scipy.ndimage import gaussian_filter
import calc_Utilities as UT
import scipy.stats as sts
import scipy.stats.mstats as mstats

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

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

variablesall = ['PRECT']
numOfEns = 30
numOfEns_10ye = 9
pickSMILEall = [[]] 
###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Dark_Figures/'
directorydata = '/home/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/'
directorydata2 = '/home/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/LRP/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
scenarioall = ['SSP119','SSP245']
dataset_obs = 'ERA5_MEDS'
seasons = ['annual']
monthlychoice = seasons[0]
variq = variablesall[0]
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
yearsobs = np.arange(1979+window,2021+1,1)
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
savename = 'ANNv2_EmissionScenarioBinary_ssp_119-245_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
print('*Filename == < %s >' % savename) 
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
    lrp = data.variables['LRP'][:][-1,:,:,:]
    data.close()
    
    return lrp,lat,lon

### Read in training and testing predictions and labels
predos = np.int_(np.genfromtxt(directorydata + 'overshootPredictedLabels_' + savename + '.txt'))[-1,:]
predos_10ye = np.int_(np.genfromtxt(directorydata + 'overshoot_10yePredictedLabels_' + savename + '.txt'))[-1,:]

### Read in LRP maps
lrp_os,lat1,lon1 = readData(directorydata2,'OS',savename)
lrp_os_10ye,lat1,lon1 = readData(directorydata2,'OS_10ye',savename)
lon2,lat2 = np.meshgrid(lon1,lat1)

###############################################################################
###############################################################################
############################################################################### 
### Find transition from SSP245 to SSP119
os_years = yearsall[0]
os_10ye_years = yearsall[0]

transition_osq = np.where((os_years >= 2090) & (os_years <= 2100))[0]
transition_10ye_osq = np.where((os_10ye_years >= 2066) & (os_10ye_years <= 2072))[0]

lrp_os_tran = lrp_os[transition_osq]
lrp_os_10ye_tran = lrp_os_10ye[transition_10ye_osq]

### Calculate difference in transition periods
diff_tran = np.nanmean(lrp_os_10ye_tran,axis=0) - np.nanmean(lrp_os_tran,axis=0)

### Calculate mean composites
mean_tran_os = np.nanmean(lrp_os_tran,axis=0)
mean_tran_os_10ye = np.nanmean(lrp_os_10ye_tran,axis=0)
data_tran = [mean_tran_os,mean_tran_os_10ye,diff_tran]

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of LRP means OS
limit = np.arange(-0.2,0.201,0.001)
barlim = np.round(np.arange(-0.2,0.21,0.2),2)
if variq == 'T2M':
    cmap = cmr.fusion_r
elif any([variq == 'PRECT',variq=='WA']):
    cmap = cmr.waterlily
label = r'\textbf{Relevance - [ %s ] - OS}' % (variq)

fig = plt.figure(figsize=(9,3))
for r in range(len(data_tran)):
    var = data_tran[r]
    # var = gaussian_filter(data_tran[r],sigma=sigmafactor,order=0)
    
    ax1 = plt.subplot(1,len(data_tran),r+1)
    m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
        
    # var, lons_cyclic = addcyclic(var, lon1)
    # var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    # lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    # x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='darkgrey',color='darkgrey',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    x,y = m(lon2,lat2)
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    cs1.set_cmap(cmap) 
            
    ax1.annotate(r'\textbf{%s}' % (experimentnames[r]),xy=(0,0),xytext=(0.5,1.10),
                  textcoords='axes fraction',color='w',fontsize=20,
                  rotation=0,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('darkgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'PRESENTATION_LRPComposites_%s_OS_exploreTransitions.png' % (savename),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Calculate mean relevance over the North Atlantic
latmin = 45
latmax = 60
lonmin = 310
lonmax = 340
latq = np.where((lat1 >= latmin) & (lat1 <= latmax))[0]
lonq = np.where((lon1 >= lonmin) & (lon1 <= lonmax))[0]
latNA1 = lat1[latq]
lonNA1 = lon1[lonq]
lonNA2,latNA2 = np.meshgrid(lonNA1,latNA1)

NA_lrp_os1 = lrp_os[:,latq,:]
NA_lrp_os = NA_lrp_os1[:,:,lonq]
ave_NA_os = UT.calc_weightedAve(NA_lrp_os,latNA2)

NA_lrp_os1_10ye = lrp_os_10ye[:,latq,:]
NA_lrp_os_10ye = NA_lrp_os1_10ye[:,:,lonq]
ave_NA_os_10ye = UT.calc_weightedAve(NA_lrp_os_10ye,latNA2)

### Calculate percentile for OS
lrp_osq = lrp_os.copy()
lrp_osflat = lrp_osq.reshape(lrp_os.shape[0],lrp_os.shape[1]*lrp_os.shape[2])
lrp_osflat[np.where(lrp_osflat < 0)] = np.nan

os_percf = np.zeros(lrp_osflat.shape)*np.nan
for itime in np.arange(0,lrp_osflat.shape[0]):  
    x = lrp_osflat[itime,:]
    ranks = mstats.rankdata(np.ma.masked_invalid(x))
    length = np.count_nonzero(~np.isnan(x))
    os_percf[itime,:] = (ranks-1)/length

os_perc = os_percf.reshape(lrp_os.shape[0],lrp_os.shape[1],lrp_os.shape[2])
NA_lrp_os1_perc = os_perc[:,latq,:]
NA_lrp_os_perc = NA_lrp_os1_perc[:,:,lonq]
ave_NA_os_perc = UT.calc_weightedAve(NA_lrp_os_perc,latNA2)

###############################################################################
### Calculate percentile for OS_10ye 
lrp_os_10yeq = lrp_os_10ye.copy()
lrp_osflat_10ye = lrp_os_10yeq.reshape(lrp_os_10ye.shape[0],lrp_os_10ye.shape[1]*lrp_os_10ye.shape[2])
lrp_osflat_10ye[np.where(lrp_osflat_10ye < 0)] = np.nan

os_percf_10ye = np.zeros(lrp_osflat_10ye.shape)*np.nan
for itime in np.arange(0,lrp_osflat_10ye.shape[0]):  
    x_10ye = lrp_osflat_10ye[itime,:]
    ranks_10ye = mstats.rankdata(np.ma.masked_invalid(x_10ye))
    length_10ye = np.count_nonzero(~np.isnan(x_10ye))
    os_percf_10ye[itime,:] = (ranks_10ye-1)/length_10ye 

os_perc_10ye = os_percf_10ye.reshape(lrp_os_10ye.shape[0],lrp_os_10ye.shape[1],lrp_os_10ye.shape[2])
NA_lrp_os1_perc_10ye = os_perc_10ye[:,latq,:]
NA_lrp_os_perc_10ye = NA_lrp_os1_perc_10ye[:,:,lonq]
ave_NA_os_perc_10ye = UT.calc_weightedAve(NA_lrp_os_perc_10ye,latNA2)

###############################################################################
###############################################################################
###############################################################################
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
            
### Begin plot
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.3)

plt.plot(os_years,ave_NA_os,color='turquoise',marker='o',linewidth=1,label=r'%s' % experimentnames[0],clip_on=False)
plt.plot(os_10ye_years,ave_NA_os_10ye,color='lightcoral',marker='o',linewidth=3,label=r'%s' % experimentnames[1],clip_on=False)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
              bbox_to_anchor=(0.5,-0.08),fancybox=True,ncol=5,frameon=False,
              handlelength=3,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(2015,2100+1,10),map(str,np.arange(2015,2100+1,10)),size=5.45)
plt.yticks(np.arange(-1,1.01,0.05),map(str,np.round(np.arange(-1,1.01,0.05),2)),size=6)
plt.xlim([2015,2100])   
plt.ylim([-0.15,0.15])           

plt.ylabel(r'\textbf{Relevance}',color='darkgrey',fontsize=8,labelpad=8)
plt.title(r'\textbf{LRP Relevance over the North Atlantic for %s}' % (variq),color='darkgrey',fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'PRESENTATION_%s_NorthAtlantic_Relevance.png' % (savename),dpi=300)

###############################################################################
### Begin plot
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.3)

plt.plot(os_years,ave_NA_os_perc*100,color='turquoise',marker='o',linewidth=1,label=r'%s' % experimentnames[0],clip_on=False)
plt.plot(os_10ye_years,ave_NA_os_perc_10ye*100,color='lightcoral',marker='o',linewidth=3,label=r'%s' % experimentnames[1],clip_on=False)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
              bbox_to_anchor=(0.5,-0.08),fancybox=True,ncol=5,frameon=False,
              handlelength=3,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(2015,2100+1,10),map(str,np.arange(2015,2100+1,10)),size=5.45)
plt.yticks(np.arange(0,110,10),map(str,np.round(np.arange(0,110,10),2)),size=6)
plt.xlim([2015,2100])   
plt.ylim([0,70])           

plt.ylabel(r'\textbf{Percentile of Relevance}',color='darkgrey',fontsize=8,labelpad=8)
plt.title(r'\textbf{LRP Percentile over the North Atlantic for %s}' % (variq),color='darkgrey',fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'PRESENTATION_%s_NorthAtlantic_PercentileRelevance.png' % (savename),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Calculate mean relevance over the northern North Atlantic
latmin = 55
latmax = 72
lonmin = 320
lonmax = 359.99
latq = np.where((lat1 >= latmin) & (lat1 <= latmax))[0]
lonq = np.where((lon1 >= lonmin) & (lon1 <= lonmax))[0]
latNA1 = lat1[latq]
lonNA1 = lon1[lonq]
lonNA2,latNA2 = np.meshgrid(lonNA1,latNA1)

NA_lrp_os1 = lrp_os[:,latq,:]
NA_lrp_os = NA_lrp_os1[:,:,lonq]
ave_NA_os = UT.calc_weightedAve(NA_lrp_os,latNA2)

NA_lrp_os1_10ye = lrp_os_10ye[:,latq,:]
NA_lrp_os_10ye = NA_lrp_os1_10ye[:,:,lonq]
ave_NA_os_10ye = UT.calc_weightedAve(NA_lrp_os_10ye,latNA2)

### Calculate percentile for OS
lrp_osq = lrp_os.copy()
lrp_osflat = lrp_osq.reshape(lrp_os.shape[0],lrp_os.shape[1]*lrp_os.shape[2])
lrp_osflat[np.where(lrp_osflat < 0)] = np.nan

os_percf = np.zeros(lrp_osflat.shape)*np.nan
for itime in np.arange(0,lrp_osflat.shape[0]):  
    x = lrp_osflat[itime,:]
    ranks = mstats.rankdata(np.ma.masked_invalid(x))
    length = np.count_nonzero(~np.isnan(x))
    os_percf[itime,:] = (ranks-1)/length

os_perc = os_percf.reshape(lrp_os.shape[0],lrp_os.shape[1],lrp_os.shape[2])
NA_lrp_os1_perc = os_perc[:,latq,:]
NA_lrp_os_perc = NA_lrp_os1_perc[:,:,lonq]
ave_NA_os_perc = UT.calc_weightedAve(NA_lrp_os_perc,latNA2)

###############################################################################
### Calculate percentile for OS_10ye 
lrp_os_10yeq = lrp_os_10ye.copy()
lrp_osflat_10ye = lrp_os_10yeq.reshape(lrp_os_10ye.shape[0],lrp_os_10ye.shape[1]*lrp_os_10ye.shape[2])
lrp_osflat_10ye[np.where(lrp_osflat_10ye < 0)] = np.nan

os_percf_10ye = np.zeros(lrp_osflat_10ye.shape)*np.nan
for itime in np.arange(0,lrp_osflat_10ye.shape[0]):  
    x_10ye = lrp_osflat_10ye[itime,:]
    ranks_10ye = mstats.rankdata(np.ma.masked_invalid(x_10ye))
    length_10ye = np.count_nonzero(~np.isnan(x_10ye))
    os_percf_10ye[itime,:] = (ranks_10ye-1)/length_10ye 

os_perc_10ye = os_percf_10ye.reshape(lrp_os_10ye.shape[0],lrp_os_10ye.shape[1],lrp_os_10ye.shape[2])
NA_lrp_os1_perc_10ye = os_perc_10ye[:,latq,:]
NA_lrp_os_perc_10ye = NA_lrp_os1_perc_10ye[:,:,lonq]
ave_NA_os_perc_10ye = UT.calc_weightedAve(NA_lrp_os_perc_10ye,latNA2)


###############################################################################
###############################################################################
###############################################################################
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
            
### Begin plot
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.3)

plt.plot(os_years,ave_NA_os,color='turquoise',marker='o',linewidth=1,label=r'%s' % experimentnames[0],clip_on=False)
plt.plot(os_10ye_years,ave_NA_os_10ye,color='lightcoral',marker='o',linewidth=3,label=r'%s' % experimentnames[1],clip_on=False)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
              bbox_to_anchor=(0.5,-0.08),fancybox=True,ncol=5,frameon=False,
              handlelength=3,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(2015,2100+1,10),map(str,np.arange(2015,2100+1,10)),size=5.45)
plt.yticks(np.arange(-1,1.01,0.05),map(str,np.round(np.arange(-1,1.01,0.05),2)),size=6)
plt.xlim([2015,2100])   
plt.ylim([-0.15,0.15])           

plt.ylabel(r'\textbf{Relevance}',color='darkgrey',fontsize=8,labelpad=8)
plt.title(r'\textbf{LRP Relevance over the Northern North Atlantic for %s}' % (variq),color='darkgrey',fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'PRESENTATION_%s_NorthNorthAtlantic_Relevance.png' % (savename),dpi=300)

###############################################################################
### Begin plot
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.3)

plt.plot(os_years,ave_NA_os_perc*100,color='turquoise',marker='o',linewidth=1,label=r'%s' % experimentnames[0],clip_on=False)
plt.plot(os_10ye_years,ave_NA_os_perc_10ye*100,color='lightcoral',marker='o',linewidth=3,label=r'%s' % experimentnames[1],clip_on=False)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
              bbox_to_anchor=(0.5,-0.08),fancybox=True,ncol=5,frameon=False,
              handlelength=3,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(2015,2100+1,10),map(str,np.arange(2015,2100+1,10)),size=5.45)
plt.yticks(np.arange(0,110,10),map(str,np.round(np.arange(0,110,10),2)),size=6)
plt.xlim([2015,2100])   
plt.ylim([0,70])           

plt.ylabel(r'\textbf{Percentile of Relevance}',color='darkgrey',fontsize=8,labelpad=8)
plt.title(r'\textbf{LRP Percentile over the Northern North Atlantic for %s}' % (variq),color='darkgrey',fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'PRESENTATION_%s_NorthNorthAtlantic_PercentileRelevance.png' % (savename),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Calculate mean relevance over Central Africa
latmin = 0
latmax = 15
lonmin = 0
lonmax = 40
latq = np.where((lat1 >= latmin) & (lat1 <= latmax))[0]
lonq = np.where((lon1 >= lonmin) & (lon1 <= lonmax))[0]
latNA1 = lat1[latq]
lonNA1 = lon1[lonq]
lonNA2,latNA2 = np.meshgrid(lonNA1,latNA1)

NA_lrp_os1 = lrp_os[:,latq,:]
NA_lrp_os = NA_lrp_os1[:,:,lonq]
ave_NA_os = UT.calc_weightedAve(NA_lrp_os,latNA2)

NA_lrp_os1_10ye = lrp_os_10ye[:,latq,:]
NA_lrp_os_10ye = NA_lrp_os1_10ye[:,:,lonq]
ave_NA_os_10ye = UT.calc_weightedAve(NA_lrp_os_10ye,latNA2)

### Calculate percentile for OS
lrp_osq = lrp_os.copy()
lrp_osflat = lrp_osq.reshape(lrp_os.shape[0],lrp_os.shape[1]*lrp_os.shape[2])
lrp_osflat[np.where(lrp_osflat < 0)] = np.nan

os_percf = np.zeros(lrp_osflat.shape)*np.nan
for itime in np.arange(0,lrp_osflat.shape[0]):  
    x = lrp_osflat[itime,:]
    ranks = mstats.rankdata(np.ma.masked_invalid(x))
    length = np.count_nonzero(~np.isnan(x))
    os_percf[itime,:] = (ranks-1)/length

os_perc = os_percf.reshape(lrp_os.shape[0],lrp_os.shape[1],lrp_os.shape[2])
NA_lrp_os1_perc = os_perc[:,latq,:]
NA_lrp_os_perc = NA_lrp_os1_perc[:,:,lonq]
ave_NA_os_perc = UT.calc_weightedAve(NA_lrp_os_perc,latNA2)

###############################################################################
### Calculate percentile for OS_10ye 
lrp_os_10yeq = lrp_os_10ye.copy()
lrp_osflat_10ye = lrp_os_10yeq.reshape(lrp_os_10ye.shape[0],lrp_os_10ye.shape[1]*lrp_os_10ye.shape[2])
lrp_osflat_10ye[np.where(lrp_osflat_10ye < 0)] = np.nan

os_percf_10ye = np.zeros(lrp_osflat_10ye.shape)*np.nan
for itime in np.arange(0,lrp_osflat_10ye.shape[0]):  
    x_10ye = lrp_osflat_10ye[itime,:]
    ranks_10ye = mstats.rankdata(np.ma.masked_invalid(x_10ye))
    length_10ye = np.count_nonzero(~np.isnan(x_10ye))
    os_percf_10ye[itime,:] = (ranks_10ye-1)/length_10ye 

os_perc_10ye = os_percf_10ye.reshape(lrp_os_10ye.shape[0],lrp_os_10ye.shape[1],lrp_os_10ye.shape[2])
NA_lrp_os1_perc_10ye = os_perc_10ye[:,latq,:]
NA_lrp_os_perc_10ye = NA_lrp_os1_perc_10ye[:,:,lonq]
ave_NA_os_perc_10ye = UT.calc_weightedAve(NA_lrp_os_perc_10ye,latNA2)


###############################################################################
###############################################################################
###############################################################################
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
            
### Begin plot
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.3)

plt.plot(os_years,ave_NA_os,color='turquoise',marker='o',linewidth=1,label=r'%s' % experimentnames[0],clip_on=False)
plt.plot(os_10ye_years,ave_NA_os_10ye,color='lightcoral',marker='o',linewidth=3,label=r'%s' % experimentnames[1],clip_on=False)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
              bbox_to_anchor=(0.5,-0.08),fancybox=True,ncol=5,frameon=False,
              handlelength=3,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(2015,2100+1,10),map(str,np.arange(2015,2100+1,10)),size=5.45)
plt.yticks(np.arange(-1,1.01,0.05),map(str,np.round(np.arange(-1,1.01,0.05),2)),size=6)
plt.xlim([2015,2100])   
plt.ylim([-0.15,0.15])           

plt.ylabel(r'\textbf{Relevance}',color='darkgrey',fontsize=8,labelpad=8)
plt.title(r'\textbf{LRP Relevance over Central Africa for %s}' % (variq),color='darkgrey',fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'PRESENTATION_%s_CentralAfrica_Relevance.png' % (savename),dpi=300)

###############################################################################
### Begin plot
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.3)

plt.plot(os_years,ave_NA_os_perc*100,color='turquoise',marker='o',linewidth=1,label=r'%s' % experimentnames[0],clip_on=False)
plt.plot(os_10ye_years,ave_NA_os_perc_10ye*100,color='lightcoral',marker='o',linewidth=3,label=r'%s' % experimentnames[1],clip_on=False)
    
leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
              bbox_to_anchor=(0.5,-0.08),fancybox=True,ncol=5,frameon=False,
              handlelength=3,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(2015,2100+1,10),map(str,np.arange(2015,2100+1,10)),size=5.45)
plt.yticks(np.arange(0,110,10),map(str,np.round(np.arange(0,110,10),2)),size=6)
plt.xlim([2015,2100])   
plt.ylim([0,70])           

plt.ylabel(r'\textbf{Percentile of Relevance}',color='darkgrey',fontsize=8,labelpad=8)
plt.title(r'\textbf{LRP Percentile over Central Africa for %s}' % (variq),color='darkgrey',fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'PRESENTATION_%s_Central Africa_PercentileRelevance.png' % (savename),dpi=300)
