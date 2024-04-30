"""
Calculate manuscript figure of US TMAX extremes

Author    : Zachary M. Labe
Date      : 10 October 2023
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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['TMAX']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 9
yearsh = np.arange(1921,2014+1,1)
years = np.arange(1921,2100+1)
years_ssp245 = np.arange(2015,2100+1)
years_os = np.arange(2015,2100+1)
years_os10ye = np.arange(2031,2100+1)
yearsall = [years_ssp245,years,years_os,years_os10ye]

minb = 1981
maxb = 2010
junedays = np.arange(0,30,1)
julydays = np.arange(0,31,1)
augustdays = np.arange(0,31,1)
dayslength = len(junedays) + len(julydays) + len(augustdays)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/MS_Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'US'

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
def calcBandwidth(data):
    ### Bandwidth cross-validation (jakevdp.github.io/blog/2013/12/01/kernel-density-esimation/)
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth':np.linspace(0.1,100.0,100)},cv = 20)
    grid.fit(data[:,None])
    bandwidth = grid.best_params_['bandwidth']
    print('Bandwidth is ---> %s!' % bandwidth)
    return bandwidth
###############################################################################
###############################################################################
###############################################################################
def kde_sklearn(x,grid,bandwidth,**kwargs):
    """kerndel density estimation with scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth,**kwargs)
    kde_skl.fit(x[:,np.newaxis])
    
    ### score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(grid[:,np.newaxis])
    
    return np.exp(log_pdf)
###############################################################################
###############################################################################
###############################################################################
### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)

### Read in SPEAR_MED
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
count90 = data.variables['count90'][:]
data.close()

### Read in SPEAR_MED_SSP245
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
nameSSP245 = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP245' + '.nc'
filenameSSP245 = directorydatah + nameSSP245
dataSSP245 = Dataset(filenameSSP245)
count90SSP245 = dataSSP245.variables['count90'][:,-len(years_ssp245):,:,:]
data.close()

### Read in SPEAR_MED_SSP534OS
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
filename_os = directorydatah + name_os
data_os = Dataset(filename_os)
count90_os = data_os.variables['count90'][:,-len(years_os):,:,:]
data_os.close()

### Read in SPEAR_MED_SSP534OS_10ye
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os10ye = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
filename_os10ye = directorydatah + name_os10ye
data_os10ye = Dataset(filename_os10ye)
count90_os10ye = data_os10ye.variables['count90'][:]
data_os10ye.close()

### Calculate spatial averages
lon2us,lat2us = np.meshgrid(lonus,latus)
avg_count90 = UT.calc_weightedAve(count90,lat2us)
avg_count90SSP245 = UT.calc_weightedAve(count90SSP245,lat2us)
avg_count90_os = UT.calc_weightedAve(count90_os,lat2us)
avg_count90_os10ye = UT.calc_weightedAve(count90_os10ye,lat2us)

### Calculate ensemble means
ave_avg = np.nanmean(avg_count90,axis=0)
ave_avgSSP245 = np.nanmean(avg_count90SSP245,axis=0)
ave_os_avg = np.nanmean(avg_count90_os,axis=0)
ave_os_10ye_avg = np.nanmean(avg_count90_os10ye,axis=0)

### Max/min statistics
ssp585_max = np.nanmax(avg_count90,axis=0)
ssp585_min = np.nanmin(avg_count90,axis=0)

ssp245_max = np.nanmax(avg_count90SSP245,axis=0)
ssp245_min = np.nanmin(avg_count90SSP245,axis=0)

os_max = np.nanmax(avg_count90_os,axis=0)
os_min = np.nanmin(avg_count90_os,axis=0)

os10ye_max = np.nanmax(avg_count90_os10ye,axis=0)
os10ye_min = np.nanmin(avg_count90_os10ye,axis=0)

### Collect plots for timeseries
meanens = [ave_avgSSP245,ave_avg,ave_os_avg,ave_os_10ye_avg]
maxens = [ssp245_max,ssp585_max,os_max,os10ye_max]
minens = [ssp245_min,ssp585_min,os_min,os10ye_min]
scenarioalln = ['SSP2-4.5','SSP5-8.5','SSP5-3.4OS','SSP5-3.4OS_10ye']

###############################################################################
###############################################################################
###############################################################################  
### Slice epochs
yearq = np.where((years >= minb) & (years <= maxb))[0]
climoEpoch = avg_count90[:,yearq].ravel()

yearq_end = np.where((years >= 2071) & (years <= 2100))[0]
yearq_ssp245_end = np.where((years_ssp245 >= 2071) & (years_ssp245 <= 2100))[0]
yearq_os_end = np.where((years_os >= 2071) & (years_os <= 2100))[0]
yearq_os10ye_end = np.where((years_os10ye >= 2071) & (years_os10ye <= 2100))[0]

ssp585_end = avg_count90[:,yearq_end].ravel()
ssp245_end = avg_count90SSP245[:,yearq_ssp245_end].ravel()
os_end = avg_count90_os[:,yearq_os_end].ravel()
os10ye_end = avg_count90_os10ye[:,yearq_os10ye_end].ravel()

### Calculate frequency 
climo_freq = (climoEpoch/dayslength) *100.
ssp585_freq = (ssp585_end/dayslength) *100.
ssp245_freq = (ssp245_end/dayslength) *100.
os_freq = (os_end/dayslength) *100.
os10ye_freq = (os10ye_end/dayslength) *100.

### Calculate bandwidth for KDE
climo_band = calcBandwidth(climo_freq)
ssp585_band = calcBandwidth(ssp585_freq)
ssp245_band = calcBandwidth(ssp245_freq)
os_band = calcBandwidth(os_freq)
os10ye_band = calcBandwidth(os10ye_freq)

### Calculate KDE
grid = np.arange(0,101,1)
climo_kde = kde_sklearn(climo_freq,grid,climo_band)
climo_kde[np.where(climo_kde <= 0.0001)] = np.nan

ssp585_kde = kde_sklearn(ssp585_freq,grid,ssp585_band)
ssp585_kde[np.where(ssp585_kde <= 0.0001)] = np.nan

ssp245_kde = kde_sklearn(ssp245_freq,grid,ssp245_band)
ssp245_kde[np.where(ssp245_kde <= 0.0001)] = np.nan

os_kde = kde_sklearn(os_freq,grid,os_band)
os_kde[np.where(os_kde <= 0.0001)] = np.nan

os10ye_kde = kde_sklearn(os10ye_freq,grid,os10ye_band)
os10ye_kde[np.where(os10ye_kde <= 0.0001)] = np.nan

allpdfs = [climo_kde,ssp585_kde,ssp245_kde,os_kde,os10ye_kde]
scenarioalln2 = ['1981-2010 SPEAR_MED','SSP5-8.5','SSP2-4.5','SSP5-3.4OS','SSP5-3.4OS_10ye']

### Statistical tests
overshoots_stat = sts.ks_2samp(os_freq,os10ye_freq,alternative='two-sided',method='auto')
climo_10ye_stat = sts.ks_2samp(climo_freq,os10ye_freq,alternative='two-sided',method='auto')

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
        
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(121)

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

color = [cmr.dusk(0.01),cmr.sapphire(0.5),cmr.dusk(0.8)]
for i,c in zip(range(1,len(meanens),1),color): 
    if i == 0:
        c = 'dimgrey'
    plt.fill_between(x=yearsall[i],y1=minens[i],y2=maxens[i],facecolor=c,zorder=1,
             alpha=0.4,edgecolor='none')
    if scenarioalln[i] == 'SSP5-3.4OS_10ye':
        plt.plot(yearsall[i],meanens[i],linestyle='--',linewidth=2,color=c,
                 label=r'\textbf{%s}' % scenarioalln[i],zorder=2,dashes=(1,.3))
    elif scenarioalln[i] == 'SSP5-3.4OS':
        plt.plot(yearsall[i],meanens[i],linestyle='-',linewidth=2,color=c,
                 label=r'\textbf{%s}' % scenarioalln[i],zorder=2)
    else:
        plt.plot(yearsall[i],meanens[i],linestyle='-',linewidth=4,color=c,
                 label=r'\textbf{%s}' % scenarioalln[i],zorder=2)
    
leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
      bbox_to_anchor=(0.5,0.06),fancybox=True,ncol=3,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(0,101,10),2),np.round(np.arange(0,101,10),2))
plt.xlim([2015,2100])
plt.ylim([0,80])

plt.text(2015,81,r'\textbf{[a]}',fontsize=12,color='dimgrey')
plt.xlabel(r'\textbf{Years}',fontsize=7,color='k')
plt.ylabel(r'\textbf{Count of days over Tx90}',fontsize=7,color='k')

###############################################################################
###############################################################################
###############################################################################  
### PDF graphs
ax = plt.subplot(122)

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

color = [cmr.dusk(0.01),'darkgrey',cmr.dusk(0.3),cmr.sapphire(0.5),cmr.dusk(0.8)]
for i,c in zip(range(len(allpdfs)),color): 
    if i == 0:
        plt.plot(grid,allpdfs[i],linestyle='--',color=color[i],label=r'\textbf{%s}' % scenarioalln2[i],
                 linewidth=4,clip_on=False,dashes=(1,0.3))
    else:
        plt.plot(grid,allpdfs[i],linestyle='-',color=color[i],label=r'\textbf{%s}' % scenarioalln2[i],
                 linewidth=2.5,clip_on=False)
    
leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
      bbox_to_anchor=(0.625,1),fancybox=True,ncol=2,frameon=False,
      handlelength=2,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.round(np.arange(0,101,10),2),np.round(np.arange(0,101,10),2))
plt.yticks(np.round(np.arange(0,0.11,0.01),3),np.round(np.arange(0,0.11,0.01),3))
plt.xlim([0,100])
plt.ylim([0,0.1])

plt.text(0,0.101,r'\textbf{[b]}',fontsize=12,color='dimgrey')
plt.xlabel(r'\textbf{Frequency of Tx90 Days Per Summer}',fontsize=7,color='k')
plt.ylabel(r'\textbf{PDF}',fontsize=7,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'MSFigure_Heatwave_TimeSeries-US-TMAX_PDFs_v1.png',dpi=300)
