"""
Calculate timeseries of US means of q (Dry)

Author    : Zachary M. Labe
Date      : 24 July 2023
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

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['q']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
yearsh = np.arange(1921,2014+1,1)
years = np.arange(1921,2100+1)
years_ssp245 = np.arange(2011,2100+1)
years_os = np.arange(2011,2100+1)
years_LM42 = np.arange(1921,2070+1,1)
years_os_10ye = np.arange(2031,2100+1)
years_os_amoc = np.arange(2041,2100+1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
dataset_obs = 'ERA5_MEDS'
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
def findNearestValueIndex(array,value):
    index = (np.abs(array-value)).argmin()
    return index
###############################################################################
###############################################################################
###############################################################################
### Get data
selectGWL = 1.7
selectGWLn = '%s' % (int(selectGWL*10))
yrplus = 3
lat_bounds,lon_bounds = UT.regions('Globe')

spear_mt,lats,lons = read_primary_dataset('T2M','SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_ht,lats,lons = read_primary_dataset('T2M','SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osmt,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10yet,lats,lons = read_primary_dataset('T2M','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
climoh_speart = np.nanmean(np.nanmean(spear_ht[:,yearq,:,:],axis=1),axis=0)

spear_aht = spear_ht - climoh_speart[np.newaxis,np.newaxis,:,:]
spear_amt = spear_mt - climoh_speart[np.newaxis,np.newaxis,:,:]
spear_aosmt = spear_osmt - climoh_speart[np.newaxis,np.newaxis,:,:]
spear_aosm_10yet = spear_osm_10yet - climoh_speart[np.newaxis,np.newaxis,:,:]

### Calculate global average in SPEAR_MED
lon2,lat2 = np.meshgrid(lons,lats)
spear_ah_globeht = UT.calc_weightedAve(spear_aht,lat2)
spear_am_globeht = UT.calc_weightedAve(spear_amt,lat2)
spear_osm_globeht = UT.calc_weightedAve(spear_aosmt,lat2)
spear_osm_10ye_globeht = UT.calc_weightedAve(spear_aosm_10yet,lat2)

### Calculate GWL for ensemble means
gwl_spearht = np.nanmean(spear_ah_globeht,axis=0)
gwl_spearft = np.nanmean(spear_am_globeht,axis=0)
gwl_ost = np.nanmean(spear_osm_globeht,axis=0)
gwl_os_10yet = np.nanmean(spear_osm_10ye_globeht,axis=0)

### Combined gwl
gwl_allt = np.append(gwl_spearht,gwl_spearft,axis=0)

### Calculate overshoot times
os_yr = np.where((years_os == 2040))[0][0]
os_10ye_yr = np.where((years_os_10ye == 2031))[0][0]

### Find year of selected GWL
ssp_GWLt = findNearestValueIndex(gwl_spearft,selectGWL)
ssp_GWL = ssp_GWLt

os_first_GWLt = findNearestValueIndex(gwl_ost[:os_yr],selectGWL)
os_second_GWLt = findNearestValueIndex(gwl_ost[os_yr:],selectGWL)+(len(years)-len(years[os_yr:]))
os_first_GWL = os_first_GWLt
os_second_GWL = os_second_GWLt

os_10ye_first_GWLt = findNearestValueIndex(gwl_os_10yet[:os_yr],selectGWL) # need to account for further warming after 2031 to reach 1.5C
os_10ye_second_GWLt = findNearestValueIndex(gwl_os_10yet[os_yr:],selectGWL)+(len(years)-len(years[os_yr:])) # need to account for further warming after 2031 to reach 1.5C
os_10ye_first_GWL = os_10ye_first_GWLt
os_10ye_second_GWL = os_10ye_second_GWLt 

### Epochs for +- years around selected GWL
lat_bounds,lon_bounds = UT.regions(reg_name)

### Read in SPEAR_MED
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
freq10 = data.variables['freq10'][:]
data.close()

# ### Read in SPEAR_MED_SSP245
# directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
# nameSSP245 = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP245' + '.nc'
# filenameSSP245 = directorydatah + nameSSP245
# dataSSP245 = Dataset(filenameSSP245)
# freq10SSP245 = dataSSP245.variables['freq10'][:]
# data.close()

### Read in SPEAR_MED_SSP534OS
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
filename_os = directorydatah + name_os
data_os = Dataset(filename_os)
freq10_os = data_os.variables['freq10'][:]
data_os.close()

### Read in SPEAR_MED_SSP534OS_10ye
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os10ye = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
filename_os10ye = directorydatah + name_os10ye
data_os10ye = Dataset(filename_os10ye)
freq10_os10ye = data_os10ye.variables['freq10'][:]
data_os10ye.close()

### Read in SPEAR_MED_LM42p2_test
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_LM42 = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_LM42p2_test' + '.nc'
filename_LM42 = directorydatah + name_LM42
data_LM42 = Dataset(filename_LM42)
freq10_LM42 = data_LM42.variables['freq10'][:]
data_LM42.close()

### Calculate spatial averages
lon2us,lat2us = np.meshgrid(lonus,latus)
avg_freq10 = UT.calc_weightedAve(freq10,lat2us)
# avg_freq10SSP245 = UT.calc_weightedAve(freq10SSP245,lat2us)
avg_freq10_os = UT.calc_weightedAve(freq10_os,lat2us)
avg_freq10_os10ye = UT.calc_weightedAve(freq10_os10ye,lat2us)
avg_freq10_LM42 = UT.calc_weightedAve(freq10_LM42,lat2us)

### Calculate ensemble means
ave_avg = np.nanmean(avg_freq10,axis=0)
# ave_avgSSP245 = np.nanmean(avg_freq10SSP245,axis=0)
ave_os_avg = np.nanmean(avg_freq10_os,axis=0)
ave_os_10ye_avg = np.nanmean(avg_freq10_os10ye,axis=0)
ave_LM42_avg = np.nanmean(avg_freq10_LM42,axis=0)

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
ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.8)

plt.plot(years,ave_avg*100.,linestyle='-',linewidth=2,color='maroon',zorder=3,label=r'\textbf{SPEAR_MED_SSP585}')    
# plt.plot(years_ssp245,ave_avgSSP245*100.,linestyle='-',linewidth=1,color='salmon',zorder=3,label=r'\textbf{SPEAR_MED_SSP245}')    

plt.plot(years_os,ave_os_avg*100.,linestyle='-',linewidth=2,color='darkslategrey',zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS}')    

plt.plot(years_LM42,ave_LM42_avg*100.,linestyle='--',linewidth=1,color='r',dashes=(1,0.3),zorder=3,label=r'\textbf{SPEAR_MED_LM42p2_test}')

plt.plot(years_os_10ye,ave_os_10ye_avg*100.,linestyle='-',linewidth=2,color='teal',zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS_10ye}')    

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(0,101,10),2),np.round(np.arange(0,101,10),2))
plt.xlim([2015,2100])
plt.ylim([0,80])

plt.ylabel(r'\textbf{Frequency of qx10-Dry over CONUS [Percent]}',fontsize=7,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_qx10_%s_%s_LM42.png' % (seasons[0],reg_name),dpi=300)

###############################################################################
###############################################################################
############################################################################### 
### Calculate Tx95

### Read in SPEAR_MED
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
freq05 = data.variables['freq05'][:]
data.close()

# ### Read in SPEAR_MED_SSP245
# directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
# nameSSP245 = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP245' + '.nc'
# filenameSSP245 = directorydatah + nameSSP245
# dataSSP245 = Dataset(filenameSSP245)
# freq05SSP245 = dataSSP245.variables['freq05'][:]
# data.close()

### Read in SPEAR_MED_SSP534OS
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
filename_os = directorydatah + name_os
data_os = Dataset(filename_os)
freq05_os = data_os.variables['freq05'][:]
data_os.close()

### Read in SPEAR_MED_SSP534OS_10ye
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os10ye = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
filename_os10ye = directorydatah + name_os10ye
data_os10ye = Dataset(filename_os10ye)
freq05_os10ye = data_os10ye.variables['freq05'][:]
data_os10ye.close()

### Read in SPEAR_MED_LM42p2_test
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_LM42 = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_LM42p2_test' + '.nc'
filename_LM42 = directorydatah + name_LM42
data_LM42 = Dataset(filename_LM42)
freq05_LM42 = data_LM42.variables['freq05'][:]
data_LM42.close()

### Calculate spatial averages
lon2us,lat2us = np.meshgrid(lonus,latus)
avg_freq05 = UT.calc_weightedAve(freq05,lat2us)
# avg_freq05SSP245 = UT.calc_weightedAve(freq05SSP245,lat2us)
avg_freq05_os = UT.calc_weightedAve(freq05_os,lat2us)
avg_freq05_os10ye = UT.calc_weightedAve(freq05_os10ye,lat2us)
avg_freq05_LM42 = UT.calc_weightedAve(freq05_LM42,lat2us)

### Calculate ensemble means
ave_avg95 = np.nanmean(avg_freq05,axis=0)
# ave_avg95SSP245 = np.nanmean(avg_freq05SSP245,axis=0)
ave_os_avg95 = np.nanmean(avg_freq05_os,axis=0)
ave_os_10ye_avg95 = np.nanmean(avg_freq05_os10ye,axis=0)
ave_LM42_avg95 = np.nanmean(avg_freq05_LM42,axis=0)

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
ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.8)

plt.plot(years,ave_avg95*100.,linestyle='-',linewidth=2,color='maroon',zorder=3,label=r'\textbf{SPEAR_MED_SSP585}')    
# plt.plot(years_ssp245,ave_avg95SSP245*100.,linestyle='-',linewidth=1,color='salmon',zorder=3,label=r'\textbf{SPEAR_MED_SSP245}')  

plt.plot(years_os,ave_os_avg95*100.,linestyle='-',linewidth=2,color='darkslategrey',zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS}')    

plt.plot(years_LM42,ave_LM42_avg95*100.,linestyle='--',linewidth=1,color='r',dashes=(1,0.3),zorder=3,label=r'\textbf{SPEAR_MED_LM42p2_test}')

plt.plot(years_os_10ye,ave_os_10ye_avg95*100.,linestyle='-',linewidth=2,color='teal',zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS_10ye}')    

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(0,101,10),2),np.round(np.arange(0,101,10),2))
plt.xlim([2015,2100])
plt.ylim([0,80])

plt.ylabel(r'\textbf{Frequency of qx05-Dry over CONUS [Percent]}',fontsize=7,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_qx05-Dry_%s_%s_LM42.png' % (seasons[0],reg_name),dpi=300)
