"""
Calculate timeseries of Global means of T2M 

Author    : Zachary M. Labe
Date      : 17 November
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

variablesall = ['rh_ref']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 9
yearsh = np.arange(1921,2014+1,1)
years = np.arange(1921,2100+1)
years_ssp245 = np.arange(2011,2100+1)
years_os = np.arange(2011,2100+1)
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
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
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
lat_bounds,lon_bounds = UT.regions('Globe')

spear_mt,lats,lons = read_primary_dataset('T2M','SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_ht,lats,lons = read_primary_dataset('T2M','SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osmt,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_SSP245,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoice,'SSP245',lat_bounds,lon_bounds)
spear_osm_10yet,lats,lons = read_primary_dataset('T2M','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
climoh_speart = np.nanmean(np.nanmean(spear_ht[:,yearq,:,:],axis=1),axis=0)

spear_aht = spear_ht - climoh_speart[np.newaxis,np.newaxis,:,:]
spear_SSP245 = spear_SSP245 - climoh_speart[np.newaxis,np.newaxis,:,:]
spear_amt = spear_mt - climoh_speart[np.newaxis,np.newaxis,:,:]
spear_aosmt = spear_osmt - climoh_speart[np.newaxis,np.newaxis,:,:]
spear_aosm_10yet = spear_osm_10yet - climoh_speart[np.newaxis,np.newaxis,:,:]

### Calculate global average in SPEAR_MED
lon2,lat2 = np.meshgrid(lons,lats)
spear_ah_globeht = UT.calc_weightedAve(spear_aht,lat2)
spear_SSP245_globeht = UT.calc_weightedAve(spear_SSP245,lat2)
spear_am_globeht = UT.calc_weightedAve(spear_amt,lat2)
spear_osm_globeht = UT.calc_weightedAve(spear_aosmt,lat2)
spear_osm_10ye_globeht = UT.calc_weightedAve(spear_aosm_10yet,lat2)

### Calculate GWL for ensemble means
gwl_spearht = np.nanmean(spear_ah_globeht,axis=0)
gwl_spearSSP245 = np.nanmean(spear_SSP245_globeht,axis=0)
gwl_spearft = np.nanmean(spear_am_globeht,axis=0)
gwl_ost = np.nanmean(spear_osm_globeht,axis=0)
gwl_os_10yet = np.nanmean(spear_osm_10ye_globeht,axis=0)

### Combined gwl
gwl_allt = np.append(gwl_spearht,gwl_spearft,axis=0)

### Read in SPEAR_MED
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_mALL,latsus,lonsus = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_hALL,latsus,lonsus = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osmALL,latsus,lonsus = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10yeALL,latsus,lonsus = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Calculate anomalies
yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
climoh = np.nanmean(np.nanmean(spear_hALL[:,yearq,:,:],axis=1),axis=0)

spear_mALLa = spear_hALL - climoh[np.newaxis,np.newaxis,:,:]
spear_hALLa = spear_mALL - climoh[np.newaxis,np.newaxis,:,:]
spear_osmALLa = spear_osmALL - climoh[np.newaxis,np.newaxis,:,:]
spear_osm_10yeALLa = spear_osm_10yeALL - climoh[np.newaxis,np.newaxis,:,:]

### Mask over the USA
spear_m,maskobs = dSS.mask_CONUS(spear_mALLa,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_h,maskobs = dSS.mask_CONUS(spear_hALLa,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_osm,maskobs = dSS.mask_CONUS(spear_osmALLa,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_osm_10ye,maskobs = dSS.mask_CONUS(spear_osm_10yeALLa,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)

### Start in 2011
spear_os_2011 = np.append(spear_h[:,-3:,:,:],spear_osm[:,:,:,:],axis=1)
spear_osm_10ye_2031 = spear_osm_10ye[:,-70:,:,:]

### Calculate spatial averages
lon2us,lat2us = np.meshgrid(lonsus,latsus)
avg_freq90_h = UT.calc_weightedAve(spear_h,lat2us)
avg_freq90 = UT.calc_weightedAve(spear_m,lat2us)
avg_freq90_os = UT.calc_weightedAve(spear_os_2011,lat2us)
avg_freq90_os10ye = UT.calc_weightedAve(spear_osm_10ye_2031,lat2us)

### Calculate ensemble means
ave_avg_h = np.nanmean(avg_freq90_h,axis=0)
ave_avg = np.nanmean(avg_freq90,axis=0)
ave_os_avg = np.nanmean(avg_freq90_os,axis=0)
ave_os_10ye_avg = np.nanmean(avg_freq90_os10ye,axis=0)

############################################################################### 
############################################################################### 
############################################################################### 
### Organize variables
years = np.arange(1921,2100+1)
years_os = np.arange(2011,2100+1)
years_os10ye = np.arange(2031,2100+1)

gmst_585 = gwl_allt
heat_585 = np.append(ave_avg_h,ave_avg)

gmst_os = np.append(gwl_spearht[-4:],gwl_ost,axis=0)
heat_os = ave_os_avg

gmst_os10ye = gwl_os_10yet[-len(ave_os_10ye_avg):]
heat_os10ye = ave_os_10ye_avg

### Calculate decadal averages
dec_gmst_585 = []
dec_heat_585 = []
dec_years_585 = []
for y in range(0,len(gmst_585),10):
    print(y,y+10,years[y:y+10])
    
    dec_gmst_585q = np.nanmean(gmst_585[y:y+10])
    dec_gmst_585.append(dec_gmst_585q)
    
    dec_heat_585q = np.nanmean(heat_585[y:y+10])
    dec_heat_585.append(dec_heat_585q)
    
    dec_years_585q = years[y+10-1]
    dec_years_585.append(dec_years_585q)
###############################################################################      
dec_gmst_os = []
dec_heat_os = []
dec_years_os = []
for y in range(0,len(gmst_os),10):
    print(y,y+10,years[y:y+10])
    
    dec_gmst_osq = np.nanmean(gmst_os[y:y+10])
    dec_gmst_os.append(dec_gmst_osq)
    
    dec_heat_osq = np.nanmean(heat_os[y:y+10])
    dec_heat_os.append(dec_heat_osq)
    
    dec_years_osq = years_os[y+10-1]
    dec_years_os.append(dec_years_osq)
###############################################################################      
dec_gmst_os10ye = []
dec_heat_os10ye = []
dec_years_os10ye = []
for y in range(0,len(gmst_os10ye),10):
    print(y,y+10,years[y:y+10])
    
    dec_gmst_os10yeq = np.nanmean(gmst_os10ye[y:y+10])
    dec_gmst_os10ye.append(dec_gmst_os10yeq)
    
    dec_heat_os10yeq = np.nanmean(heat_os10ye[y:y+10])
    dec_heat_os10ye.append(dec_heat_os10yeq)
    
    dec_years_os10yeq = years_os10ye[y+10-1]
    dec_years_os10ye.append(dec_years_os10yeq)

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

plt.plot(dec_gmst_585,dec_heat_585,linestyle='-',linewidth=2,color='maroon',zorder=3,label=r'\textbf{SPEAR_MED_SSP585}',marker='o',markersize=5)  

plt.plot(dec_gmst_os,dec_heat_os,linestyle='-',linewidth=2,color='darkslategrey',zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS}',marker='o',markersize=5)  
plt.plot(dec_gmst_os[-1],dec_heat_os[-1],linestyle='-',linewidth=2,color='k',zorder=3,marker='o',markersize=7)   

plt.plot(dec_gmst_os10ye,dec_heat_os10ye,linestyle='-',linewidth=2,color='teal',zorder=3,label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',marker='o',markersize=5) 
plt.plot(dec_gmst_os10ye[-1],dec_heat_os10ye[-1],linestyle='-',linewidth=2,color='k',zorder=3,marker='o',markersize=7)    

leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
      bbox_to_anchor=(0.5,1.14),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(-10,11,0.5),np.arange(-10,11,0.5))
plt.yticks(np.round(np.arange(-5,5.1,0.5),2),np.round(np.arange(-5,5.1,0.5),2))
plt.xlim([-1,5])
plt.ylim([-5,1])

# plt.text(0.78,20.5,r'\textbf{2020}',fontsize=7,color='darkgrey')
# plt.text(1.21,16,r'\textbf{2100}',fontsize=7,color='k')
# plt.text(1.74,22.5,r'\textbf{2100}',fontsize=7,color='k')
# plt.text(2.51,38,r'\textbf{2100}',fontsize=7,color='k')

plt.xlabel(r'\textbf{GMST relative to 1921-1950}',fontsize=7,color='k')
plt.ylabel(r'\textbf{RH over US [Percent]}',fontsize=7,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'TimeSeries_Scale_GMST-RH_%s_%s_byDecade.png' % (seasons[0],reg_name),dpi=300)
