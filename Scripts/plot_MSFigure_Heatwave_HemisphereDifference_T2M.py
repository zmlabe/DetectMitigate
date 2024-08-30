"""
Plot difference in GMST between hemispheres

Author    : Zachary M. Labe
Date      : 26 August 2024
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
import read_SPEAR_MED_LM42p2_test as LM

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['T2M']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 9
years = np.arange(2015,2100+1,1)
years_os10ye = np.arange(2031,2100+1,1)
years_LM42 = np.arange(2015,2070+1)
yearsh = np.arange(1921,2014+1,1)
yearsh_LM42 = np.arange(1921,2070+1,1)

yearsall = np.arange(1921,2100+1,1)
yearsall_LM42 = np.arange(1921,2070+1,1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/MSFigures_Heat/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
dataset_obs = 'ERA5_MEDS'
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'Globe'

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
### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_m,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_h,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osm,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Read in all of LM4.2
lat,lon,spear_LM42_all = LM.read_SPEAR_MED_LM42p2_test('/work/Zachary.Labe/Data/',variq,monthlychoice,4,np.nan,3,'all')

### Calculate anomalies
yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
climo_spear = np.nanmean(spear_h[:,yearq,:,:],axis=1)

spear_am = spear_m - climo_spear[:,np.newaxis,:,:]
spear_aosm = spear_osm - climo_spear[:,np.newaxis,:,:]
spear_aosm_10ye = spear_osm_10ye - climo_spear[:,np.newaxis,:,:]

### Hemispheres
latq_NH = np.where((lats > 0))[0]
lats_NH = lats[latq_NH]
latq_SH = np.where((lats < 0))[0]
lats_SH = lats[latq_SH]

lon2_NH,lat2_NH = np.meshgrid(lons,lats_NH)
lon2_SH,lat2_SH = np.meshgrid(lons,lats_SH)

spear_m_NH = spear_am[:,:,latq_NH,:]
spear_osm_NH = spear_aosm[:,:,latq_NH,:]
spear_osm_10ye_NH = spear_aosm_10ye[:,:,latq_NH,:]

spear_m_SH = spear_am[:,:,latq_SH,:]
spear_osm_SH = spear_aosm[:,:,latq_SH,:]
spear_osm_10ye_SH = spear_aosm_10ye[:,:,latq_SH,:]

### Calculate global means
ave_NH = UT.calc_weightedAve(spear_m_NH,lat2_NH)
ave_os_NH = UT.calc_weightedAve(spear_osm_NH,lat2_NH)
ave_os_10ye_NH = UT.calc_weightedAve(spear_osm_10ye_NH,lat2_NH)

ave_SH = UT.calc_weightedAve(spear_m_SH,lat2_SH)
ave_os_SH = UT.calc_weightedAve(spear_osm_SH,lat2_SH)
ave_os_10ye_SH = UT.calc_weightedAve(spear_osm_10ye_SH,lat2_SH)

### Calculate ensemble mean and spread
ave_NH_avgh = np.nanmean(ave_NH,axis=0)
ave_NH_minh = np.nanmin(ave_NH,axis=0)
ave_NH_maxh = np.nanmax(ave_NH,axis=0)

ave_os_NH_avgh = np.nanmean(ave_os_NH,axis=0)
ave_os_NH_minh = np.nanmin(ave_os_NH,axis=0)
ave_os_NH_maxh = np.nanmax(ave_os_NH,axis=0)

ave_os_10ye_NH_avgh = np.nanmean(ave_os_10ye_NH,axis=0)
ave_os_10ye_NH_minh = np.nanmin(ave_os_10ye_NH,axis=0)
ave_os_10ye_NH_maxh = np.nanmax(ave_os_10ye_NH,axis=0)

ave_SH_avgh = np.nanmean(ave_SH,axis=0)
ave_SH_minh = np.nanmin(ave_SH,axis=0)
ave_SH_maxh = np.nanmax(ave_SH,axis=0)

ave_os_SH_avgh = np.nanmean(ave_os_SH,axis=0)
ave_os_SH_minh = np.nanmin(ave_os_SH,axis=0)
ave_os_SH_maxh = np.nanmax(ave_os_SH,axis=0)

ave_os_10ye_SH_avgh = np.nanmean(ave_os_10ye_SH,axis=0)
ave_os_10ye_SH_minh = np.nanmin(ave_os_10ye_SH,axis=0)
ave_os_10ye_SH_maxh = np.nanmax(ave_os_10ye_SH,axis=0)

###############################################################################
###############################################################################
###############################################################################
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
###############################################################################
###############################################################################
###############################################################################
### Plot historical baseline
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
ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.4)

plt.plot(years,ave_NH_avgh,linestyle='-',linewidth=2,color='maroon',
          clip_on=False,zorder=3,label=r'\textbf{N. Hemisphere}')    
plt.plot(years,ave_SH_avgh,linestyle='--',linewidth=2,color='maroon',
          clip_on=False,zorder=3,dashes=(1,0.7),label=r'\textbf{S. Hemisphere}')

plt.plot(years,ave_os_NH_avgh,linestyle='-',linewidth=2,color='darkslategrey',
          clip_on=False,zorder=3,label=r'\textbf{N. Hemisphere}')    
plt.plot(years,ave_os_SH_avgh,linestyle='--',linewidth=2,color='darkslategrey',
          clip_on=False,zorder=3,dashes=(1,0.7),label=r'\textbf{S. Hemisphere}')

plt.plot(years[-len(years_os10ye):],ave_os_10ye_NH_avgh[-len(years_os10ye):],linestyle='-',linewidth=2,color='lightseagreen',
          clip_on=False,zorder=3,label=r'\textbf{N. Hemisphere}')    
plt.plot(years[-len(years_os10ye):],ave_os_10ye_SH_avgh[-len(years_os10ye):],linestyle='--',linewidth=2,color='lightseagreen',
          clip_on=False,zorder=3,dashes=(1,0.7),label=r'\textbf{S. Hemisphere}')

# plt.axhline(y=np.max(ave_os_10ye_NH_avgh),color='lightseagreen',linestyle='-',linewidth=0.5,clip_on=False)
# plt.axhline(y=np.max(ave_os_10ye_SH_avgh),color='lightseagreen',linestyle='-',linewidth=0.5,clip_on=False)

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.2),fancybox=True,ncol=3,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
### Plot maximum GMST for OS runs
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/MitigateHeat/'
osq = np.genfromtxt(directoryoutput + 'Max_GMST_SSP534OS_Annual.txt')[2]
os10yeq = np.genfromtxt(directoryoutput + 'Max_GMST_SSP534OS_10ye_Annual.txt')[2]
plt.axvline(x=years[int(osq)],color='darkslategrey',linewidth=2,linestyle='-',zorder=200)
plt.axvline(x=years[int(os10yeq)],color='lightseagreen',linewidth=2,linestyle='-',zorder=200)
plt.axvline(x=2040,color='darkslategrey',linewidth=1,linestyle=':',zorder=100)
plt.axvline(x=2031,color='lightseagreen',linewidth=1,linestyle=':',zorder=100)  

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2))
plt.xlim([2015,2100])
plt.ylim([0,7])

plt.text(2100,7,r'\textbf{[a]}',fontsize=10,color='k')

plt.ylabel(r'\textbf{Temperature Anomaly [$^{\circ}$C]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'MSFigure_Heatwave_TimeSeries_Hemisphere_T2M_historicalbaseline_JJA.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot historical baseline difference between hemispheres
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
ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.4)

plt.plot(years,ave_NH_avgh-ave_SH_avgh,linestyle='-',linewidth=4,color='maroon',
          clip_on=False,zorder=3,label=r'\textbf{SSP5-8.5}')    

plt.plot(years,ave_os_NH_avgh-ave_os_SH_avgh,linestyle='-',linewidth=4,color='darkslategrey',
          clip_on=False,zorder=3,label=r'\textbf{SSP5-3.4OS}')    

plt.plot(years[-len(years_os10ye):],ave_os_10ye_NH_avgh[-len(years_os10ye):]-ave_os_10ye_SH_avgh[-len(years_os10ye):],linestyle='-',linewidth=2,color='lightseagreen',
          clip_on=False,zorder=3,label=r'\textbf{SSP5-3.4OS_10ye}')    

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.2),fancybox=True,ncol=3,frameon=False,
      handlelength=0,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
### Plot maximum GMST for OS runs
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/MitigateHeat/'
osq = np.genfromtxt(directoryoutput + 'Max_GMST_SSP534OS_Annual.txt')[2]
os10yeq = np.genfromtxt(directoryoutput + 'Max_GMST_SSP534OS_10ye_Annual.txt')[2]
plt.axvline(x=years[int(osq)],color='darkslategrey',linewidth=2,linestyle='-',zorder=200)
plt.axvline(x=years[int(os10yeq)],color='lightseagreen',linewidth=2,linestyle='-',zorder=200)
plt.axvline(x=2040,color='darkslategrey',linewidth=1,linestyle=':',zorder=100)
plt.axvline(x=2031,color='lightseagreen',linewidth=1,linestyle=':',zorder=100)   

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(-18,18.1,0.25),2),np.round(np.arange(-18,18.1,0.25),2))
plt.xlim([2015,2100])
plt.ylim([0,2])

plt.text(2100,2,r'\textbf{[b]}',fontsize=10,color='k')

plt.ylabel(r'\textbf{Hemisphere Temperature Difference [$^{\circ}$C]}',
            fontsize=10,color='k')

plt.tight_layout()
plt.savefig(directoryfigure + 'MSFigure_Heatwave_TimeSeries_HemisphereDifference_T2M_historicalbaseline_JJA.png',dpi=300)
