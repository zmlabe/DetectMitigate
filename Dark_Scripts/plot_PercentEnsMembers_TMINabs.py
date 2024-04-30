"""
Calculate probability of members above threshold

Author    : Zachary M. Labe
Date      : 24 October 2023
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

variablesall = ['TMINabs']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
yearsf = np.arange(2015,2100+1)
yearsh = np.arange(1921,2014+1,1)
yearsall = np.arange(1921,2100+1,1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED','SPEAR_MED_Scenario','SPEAR_MED_Scenario']
modelnames = np.repeat(['SSP5-8.5','SSP5-3.4OS','SSP5-3.4OS_10ye'],6)
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'US'
periods_30 = ['1981-2010','2011-2040','2041-2070','2071-2100']
periods_15 = [[2015,2029],[2030,2044],[2045,2059],[2060,2074],[2075,2089],[2090,2100]]

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

### Meshgrid and mask by CONUS
lon2,lat2 = np.meshgrid(lons,lats)

if reg_name == 'US':
    data_obsnan = np.full([1,lats.shape[0],lons.shape[0]],np.nan)
    spear_mask,data_obsnan = dSS.mask_CONUS(spear_m,data_obsnan,'MEDS',lat_bounds,lon_bounds)
    spear_h_mask,data_obsnan = dSS.mask_CONUS(spear_h,data_obsnan,'MEDS',lat_bounds,lon_bounds)
    spear_os_mask,data_obsnan = dSS.mask_CONUS(spear_osm,data_obsnan,'MEDS',lat_bounds,lon_bounds)
    spear_os10ye_mask,data_obsnan = dSS.mask_CONUS(spear_osm_10ye,data_obsnan,'MEDS',lat_bounds,lon_bounds)
else:
    spear_mask = spear_m
    
### Calculate threshold for each grid point
yearq = np.where((yearsh >= 1981) & (yearsh <= 2010))[0]
climoh_spear = spear_h_mask[:,yearq,:,:]
climoh_spear_mean = np.nanmax(climoh_spear[:,:,:,:],axis=1) # max of the TMINabs over 1981-2010
climoh_spear_mean_EnsMax = np.nanmax(climoh_spear_mean[:,:,:],axis=0) # max ensemble member

ssp585 = np.full((len(periods_15),lats.shape[0],lons.shape[0]),np.nan)
ssp585_os = np.full((len(periods_15),lats.shape[0],lons.shape[0]),np.nan)
ssp585_os10ye = np.full((len(periods_15),lats.shape[0],lons.shape[0]),np.nan)
for yrp in range(len(periods_15)):
    for i in range(lats.shape[0]):
        for j in range(lons.shape[0]):
            location_climo = climoh_spear_mean_EnsMax[i,j]
            
            ### Slice each time period
            yrq = np.where((yearsf >= periods_15[yrp][0]) & (yearsf <= periods_15[yrp][1]))[0]
            spearTime = spear_mask[:,yrq,i,j]
            spearTime_ravel = spearTime.ravel()
            
            spearTime_os = spear_os_mask[:,yrq,i,j]
            spearTime_os_ravel = spearTime_os.ravel()
            
            spearTime_os10ye = spear_os10ye_mask[:,yrq,i,j]
            spearTime_os10ye_ravel = spearTime_os10ye.ravel()

            ### Calculate probability of exceeding threshold for years times ens
            if len(np.where(spearTime_ravel > location_climo)[0]) != 0.:
                ssp585[yrp,i,j] = (len(np.where(spearTime_ravel > location_climo)[0])/len(spearTime_ravel)) * 100.
                ssp585_os[yrp,i,j] = (len(np.where(spearTime_os_ravel > location_climo)[0])/len(spearTime_os_ravel)) * 100.
                ssp585_os10ye[yrp,i,j] = (len(np.where(spearTime_os10ye_ravel > location_climo)[0])/len(spearTime_os10ye_ravel)) * 100.
            else:
                ssp585[yrp,i,j] = 0.
                ssp585_os[yrp,i,j] = 0.
                ssp585_os10ye[yrp,i,j] = 0.
    print(yrq,len(spearTime_ravel),len(spearTime_os_ravel),len(spearTime_os10ye_ravel))
 
### Get data ready to plot
preparedata = [ssp585,ssp585_os,ssp585_os10ye]
 
###############################################################################
###############################################################################
###############################################################################
### Plot differences in land fraction
fig = plt.figure(figsize=(10,5))

label = r'\textbf{Probability of exceeding TMIN(1981-2010) [\%]}'
limit = np.arange(0,101,2)
barlim = np.round(np.arange(0,101,10),2)
cmap = cmr.fall_r

for i in range(len(periods_15)*3):
    ax = plt.subplot(3,len(periods_15),i+1)
    
    if i < len(periods_15):
        var = preparedata[0][i]
    elif i >= len(periods_15) and i < (len(periods_15)*2):
        var = preparedata[1][i-len(periods_15)]
    elif i >= (len(periods_15)*2) and i < (len(periods_15)*3):
        var = preparedata[2][i-(len(periods_15)*2)]
    lat1 = lats
    lon1 = lons
    
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
                area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=1)
    m.drawstates(color='darkgrey',linewidth=0.5)
    m.drawcountries(color='darkgrey',linewidth=0.5)

    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgrey',
                      linewidth=0.7)
    circle.set_clip_on(False)
        
    lon2,lat2 = np.meshgrid(lon1,lat1)
    
    cs1 = m.contourf(lon2,lat2,var,limit,extend='neither',latlon=True)
    
    cs1.set_cmap(cmap)
    
    ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(0.02,1.06),
              textcoords='axes fraction',color='k',fontsize=7,
              rotation=0,ha='center',va='center')
    
    if any([i==0,i==6,i==12]):
        ax.annotate(r'\textbf{%s}' % (modelnames[i]),xy=(0,0),xytext=(-0.12,0.5),
                  textcoords='axes fraction',color='dimgrey',fontsize=11,
                  rotation=90,ha='center',va='center')     
    if i < 6:
        ax.annotate(r'\textbf{%s-%s}' % (periods_15[i][0],periods_15[i][1]),xy=(0,0),xytext=(0.5,1.2),
                  textcoords='axes fraction',color='k',fontsize=15,
                  rotation=0,ha='center',va='center')       
    
cbar_ax1 = fig.add_axes([0.32,0.1,0.4,0.02])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='neither',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=8,color='k',labelpad=3)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=-0.5)
        
plt.savefig(directoryfigure + 'PercentEnsMembers_TMINabs.png',dpi=300)
