"""
Calculate a map of the warmest year for the ensemble mean in the overshoot runs
for the count of summer temperature extremes but with running mean

Author    : Zachary M. Labe
Date      : 31 July 2023
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import cmasher as cmr
import palettable.cubehelix as cm
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import sys
import scipy.stats as sts
import pandas as pd

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['TMAX']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
years = np.arange(2015,2100+1)
rolling_years = 5 # years

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','SSP5-34OS minus SSP5-34OS_10ye']
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'Globe'
varcount = 'count90'
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

###############################################################################
###############################################################################
###############################################################################
### Read in data for OS daily extremes
### Read in SPEAR_MED
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
spear = data.variables[varcount][:]
data.close()

### Select only 1921 to 2015
years15 = np.arange(1921,2014+1,1)
yearsall = np.arange(1921,2100+1,1)
spearslice = np.nanmean(spear[:,:len(years15),:,:],axis=0)

### Read in SPEAR_MED_SSP534OS
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
filename_os = directorydatah + name_os
data_os = Dataset(filename_os)
spear_osm = data_os.variables[varcount][:]
lats = data_os.variables['lat'][:]
lons = data_os.variables['lon'][:]
data_os.close()

### Read in SPEAR_MED_SSP534OS_10ye
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os10ye = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
filename_os10ye = directorydatah + name_os10ye
data_os10ye = Dataset(filename_os10ye)
spear_osm_10ye = data_os10ye.variables[varcount][:]
data_os10ye.close()
lon2,lat2 = np.meshgrid(lons,lats)

### Calculate the ensemble mean and only use 2015-2031
spear_osm_meann = np.nanmean(spear_osm,axis=0)[-len(years):,:,:]
spear_osm_10ye_meanq = np.nanmean(spear_osm_10ye,axis=0)

### Add on years for 2015-2030
spear_osm_10ye_meann = np.append(spear_osm_meann[:(len(years)-len(spear_osm_10ye_meanq)),:,:],spear_osm_10ye_meanq,axis=0)

### Add on years 1921-2014
spear_osm_mean = np.append(spearslice,spear_osm_meann,axis=0)
spear_osm_10ye_mean = np.append(spearslice,spear_osm_10ye_meann,axis=0)

### Locate max of warmest year at each grid point
map_os = np.empty((lats.shape[0],lons.shape[0]))
map_os_10ye = np.empty((lats.shape[0],lons.shape[0]))
map_os_index = np.empty((lats.shape[0],lons.shape[0]))
map_os_10ye_index = np.empty((lats.shape[0],lons.shape[0]))
for i in range(lats.shape[0]):
    for j in range(lons.shape[0]):
        gridpoint_osr = spear_osm_mean[:,i,j]
        gridpoint_osr_10ye = spear_osm_10ye_mean[:,i,j]
        
        ### Calculate running mean
        gridpoint_os = pd.Series(gridpoint_osr).rolling(window=rolling_years).mean().values
        gridpoint_os_10ye = pd.Series(gridpoint_osr_10ye).rolling(window=rolling_years).mean().values
        
        gridpoint_os = gridpoint_os[-len(years):]
        gridpoint_os_10ye = gridpoint_os_10ye[-len(years):]
        
        maxindex_os = np.nanargmax(gridpoint_os)
        maxindex_os_10ye = np.nanargmax(gridpoint_os_10ye)
        map_os_index[i,j] = maxindex_os 
        map_os_10ye_index[i,j] = maxindex_os_10ye 
        
        yrlabel_os = years[maxindex_os]
        yrlabel_os_10ye = years[maxindex_os_10ye]
        
        map_os[i,j] = yrlabel_os
        map_os_10ye[i,j] = yrlabel_os_10ye
        
warmestyear = [map_os,map_os_10ye]

### Calculate difference in warmest year
differenceWarmestYear = map_os - map_os_10ye

### Benefit calculation
diffbenefit = np.empty((lats.shape[0],lons.shape[0]))
diffbenefit[:] = np.nan
for i in range(lats.shape[0]):
    for j in range(lons.shape[0]):
        gridpoint_osr = spear_osm_mean[:,i,j]
        gridpoint_osr_10ye = spear_osm_10ye_mean[:,i,j]
        
        ### Calculate running mean then slice 2015-2100
        gridpoint_os = pd.Series(gridpoint_osr).rolling(window=rolling_years).mean().values
        gridpoint_os_10ye = pd.Series(gridpoint_osr_10ye).rolling(window=rolling_years).mean().values
        
        gridpoint_os = gridpoint_os[-len(years):]
        gridpoint_os_10ye = gridpoint_os_10ye[-len(years):]
        
        max_os = np.nanmax(gridpoint_os)
        max_os_10ye = np.nanmax(gridpoint_os_10ye)
        maxindex_os = np.nanargmax(gridpoint_os)
        maxindex_os_10ye = np.nanargmax(gridpoint_os_10ye)
        
        wherebegin_os = np.where((years == 2040))[0][0]
        wherebegin_os_10ye  = np.where((years == 2031))[0][0]
        if (maxindex_os_10ye >= wherebegin_os_10ye) and (maxindex_os >= wherebegin_os):
            
            maxAfterOS_10ye = np.nanmax(gridpoint_os_10ye[:])
            IndexmaxAfterOS_10ye = np.nanargmax(gridpoint_os_10ye[:])
            
            IndexmaxAfterOS = np.nanargmax(gridpoint_os[:])
            
            if len(np.where(gridpoint_os[IndexmaxAfterOS:] <= maxAfterOS_10ye)[0]) > 0:
                
                minwherebelowMax = np.min(np.where(gridpoint_os[IndexmaxAfterOS:] <= maxAfterOS_10ye)[0]) + IndexmaxAfterOS
                diffbenefit[i,j] = IndexmaxAfterOS_10ye - minwherebelowMax
          
# plt.figure()
# plt.plot(gridpoint_os)
# plt.plot(gridpoint_os_10ye)
# plt.axhline(np.max(gridpoint_os_10ye),color='k')
# sys.exit()

### Evaluate the hemispheres
latqS = np.where(lats < 0)[0]
latqN = np.where(lats > 0)[0]

latsS = lats[latqS]
latsN = lats[latqN]

lon2S,lat2S = np.meshgrid(lons,latsS)
lon2N,lat2N = np.meshgrid(lons,latsN)

spear_osm_meanS = spear_osm_mean[:,latqS,:]
spear_osm_10ye_meanS = spear_osm_10ye_mean[:,latqS,:]
spear_osm_meanN = spear_osm_mean[:,latqN,:]
spear_osm_10ye_meanN = spear_osm_10ye_mean[:,latqN,:]

os_South = UT.calc_weightedAve(spear_osm_meanS,lat2S)
os_North = UT.calc_weightedAve(spear_osm_meanN,lat2N)

os_10ye_South = UT.calc_weightedAve(spear_osm_10ye_meanS,lat2S)
os_10ye_North = UT.calc_weightedAve(spear_osm_10ye_meanN,lat2N)

### Check hemisphere plotsd
plt.figure()
plt.plot(os_South)
plt.plot(os_10ye_South)

plt.figure()
plt.plot(os_North)
plt.plot(os_10ye_North)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of warmst year
limit = np.arange(2015,2101,5)
barlim = np.round(np.arange(2015,2101,10),2)
# cmap = cm.classic_16.mpl_colormap
cmap = cmr.dusk
label = r'\textbf{Warmest %s [Ensemble Mean]}' % seasons[0]
fig = plt.figure(figsize=(9,3))
for r in range(len(warmestyear)):
    
    var = warmestyear[r]
    
    ax1 = plt.subplot(1,len(warmestyear),r+1)
    m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
    m.drawcoastlines(color='w',linewidth=0.3)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    x,y = m(lon2,lat2)
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    cs1.set_cmap(cmap) 
            
    ax1.annotate(r'\textbf{%s}' % (experimentnames[r]),xy=(0,0),xytext=(0.5,1.10),
                  textcoords='axes fraction',color='dimgrey',fontsize=14,
                  rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.35,0.085,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'SummerHeatExtremesTemperatureYear_OS-T2M_%s_%s_running-%syr_%s.png' % (variq,seasons[0],rolling_years,varcount),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot difference in warmest year
limit = np.arange(-50,51,5)
barlim = np.round(np.arange(-50,51,10),2)
cmap = cmocean.cm.balance
label = r'\textbf{Difference in Warmest Year [Ensemble Mean]}'

fig = plt.figure()

var = differenceWarmestYear

ax1 = plt.subplot(111)
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.5)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

x,y = m(lon2,lat2)
cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
cs1.set_cmap(cmap) 
        
ax1.annotate(r'\textbf{%s}' % (experimentnames[-1]),xy=(0,0),xytext=(0.5,1.10),
              textcoords='axes fraction',color='dimgrey',fontsize=14,
              rotation=0,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.08,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()

plt.savefig(directoryfigure + 'SummerHeatExtremesTemperatureYear-Difference_OS-T2M_%s_%s_running-%syr_%s.png' % (variq,seasons[0],rolling_years,varcount),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot difference in benefit
limit = np.arange(-40,41,1)
barlim = np.round(np.arange(-40,41,10),2)
cmap = cmocean.cm.balance
label = r'\textbf{Years}'

fig = plt.figure()

var = diffbenefit

ax1 = plt.subplot(111)
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
m.drawcoastlines(color='k',linewidth=0.5)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

x,y = m(lon2,lat2)
cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
cs1.set_cmap(cmap) 
        
ax1.annotate(r'\textbf{argmax(SSP5-34OS_10ye) - argmin(SSP5-34OS)}',xy=(0,0),xytext=(0.5,1.10),
              textcoords='axes fraction',color='dimgrey',fontsize=14,
              rotation=0,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.08,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()

plt.savefig(directoryfigure + 'SummerHeatExtremesTemperatureYear-Benefit_OS-T2M_%s_%s_running-%syr_%s.png' % (variq,seasons[0],rolling_years,varcount),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot difference in benefit
limit = np.arange(0,31,1)
barlim = np.round(np.arange(0,31,5),2)
cmap = cmr.fall_r
label = r'\textbf{Years [+10 years]}'

fig = plt.figure()

benefitdecade = diffbenefit.copy()
benefitdecade[np.where(benefitdecade > -10)] = np.nan
var = (benefitdecade*-1) - 10

ax1 = plt.subplot(111)
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
m.drawcoastlines(color='w',linewidth=0.5)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

x,y = m(lon2,lat2)
cs1 = m.contourf(lon2,lat2,var,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 
        
ax1.annotate(r'\textbf{ADDED BENEFIT BY MITIGATING 10 YEARS EARLIER}',xy=(0,0),xytext=(0.5,1.10),
              textcoords='axes fraction',color='dimgrey',fontsize=16,
              rotation=0,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.08,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()

plt.savefig(directoryfigure + 'SummerHeatExtremesTemperatureYear-BenefitMask_OS-%s_%s_running-%syr_%s.png' % (variq,seasons[0],rolling_years,varcount),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot difference in benefit for land only
limit = np.arange(0,31,1)
barlim = np.round(np.arange(0,31,5),2)
cmap = cmr.fall_r
label = r'\textbf{Years [+10 years]}'

fig = plt.figure()

benefitdecade = diffbenefit.copy()
benefitdecade[np.where(benefitdecade > -10)] = np.nan
var = (benefitdecade*-1) - 10

ax1 = plt.subplot(111)
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

x,y = m(lon2,lat2)
cs1 = m.contourf(lon2,lat2,var,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
m.drawcoastlines(color='darkgrey',linewidth=0.5,zorder=12)
        
ax1.annotate(r'\textbf{ADDED BENEFIT BY MITIGATING 10 YEARS EARLIER}',xy=(0,0),xytext=(0.5,1.10),
              textcoords='axes fraction',color='dimgrey',fontsize=16,
              rotation=0,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.08,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()

plt.savefig(directoryfigure + 'SummerHeatExtremesTemperatureYear-BenefitMask_OS-%s_%s_running-%syr_%s_land.png' % (variq,seasons[0],rolling_years,varcount),dpi=300)
