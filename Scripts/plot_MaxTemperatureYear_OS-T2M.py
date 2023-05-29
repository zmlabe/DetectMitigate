"""
Calculate a map of the warmest year for the ensemble mean in the overshoot runs

Author    : Zachary M. Labe
Date      : 26 May 2023
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

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['T2M']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 9
years = np.arange(2015,2100+1)

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
seasons = ['annual']
slicemonthnamen = ['ANNUAL']
monthlychoice = seasons[0]
reg_name = 'Globe'

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
###############################################################################
###############################################################################
###############################################################################
### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_osm,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Calculate the ensemble mean
spear_osm_mean = np.nanmean(spear_osm,axis=0)
spear_osm_10ye_mean = np.nanmean(spear_osm_10ye,axis=0)

### Locate max of warmest year at each grid point
map_os = np.empty((lats.shape[0],lons.shape[0]))
map_os_10ye = np.empty((lats.shape[0],lons.shape[0]))
map_os_index = np.empty((lats.shape[0],lons.shape[0]))
map_os_10ye_index = np.empty((lats.shape[0],lons.shape[0]))
for i in range(lats.shape[0]):
    for j in range(lons.shape[0]):
        gridpoint_os = spear_osm_mean[:,i,j]
        gridpoint_os_10ye = spear_osm_10ye_mean[:,i,j]
        
        maxindex_os = np.argmax(gridpoint_os)
        maxindex_os_10ye = np.argmax(gridpoint_os_10ye)
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
        gridpoint_os = spear_osm_mean[:,i,j]
        gridpoint_os_10ye = spear_osm_10ye_mean[:,i,j]
        
        max_os = np.nanmax(gridpoint_os)
        max_os_10ye = np.nanmax(gridpoint_os_10ye)
        maxindex_os = np.argmax(gridpoint_os)
        maxindex_os_10ye = np.argmax(gridpoint_os_10ye)
        
        wherebegin_os = np.where((years == 2040))[0][0]
        wherebegin_os_10ye  = np.where((years == 2031))[0][0]
        if (maxindex_os_10ye >= wherebegin_os_10ye) and (maxindex_os >= wherebegin_os):
            
            maxAfterOS_10ye = np.nanmax(gridpoint_os_10ye[:])
            IndexmaxAfterOS_10ye = np.argmax(gridpoint_os_10ye[:])
            
            IndexmaxAfterOS = np.argmax(gridpoint_os[:])
            
            if len(np.where(gridpoint_os[IndexmaxAfterOS:] <= maxAfterOS_10ye)[0]) > 0:
                
                minwherebelowMax = np.min(np.where(gridpoint_os[IndexmaxAfterOS:] <= maxAfterOS_10ye)[0]) + IndexmaxAfterOS
                diffbenefit[i,j] = IndexmaxAfterOS_10ye - minwherebelowMax
          
# plt.figure()
# plt.plot(gridpoint_os)
# plt.plot(gridpoint_os_10ye)
# plt.axhline(np.max(gridpoint_os_10ye),color='k')
# sys.exit()
###############################################################################
###############################################################################
###############################################################################
### Plot subplot of warmst year
limit = np.arange(2015,2101,5)
barlim = np.round(np.arange(2015,2101,10),2)
# cmap = cm.classic_16.mpl_colormap
cmap = cmr.dusk
label = r'\textbf{Warmest Year [Ensemble Mean]}'
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

plt.savefig(directoryfigure + 'MaxTemperatureYear_OS-T2M.png',dpi=300)

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
ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
              textcoords='axes fraction',color='k',fontsize=6,
              rotation=330,ha='center',va='center')
    
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

plt.savefig(directoryfigure + 'MaxTemperatureYear-Difference_OS-T2M.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot difference in benefit
limit = np.arange(-50,51,5)
barlim = np.round(np.arange(-50,51,10),2)
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
ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
              textcoords='axes fraction',color='k',fontsize=6,
              rotation=330,ha='center',va='center')
    
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

plt.savefig(directoryfigure + 'MaxTemperatureYear-Benefit_OS-T2M.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot difference in benefit
limit = np.arange(-50,51,5)
barlim = np.round(np.arange(-50,51,10),2)
cmap = cmocean.cm.balance
label = r'\textbf{Years [Masked benefits worse than 1 decade]}'

fig = plt.figure()

benefitdecade = diffbenefit.copy()
benefitdecade[np.where(benefitdecade > -10)] = np.nan
var = benefitdecade

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
ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
              textcoords='axes fraction',color='k',fontsize=6,
              rotation=330,ha='center',va='center')
    
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

plt.savefig(directoryfigure + 'MaxTemperatureYear-BenefitMask_OS-T2M.png',dpi=300)
