"""
Explore heat extremes in the OS runs
 
Author    : Zachary M. Labe
Date      : 21 July 2023
"""

import matplotlib.pyplot as plt
import calc_Utilities as UT
import xarray as xr
import numpy as np
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean
import cmasher as cmr
import os, glob
import calc_Stats as dSS
import sys
from netCDF4 import Dataset

import time
startTime = time.time()

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Directories
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
directorydata = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'

### Parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
vari = 'TMAX'
variq = 't_ref_max'
resolution = 'MEDS' 
minb = 1981
maxb = 2010
junedays = np.arange(0,30,1)
julydays = np.arange(0,31,1)
augustdays = np.arange(0,31,1)
dayslength = len(junedays) + len(julydays) + len(augustdays)
reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)
def setcolor(x, color):
      for m in x:
          for t in x[m][1]:
              t.set_color(color)
   
def readData(model,reg_name):
    directorydata = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
    
    ### Select model
    if model == 'SPEAR_MED':
        years = np.arange(1921,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS':
        years = np.arange(2011,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP245':
        years = np.arange(2011,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS_10ye':
        years = np.arange(2031,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv':
        years = np.arange(2041,2100+1)
        ENS = 9
    elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv':
        years = np.arange(2041,2100+1)
        ENS = 9
    elif model == 'SPEAR_MED_LM42p2_test':
        years = np.arange(1921,2070+1)
        ENS = 3
    
    if reg_name == 'US':
        daysall = np.empty((ENS,years.shape[0],dayslength,70,144))
    elif reg_name == 'Globe':
        daysall = np.empty((ENS,years.shape[0],dayslength,360,576))
    for e in range(ENS):
        data = Dataset(directorydata + 'June/June_%s_%s_%s_ens%s.nc' % (vari,reg_name,model,str(e+1).zfill(2)))
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
        june = data.variables['%s' % variq][:].reshape(len(years),len(junedays),lat.shape[0],lon.shape[0])
        data.close()
        
        data2 = Dataset(directorydata + 'July/July_%s_%s_%s_ens%s.nc' % (vari,reg_name,model,str(e+1).zfill(2)))
        july = data2.variables['%s' % variq][:].reshape(len(years),len(julydays),lat.shape[0],lon.shape[0])
        data2.close()

        data3 = Dataset(directorydata + 'August/August_%s_%s_%s_ens%s.nc' % (vari,reg_name,model,str(e+1).zfill(2)))
        august = data3.variables['%s' % variq][:].reshape(len(years),len(augustdays),lat.shape[0],lon.shape[0])
        data3.close()        

        ### Convert units if needed
        if any([vari == 'T2M',vari == 'TMAX',vari == 'TMIN']):
            daysall[e,:,:,:,:] = np.concatenate([june,july,august],axis=1) - 273.15 # K to C
            print('Completed: Changed units (K to C)!')
        else:
            daysall[e,:,:,:,:] = np.concatenate([june,july,august],axis=1)
            
    ### Meshgrid and mask by CONUS
    lon2,lat2 = np.meshgrid(lon,lat)
    
    if reg_name == 'US':
        data_obsnan = np.full([1,lat.shape[0],lon.shape[0]],np.nan)
        datamask,data_obsnan = dSS.mask_CONUS(daysall,data_obsnan,resolution,lat_bounds,lon_bounds)
    else:
        datamask = daysall
        
    return datamask,lat,lon,years

def calc_heatExtremes(datamask,model,lat,lon):
    
    ### Select model
    if model == 'SPEAR_MED':
        years = np.arange(1921,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS':
        years = np.arange(2011,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP245':
        years = np.arange(2011,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS_10ye':
        years = np.arange(2031,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv':
        years = np.arange(2041,2100+1)
        ENS = 9
    elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv':
        years = np.arange(2041,2100+1)
        ENS = 9
    elif model == 'SPEAR_MED_LM42p2_test':
        years = np.arange(1921,2070+1)
        ENS = 3
    
    if any([model == 'SPEAR_MED',model == 'SPEAR_MED_LM42p2_test']):
        ### Calculate baseline
        baseline = np.arange(minb,maxb+1,1)
        baseq = np.where((years >= minb) & (years <= maxb))[0]
        
        clim = datamask[:,baseq,:,:,:]
        climdist = clim.reshape(ENS,len(baseline)*dayslength,lat.shape[0],lon.shape[0])
    
    ### Calculate heat extremes
    tx90 = np.nanpercentile(climdist,90,axis=1)
    tx95 = np.nanpercentile(climdist,95,axis=1)
    tx99 = np.nanpercentile(climdist,99,axis=1)
    
    return tx90,tx95,tx99

### Read in data
summer,lat,lon,years = readData('SPEAR_MED',reg_name)
tx90,tx95,tx99 = calc_heatExtremes(summer,'SPEAR_MED',lat,lon)

### Calculate ensemble means
tx90ens = np.nanmean(tx90[:,:,:],axis=0)
tx95ens = np.nanmean(tx95[:,:,:],axis=0)
tx99ens = np.nanmean(tx99[:,:,:],axis=0)

###############################################################################
###############################################################################
###############################################################################
limit = np.arange(20,41,1)
barlim = np.round(np.arange(20,41,5),2)
label = r'\textbf{TxX Threshold -- 1981-2010 [JJA; $^{\circ}$C]}'

fig = plt.figure(figsize=(5,10))                                                                                                                        
ax = plt.subplot(311)     

var = tx90ens

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',area_thresh=1000)
m.drawcoastlines(color='w',linewidth=1)
m.drawstates(color='w',linewidth=0.5)
m.drawcountries(color='w',linewidth=1,zorder=12)

circle = m.drawmapboundary(fill_color='dimgrey',color='k',
                  linewidth=2)
circle.set_clip_on(False)
    
lon2,lat2 = np.meshgrid(lon,lat)

cs1 = m.pcolormesh(lon2,lat2,var,vmin=20,vmax=40,latlon=True)

cs1.set_cmap(cmr.ember)

parallels = np.arange(-90,91,5)
meridians = np.arange(-180,180,15)
par=m.drawparallels(parallels,labels=[True,False,False,True],linewidth=0.5,
                color='darkgrey',fontsize=6,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,True,False],linewidth=0.5,
                    fontsize=6,color='darkgrey',zorder=40)
setcolor(mer,'dimgrey')
setcolor(par,'dimgrey')

ax.annotate(r'\textbf{[%s]}' % (letters[0]),xy=(0,0),xytext=(0.0,1.02),
          textcoords='axes fraction',color='k',fontsize=9,
          rotation=0,ha='center',va='center')
ax.annotate(r'\textbf{90th}',xy=(0,0),xytext=(1.03,0.5),
          textcoords='axes fraction',color='k',fontsize=17,
          rotation=270,ha='center',va='center')

###############################################################################
var = tx95ens

ax = plt.subplot(312) 
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',area_thresh=1000)
m.drawcoastlines(color='w',linewidth=1)
m.drawstates(color='w',linewidth=0.5)
m.drawcountries(color='w',linewidth=1,zorder=12)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgrey',
                  linewidth=0.7)
circle.set_clip_on(False)
    
lon2,lat2 = np.meshgrid(lon,lat)

cs1 = m.pcolormesh(lon2,lat2,var,vmin=20,vmax=40,latlon=True)

cs1.set_cmap(cmr.ember)

parallels = np.arange(-90,91,5)
meridians = np.arange(-180,180,15)
par=m.drawparallels(parallels,labels=[True,False,False,True],linewidth=0.5,
                color='darkgrey',fontsize=6,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,True,False],linewidth=0.5,
                    fontsize=6,color='darkgrey',zorder=40)
setcolor(mer,'dimgrey')
setcolor(par,'dimgrey')

ax.annotate(r'\textbf{[%s]}' % (letters[1]),xy=(0,0),xytext=(0.0,1.02),
          textcoords='axes fraction',color='k',fontsize=9,
          rotation=0,ha='center',va='center')
ax.annotate(r'\textbf{95th}',xy=(0,0),xytext=(1.03,0.5),
          textcoords='axes fraction',color='k',fontsize=17,
          rotation=270,ha='center',va='center')

###############################################################################
var = tx99ens

ax = plt.subplot(313) 
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',area_thresh=1000)
m.drawcoastlines(color='w',linewidth=1)
m.drawstates(color='w',linewidth=0.5)
m.drawcountries(color='w',linewidth=1,zorder=12)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgrey',
                  linewidth=0.7)
circle.set_clip_on(False)
    
lon2,lat2 = np.meshgrid(lon,lat)

cs1 = m.pcolormesh(lon2,lat2,var,vmin=20,vmax=40,latlon=True)

cs1.set_cmap(cmr.ember)

parallels = np.arange(-90,91,5)
meridians = np.arange(-180,180,15)
par=m.drawparallels(parallels,labels=[True,False,False,True],linewidth=0.5,
                color='darkgrey',fontsize=6,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,True,False],linewidth=0.5,
                    fontsize=6,color='darkgrey',zorder=40)
setcolor(mer,'dimgrey')
setcolor(par,'dimgrey')

ax.annotate(r'\textbf{[%s]}' % (letters[2]),xy=(0,0),xytext=(0.0,1.02),
          textcoords='axes fraction',color='k',fontsize=9,
          rotation=0,ha='center',va='center')
ax.annotate(r'\textbf{99th}',xy=(0,0),xytext=(1.03,0.5),
          textcoords='axes fraction',color='k',fontsize=17,
          rotation=270,ha='center',va='center')

cbar_ax1 = fig.add_axes([0.35,0.04,0.3,0.015])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=11,color='k',labelpad=3)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=9)
cbar1.outline.set_edgecolor('dimgrey')
        
plt.tight_layout()
plt.subplots_adjust(bottom=0.07)    

plt.savefig(directoryfigure + 'Climo_1981-2010_TxX.png',dpi=600)
