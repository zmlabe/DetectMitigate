"""
Plot changes in variance for LM4 and LM42
 
Author    : Zachary M. Labe
Date      : 25 September 2023
"""

import matplotlib.pyplot as plt
import calc_Utilities as UT
import xarray as xr
import numpy as np
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean
import os, glob
import calc_Stats as dSS
import sys
from netCDF4 import Dataset
import calc_DetrendData as DT
import cmasher as cmr

import time
startTime = time.time()

###############################################################################
###############################################################################
###############################################################################
### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

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

letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
directorydata = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
modelnames = ['SPEAR_MED','SPEAR_MED_LM42']
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

years = np.arange(1921,2070+1,1)
   
### Read in the daily data
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
        ENS = 9
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
        if any([vari == 'T2M',vari == 'TMAX']):
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

summer_LM42p2_test,lat,lon,years_LM42p2_test = readData('SPEAR_MED_LM42p2_test',reg_name)
summer_all,lat,lon,years = readData('SPEAR_MED',reg_name)
lon2,lat2 = np.meshgrid(lon,lat)

### Pick years for 1981-2010
summer = summer_all[:,:summer_LM42p2_test.shape[1],:,:,:] # only do through 2070

yearq = np.where((years >= 1921) & (years <= 1950))[0] # 30 years

s_lm42 = summer_LM42p2_test[:,yearq,:,:,:]
s_all = summer[:,yearq,:,:,:]

### Calculate ensemble means
lm42_mean = np.nanmean(s_lm42,axis=0)
all_mean = np.nanmean(s_all,axis=0)

### Detrend data for SPEAR_MED
daytrend_spear = np.full(s_all.shape,np.nan)
for ens in range(len(s_all)):
    for days in range(s_all.shape[2]):
        daytrend_spear[ens,:,days,:,:] = DT.detrendDataR(s_all[ens,:,days,:,:],'surface','daily')
        
### Detrend data for SPEAR_MED_LM42
daytrend_spear_lm42 = np.full(s_lm42.shape,np.nan)
for ens in range(len(s_lm42)):
    for days in range(s_lm42.shape[2]):
        daytrend_spear_lm42[ens,:,days,:,:] = DT.detrendDataR(s_lm42[ens,:,days,:,:],'surface','daily')

### Reshape so years*days
spear_reshape = s_all.reshape(s_all.shape[0],s_all.shape[1]*s_all.shape[2],s_all.shape[3],s_all.shape[4])
spear_lm42_reshape = s_lm42.reshape(s_lm42.shape[0],s_lm42.shape[1]*s_lm42.shape[2],s_lm42.shape[3],s_lm42.shape[4])

spear_dt_reshape = daytrend_spear.reshape(daytrend_spear.shape[0],daytrend_spear.shape[1]*daytrend_spear.shape[2],daytrend_spear.shape[3],daytrend_spear.shape[4])
spear_lm42_dt_reshape = daytrend_spear_lm42.reshape(daytrend_spear_lm42.shape[0],daytrend_spear_lm42.shape[1]*daytrend_spear_lm42.shape[2],daytrend_spear_lm42.shape[3],daytrend_spear_lm42.shape[4])

### Calculate variance at each grid cell
var_spear = np.var(spear_dt_reshape,axis=1)
var_lm42 = np.var(spear_lm42_dt_reshape,axis=1)

### Calculate ensemble mean of the variance
var_spearm = np.nanmean(var_spear,axis=0)
var_lm42m = np.nanmean(var_lm42,axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot climo figures
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
def setcolor(x, color):
      for m in x:
          for t in x[m][1]:
              t.set_color(color)

fig = plt.figure(figsize=(10,3.5))

label = r'\textbf{TMAX Var. [$^{\circ}$C$^{2}$]}'
limit = np.arange(0,20.01,0.2)
barlim = np.round(np.arange(0,21,5),2)

ax = plt.subplot(1,3,1)
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',area_thresh=1000)
m.drawcoastlines(color='k',linewidth=1)
m.drawstates(color='k',linewidth=0.5)
m.drawcountries(color='k',linewidth=1,zorder=12)

circle = m.drawmapboundary(fill_color='darkgrey',color='darkgrey',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.pcolormesh(lon2,lat2,var_spearm,vmin=0,vmax=20,latlon=True)
cs1.set_cmap(cmr.ocean_r)

ax.annotate(r'\textbf{[%s]}' % (letters[0]),xy=(0,0),xytext=(0.0,1.05),
          textcoords='axes fraction',color='k',fontsize=9,
          rotation=0,ha='center',va='center')

parallels = np.arange(-90,91,5)
meridians = np.arange(-180,180,15)
par=m.drawparallels(parallels,labels=[True,False,False,True],linewidth=0.5,
                color='w',fontsize=6,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,True,False],linewidth=0.5,
                    fontsize=6,color='w',zorder=40)
setcolor(mer,'dimgrey')
setcolor(par,'dimgrey')

plt.title(r'\textbf{%s}' % modelnames[0],color='dimgrey',fontsize=20,y=1.1)

cbar_ax1 = fig.add_axes([0.09,0.12,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='k',labelpad=3)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

###############################################################################
label = r'\textbf{TMAX Var. [$^{\circ}$C$^{2}$]}'
limit = np.arange(0,20.01,0.2)
barlim = np.round(np.arange(0,21,5),2)

ax = plt.subplot(1,3,2)
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',area_thresh=1000)
m.drawcoastlines(color='k',linewidth=1)
m.drawstates(color='k',linewidth=0.5)
m.drawcountries(color='k',linewidth=1,zorder=12)

circle = m.drawmapboundary(fill_color='darkgrey',color='darkgrey',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.pcolormesh(lon2,lat2,var_lm42m,vmin=0,vmax=20,latlon=True)
cs1.set_cmap(cmr.ocean_r)

ax.annotate(r'\textbf{[%s]}' % (letters[1]),xy=(0,0),xytext=(0.0,1.05),
          textcoords='axes fraction',color='k',fontsize=9,
          rotation=0,ha='center',va='center')

parallels = np.arange(-90,91,5)
meridians = np.arange(-180,180,15)
par=m.drawparallels(parallels,labels=[False,False,False,False],linewidth=0.5,
                color='w',fontsize=6,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,True,False],linewidth=0.5,
                    fontsize=6,color='w',zorder=40)
setcolor(mer,'dimgrey')
setcolor(par,'dimgrey')

plt.title(r'\textbf{%s}' % modelnames[1],color='dimgrey',fontsize=20,y=1.1)
    
cbar_ax1 = fig.add_axes([0.4,0.12,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='k',labelpad=3)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

###############################################################################
label = r'\textbf{TMAX Var. Difference [$^{\circ}$C$^{2}$]}'
limit = np.arange(-6,6.01,0.1)
barlim = np.round(np.arange(-6,7,2),2)

ax = plt.subplot(1,3,3)
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',area_thresh=1000)
m.drawcoastlines(color='k',linewidth=1)
m.drawstates(color='k',linewidth=0.5)
m.drawcountries(color='k',linewidth=1,zorder=12)

circle = m.drawmapboundary(fill_color='darkgrey',color='darkgrey',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.pcolormesh(lon2,lat2,var_spearm-var_lm42m,vmin=-6,vmax=6,latlon=True)
cs1.set_cmap(cmocean.cm.balance)

ax.annotate(r'\textbf{[%s]}' % (letters[2]),xy=(0,0),xytext=(0.0,1.05),
          textcoords='axes fraction',color='k',fontsize=9,
          rotation=0,ha='center',va='center')

parallels = np.arange(-90,91,5)
meridians = np.arange(-180,180,15)
par=m.drawparallels(parallels,labels=[False,True,False,False],linewidth=0.5,
                color='w',fontsize=6,zorder=40)
mer=m.drawmeridians(meridians,labels=[False,False,True,False],linewidth=0.5,
                    fontsize=6,color='w',zorder=40)
setcolor(mer,'dimgrey')
setcolor(par,'dimgrey')

plt.title(r'\textbf{a minus b}',color='dimgrey',fontsize=20,y=1.1)
    
cbar_ax1 = fig.add_axes([0.71,0.12,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='k',labelpad=3)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
        
plt.savefig(directoryfigure + 'varianceChange_LM42_1921-1950.png',dpi=300)
