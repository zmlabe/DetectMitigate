"""
Plot trends in JJA daily heat extremes and q

Author    : Zachary M. Labe
Date      : 18 September 2023
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
directorydata = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/MeanJJA/'
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'

months = 'JJA'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
experi1 = ['SPEAR_MED','SPEAR_MED_LM42','b minus a']
lenmon = 3
numOfEns = 30
numOfEns_LM42 = 3
years = np.arange(1921,2100+1)
years_LM42 = np.arange(1921,2070+1)

### Select min and max years for trend
yearmin = 1990
yearmax = 2023

junedays = np.arange(0,30,1)
julydays = np.arange(0,31,1)
augustdays = np.arange(0,31,1)
dayslength = len(junedays) + len(julydays) + len(augustdays)

reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)

### Read in Q
data = Dataset(directorydata + 'JJAmean_US_q_SPEAR_MED_LM42p2_test.nc')
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
q_lm42 = data.variables['meanJJA'][:,:,:,:]
data.close()

data = Dataset(directorydata + 'JJAmean_US_q_SPEAR_MED.nc')
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
q = data.variables['meanJJA'][:,:len(years_LM42),:,:]
data.close()

### Read in TMAX
data = Dataset(directorydata + 'JJAmean_US_TMAX_SPEAR_MED_LM42p2_test.nc')
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
tmax_lm42 = data.variables['meanJJA'][:,:,:,:]
data.close()

data = Dataset(directorydata + 'JJAmean_US_TMAX_SPEAR_MED.nc')
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
tmax = data.variables['meanJJA'][:,:len(years_LM42),:,:]
data.close()

### Calculate ensemble means
q_lm42_ens = np.nanmean(q_lm42,axis=0)
q_ens = np.nanmean(q,axis=0)

tmax_lm42_ens = np.nanmean(tmax_lm42,axis=0)
tmax_ens = np.nanmean(tmax,axis=0)

### Mask nans briefly
q_lm42_ens[np.where(np.isnan(q_lm42_ens))] = 0.
q_ens[np.where(np.isnan(q_ens))] = 0.
tmax_lm42_ens[np.where(np.isnan(tmax_lm42_ens))] = 0.
tmax_ens[np.where(np.isnan(tmax_ens))] = 0.

### Select only the years for yearmin-2023
yearq = np.where((years_LM42 >= yearmin) & (years_LM42 <= 2023))[0]

### Calculate trends
q_trend_lm42 = UT.linearTrendR(q_lm42_ens,years_LM42,'surface',yearmin,yearmax)*10.
q_trend = UT.linearTrendR(q_ens,years_LM42,'surface',yearmin,yearmax)*10.

tmax_trend_lm42 = UT.linearTrendR(tmax_lm42_ens,years_LM42,'surface',yearmin,yearmax)*10.
tmax_trend = UT.linearTrendR(tmax_ens,years_LM42,'surface',yearmin,yearmax)*10.

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of different SPEAR for TMAX
fig = plt.figure(figsize=(10,3))

label = r'\textbf{Trend [$^{\circ}$C/decade] for %s-%s}' % (yearmin,yearmax)
limit = np.arange(-0.9,0.91,0.02)
barlim = np.round(np.arange(-0.9,0.91,0.3),2)

plotdata = [tmax_trend,tmax_trend_lm42,tmax_trend_lm42-tmax_trend]
plotlat = [lat,lat,lat]
plotlon = [lon,lon,lon]
variq = ['TMAX','TMAX','TMAX']

for i in range(len(plotdata)):
    ax = plt.subplot(1,3,i+1)
    
    var = plotdata[i]
    lat1 = plotlat[i]
    lon1 = plotlon[i]
    
    ### Mask zeros
    var[np.where(var == 0.)] = np.nan
    
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='dimgrey',linewidth=1)
    m.drawstates(color='dimgrey',linewidth=0.5)
    m.drawcountries(color='dimgrey',linewidth=1)
    
    circle = m.drawmapboundary(fill_color='darkgrey',color='darkgrey',
                      linewidth=0.7)
    circle.set_clip_on(False)
        
    lon2,lat2 = np.meshgrid(lon1,lat1)
    
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    cs1.set_cmap(cmocean.cm.balance)
    
    if i < 3:
        plt.title(r'\textbf{%s}' % experi1[i],fontsize=15,color='dimgrey')
    ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(0.0,1.07),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')
    
    if any([i==0]):
        ax.annotate(r'\textbf{%s}' % variq[i],xy=(0,0),xytext=(-0.08,0.5),
                      textcoords='axes fraction',color='k',fontsize=25,
                      rotation=90,ha='center',va='center')
    
cbar_ax1 = fig.add_axes([0.31,0.15,0.4,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='k',labelpad=3)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

plt.subplots_adjust(hspace=0)
# plt.tight_layout()
        
plt.savefig(directoryfigure + 'Trend_TMAX_ComparisonLM42_%s_%s-%s.png' % (months,yearmin,yearmax),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of different SPEAR for q
fig = plt.figure(figsize=(10,3))

label = r'\textbf{Trend [g/kg/decade] for %s-%s}' % (yearmin,yearmax)
limit = np.arange(-0.5,0.51,0.01)
barlim = np.round(np.arange(-0.5,0.51,0.1),2)

plotdata = [q_trend,q_trend_lm42,q_trend_lm42-q_trend]
plotlat = [lat,lat,lat]
plotlon = [lon,lon,lon]
variq = ['2-m q','2-m q','2-m q']

for i in range(len(plotdata)):
    ax = plt.subplot(1,3,i+1)
    
    var = plotdata[i] * 1000.
    lat1 = plotlat[i]
    lon1 = plotlon[i]
    
    ### Mask zeros
    var[np.where(var == 0.)] = np.nan
    
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l')
    m.drawcoastlines(color='dimgrey',linewidth=1)
    m.drawstates(color='dimgrey',linewidth=0.5)
    m.drawcountries(color='dimgrey',linewidth=1)
    
    circle = m.drawmapboundary(fill_color='darkgrey',color='darkgrey',
                      linewidth=0.7)
    circle.set_clip_on(False)
        
    lon2,lat2 = np.meshgrid(lon1,lat1)
    
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    cs1.set_cmap(cmr.seasons_r)
    
    if i < 3:
        plt.title(r'\textbf{%s}' % experi1[i],fontsize=15,color='dimgrey')
    ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(0.0,1.07),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')
    
    if any([i==0]):
        ax.annotate(r'\textbf{%s}' % variq[i],xy=(0,0),xytext=(-0.08,0.5),
                      textcoords='axes fraction',color='k',fontsize=25,
                      rotation=90,ha='center',va='center')
    
cbar_ax1 = fig.add_axes([0.31,0.15,0.4,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=10,color='k',labelpad=3)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

plt.subplots_adjust(hspace=0)
# plt.tight_layout()
        
plt.savefig(directoryfigure + 'Trend_q_ComparisonLM42_%s_%s-%s.png' % (months,yearmin,yearmax),dpi=300)
