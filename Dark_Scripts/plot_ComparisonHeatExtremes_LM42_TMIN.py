"""
Calculate timeseries of US means of TMIN 

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

variablesall = ['TMIN']
variq = variablesall[0]
numOfEns = 30
numOfEns_LM42 = 3
years = np.arange(1921,2100+1)
years_LM42 = np.arange(1921,2070+1,1)

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
reg_name = 'Globe'

### Read in SPEAR_MED
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
count90 = data.variables['count90'][:,:len(years_LM42),:,:]
data.close()

### Read in SPEAR_MED_LM42p2_test
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_LM42 = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_LM42p2_test' + '.nc'
filename_LM42 = directorydatah + name_LM42
data_LM42 = Dataset(filename_LM42)
count90_LM42 = data_LM42.variables['count90'][:,:,:,:]
data_LM42.close()

### Read in SPEAR_MED
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
count99 = data.variables['count99'][:,:len(years_LM42),:,:]
data.close()

### Read in SPEAR_MED_LM42p2_test
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_LM42 = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_LM42p2_test' + '.nc'
filename_LM42 = directorydatah + name_LM42
data_LM42 = Dataset(filename_LM42)
count99_LM42 = data_LM42.variables['count99'][:,:,:,:]
data_LM42.close()

### Calculate ensemble means
count_90m = np.nanmean(count90,axis=0)
count_99m = np.nanmean(count99,axis=0)

count_LM42_90m = np.nanmean(count90_LM42,axis=0)
count_LM42_99m = np.nanmean(count99_LM42,axis=0)

### Calculate differences
diff_90 = count_LM42_90m - count_90m
diff_99 = count_LM42_99m - count_99m

### Calculate epochs
epoch_90early = np.nanmean(diff_90[:30,:,:],axis=0)
epoch_99early = np.nanmean(diff_99[:30,:,:],axis=0)

epoch_90late = np.nanmean(diff_90[-30:,:,:],axis=0)
epoch_99late = np.nanmean(diff_99[-30:,:,:],axis=0)

count_90m_early_s = np.nanmean(count_90m[:30,:,:],axis=0)
count_90m_early_l = np.nanmean(count_LM42_90m[:30,:,:],axis=0)
count_90m_late_s = np.nanmean(count_90m[-30:,:,:],axis=0)
count_90m_late_l = np.nanmean(count_LM42_90m[-30:,:,:],axis=0)

count_99m_early_s = np.nanmean(count_99m[:30,:,:],axis=0)
count_99m_early_l = np.nanmean(count_LM42_99m[:30,:,:],axis=0)
count_99m_late_s = np.nanmean(count_99m[-30:,:,:],axis=0)
count_99m_late_l = np.nanmean(count_LM42_99m[-30:,:,:],axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot figure
lon2,lat2 = np.meshgrid(lonus,latus)

fig = plt.figure(figsize=(9,6))
for Tnt in fig.texts:
    Tnt.set_visible(False)

### Colorbar limits
barlim = np.arange(-20,21,5)
limit = np.arange(-20,21,1)
label = r'\textbf{Difference (Days)}' 

ax = plt.subplot(221)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_90early,limit,
                  extend='both',latlon=True)
                
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) LM4.2 -- LM4; Count Tn90; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(222)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_99early,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) LM4.2 -- LM4; Count Tn99; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(223)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_90late,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(c) LM4.2 -- LM4; Count Tn90; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(224)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_99late,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(d) LM4.2 -- LM4; Count Tn99; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
cbar_ax1 = fig.add_axes([0.355,0.05,0.3,0.03])                
cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

plt.savefig(directoryfigure + 'ComparisonHeatExtremes_LM42_US_JJA_TMIN.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot figure
lon2,lat2 = np.meshgrid(lonus,latus)

fig = plt.figure(figsize=(9,6))
for Tnt in fig.texts:
    Tnt.set_visible(False)

### Colorbar limits
barlim = np.arange(0,51,10)
limit = np.arange(0,51,1)
label = r'\textbf{Difference (Days)}' 

ax = plt.subplot(221)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,count_90m_early_s,limit,
                  extend='max',latlon=True)
                
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
cmap = cmr.sunburst_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) SPEAR_MED; Count Tn90; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(222)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,count_90m_early_l,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.sunburst_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) SPEAR_MED_LM42; Count Tn90; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(223)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,count_90m_late_s,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.sunburst_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(c) SPEAR_MED; Count Tn90; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(224)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,count_90m_late_l,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.sunburst_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(d) SPEAR_MED_LM42; Count Tn90; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
cbar_ax1 = fig.add_axes([0.355,0.05,0.3,0.03])                
cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

plt.savefig(directoryfigure + 'ComparisonHeatExtremesCount90_LM42_US_JJA_TMIN.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot figure
lon2,lat2 = np.meshgrid(lonus,latus)

fig = plt.figure(figsize=(9,6))
for Tnt in fig.texts:
    Tnt.set_visible(False)

### Colorbar limits
barlim = np.arange(0,51,10)
limit = np.arange(0,51,1)
label = r'\textbf{Difference (Days)}' 

ax = plt.subplot(221)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,count_99m_early_s,limit,
                  extend='max',latlon=True)
                
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
cmap = cmr.sunburst_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) SPEAR_MED; Count Tn99; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(222)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,count_99m_early_l,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.sunburst_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) SPEAR_MED_LM42; Count Tn99; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(223)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,count_99m_late_s,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.sunburst_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(c) SPEAR_MED; Count Tn99; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(224)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
            area_thresh=5000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,count_99m_late_l,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.sunburst_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(d) SPEAR_MED_LM42; Count Tn99; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
cbar_ax1 = fig.add_axes([0.355,0.05,0.3,0.03])                
cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

plt.savefig(directoryfigure + 'ComparisonHeatExtremesCount99_LM42_US_JJA_TMIN.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Plot figure
lon2,lat2 = np.meshgrid(lonus,latus)

fig = plt.figure(figsize=(9,6))
for Tnt in fig.texts:
    Tnt.set_visible(False)

### Colorbar limits
barlim = np.arange(-20,21,5)
limit = np.arange(-20,21,1)
label = r'\textbf{Difference (Days)}' 

ax = plt.subplot(221)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='dimgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_90early,limit,
                  extend='both',latlon=True)
                
# m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) LM4.2 -- LM4; Count Tn90; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(222)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='dimgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_99early,limit,
                  extend='both',latlon=True)

# m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) LM4.2 -- LM4; Count Tn99; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(223)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='dimgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_90late,limit,
                  extend='both',latlon=True)

# m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(c) LM4.2 -- LM4; Count Tn90; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(224)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='dimgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_99late,limit,
                  extend='both',latlon=True)

# m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(d) LM4.2 -- LM4; Count Tn99; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
cbar_ax1 = fig.add_axes([0.355,0.05,0.3,0.03])                
cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

plt.savefig(directoryfigure + 'ComparisonHeatExtremes_LM42_Globe_JJA_TMIN.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot figure
lon2,lat2 = np.meshgrid(lonus,latus)

fig = plt.figure(figsize=(9,6))
for Tnt in fig.texts:
    Tnt.set_visible(False)

### Colorbar limits
barlim = np.arange(-20,21,5)
limit = np.arange(-20,21,1)
label = r'\textbf{Difference (Days)}' 

ax = plt.subplot(221)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_90early,limit,
                  extend='both',latlon=True)
                
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) LM4.2 -- LM4; Count Tn90; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(222)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_99early,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) LM4.2 -- LM4; Count Tn99; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(223)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_90late,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(c) LM4.2 -- LM4; Count Tn90; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(224)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_99late,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(d) LM4.2 -- LM4; Count Tn99; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
cbar_ax1 = fig.add_axes([0.355,0.05,0.3,0.03])                
cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

plt.savefig(directoryfigure + 'ComparisonHeatExtremes_LM42_Globe-Land_JJA_TMIN.png',dpi=300)
