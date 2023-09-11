"""
Calculate timeseries of US means of q (dry)

Author    : Zachary M. Labe
Date      : 8 September 2023
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
name = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
count10 = data.variables['count10'][:,:len(years_LM42),:,:]
data.close()

### Read in SPEAR_MED_LM42p2_test
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_LM42 = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_LM42p2_test' + '.nc'
filename_LM42 = directorydatah + name_LM42
data_LM42 = Dataset(filename_LM42)
count10_LM42 = data_LM42.variables['count10'][:,:,:,:]
data_LM42.close()

### Read in SPEAR_MED
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
count01 = data.variables['count01'][:,:len(years_LM42),:,:]
data.close()

### Read in SPEAR_MED_LM42p2_test
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_LM42 = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_LM42p2_test' + '.nc'
filename_LM42 = directorydatah + name_LM42
data_LM42 = Dataset(filename_LM42)
count01_LM42 = data_LM42.variables['count01'][:,:,:,:]
data_LM42.close()

### Calculate ensemble means
count_10m = np.nanmean(count10,axis=0)
count_01m = np.nanmean(count01,axis=0)

count_LM42_10m = np.nanmean(count10_LM42,axis=0)
count_LM42_01m = np.nanmean(count01_LM42,axis=0)

### Calculate differences
diff_10 = count_LM42_10m - count_10m
diff_01 = count_LM42_01m - count_01m

### Calculate epochs
epoch_10early = np.nanmean(diff_10[:30,:,:],axis=0)
epoch_01early = np.nanmean(diff_01[:30,:,:],axis=0)

epoch_10late = np.nanmean(diff_10[-30:,:,:],axis=0)
epoch_01late = np.nanmean(diff_01[-30:,:,:],axis=0)

count_10m_early_s = np.nanmean(count_10m[:30,:,:],axis=0)
count_10m_early_l = np.nanmean(count_LM42_10m[:30,:,:],axis=0)
count_10m_late_s = np.nanmean(count_10m[-30:,:,:],axis=0)
count_10m_late_l = np.nanmean(count_LM42_10m[-30:,:,:],axis=0)

count_01m_early_s = np.nanmean(count_01m[:30,:,:],axis=0)
count_01m_early_l = np.nanmean(count_LM42_01m[:30,:,:],axis=0)
count_01m_late_s = np.nanmean(count_01m[-30:,:,:],axis=0)
count_01m_late_l = np.nanmean(count_LM42_01m[-30:,:,:],axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot figure
lon2,lat2 = np.meshgrid(lonus,latus)

fig = plt.figure(figsize=(9,6))
for txt in fig.texts:
    txt.set_visible(False)

### Colorbar limits
barlim = np.arange(-10,11,5)
limit = np.arange(-10,11,1)
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
cs = m.contourf(lon2,lat2,epoch_10early,limit,
                  extend='both',latlon=True)
                
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) LM4.2 -- LM4; Count qx90; 1921-1950}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,epoch_01early,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) LM4.2 -- LM4; Count qx99; 1921-1950}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,epoch_10late,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(c) LM4.2 -- LM4; Count qx90; 2041-2070}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,epoch_01late,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(d) LM4.2 -- LM4; Count qx99; 2041-2070}',fontsize=11,color='dimgrey')

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

plt.savefig(directoryfigure + 'ComparisonMoistExtremes_LM42_US_JJA.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot figure
lon2,lat2 = np.meshgrid(lonus,latus)

fig = plt.figure(figsize=(9,6))
for txt in fig.texts:
    txt.set_visible(False)

### Colorbar limits
barlim = np.arange(0,31,10)
limit = np.arange(0,31,1)
label = r'\textbf{Days}' 

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
cs = m.contourf(lon2,lat2,count_10m_early_s,limit,
                  extend='max',latlon=True)
                
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
cmap = cmr.fall_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) SPEAR_MED; Count qx90; 1921-1950}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,count_10m_early_l,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.fall_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) SPEAR_MED_LM42; Count qx90; 1921-1950}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,count_10m_late_s,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.fall_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(c) SPEAR_MED; Count qx90; 2041-2070}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,count_10m_late_l,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.fall_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(d) SPEAR_MED_LM42; Count qx90; 2041-2070}',fontsize=11,color='dimgrey')

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

plt.savefig(directoryfigure + 'ComparisonMoistExtremescount10_LM42_US_JJA.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot figure
lon2,lat2 = np.meshgrid(lonus,latus)

fig = plt.figure(figsize=(9,6))
for txt in fig.texts:
    txt.set_visible(False)

### Colorbar limits
barlim = np.arange(0,31,10)
limit = np.arange(0,31,1)
label = r'\textbf{Days}' 

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
cs = m.contourf(lon2,lat2,count_01m_early_s,limit,
                  extend='max',latlon=True)
                
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
cmap = cmr.fall_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) SPEAR_MED; Count qx99; 1921-1950}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,count_01m_early_l,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.fall_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) SPEAR_MED_LM42; Count qx99; 1921-1950}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,count_01m_late_s,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.fall_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(c) SPEAR_MED; Count qx99; 2041-2070}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,count_01m_late_l,limit,
                  extend='max',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmr.fall_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(d) SPEAR_MED_LM42; Count qx99; 2041-2070}',fontsize=11,color='dimgrey')

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

plt.savefig(directoryfigure + 'ComparisonMoistExtremescount01_LM42_US_JJA.png',dpi=300)

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
for txt in fig.texts:
    txt.set_visible(False)

### Colorbar limits
barlim = np.arange(-10,11,5)
limit = np.arange(-10,11,1)
label = r'\textbf{Difference (Days)}' 

ax = plt.subplot(221)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='dimgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_10early,limit,
                  extend='both',latlon=True)
                
# m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) LM4.2 -- LM4; Count qx90; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(222)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='dimgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_01early,limit,
                  extend='both',latlon=True)

# m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) LM4.2 -- LM4; Count qx99; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(223)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='dimgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_10late,limit,
                  extend='both',latlon=True)

# m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(c) LM4.2 -- LM4; Count qx90; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(224)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='dimgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_01late,limit,
                  extend='both',latlon=True)

# m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(d) LM4.2 -- LM4; Count qx99; 2041-2070}',fontsize=11,color='dimgrey')

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

plt.savefig(directoryfigure + 'ComparisonMoistExtremes_LM42_Globe_JJA.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot figure
lon2,lat2 = np.meshgrid(lonus,latus)

fig = plt.figure(figsize=(9,6))
for txt in fig.texts:
    txt.set_visible(False)

### Colorbar limits
barlim = np.arange(-10,11,5)
limit = np.arange(-10,11,1)
label = r'\textbf{Difference (Days)}' 

ax = plt.subplot(221)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_10early,limit,
                  extend='both',latlon=True)
                
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) LM4.2 -- LM4; Count qx90; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(222)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_01early,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) LM4.2 -- LM4; Count qx99; 1921-1950}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(223)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_10late,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(c) LM4.2 -- LM4; Count qx90; 2041-2070}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(224)
    
### Select map type
m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,epoch_01late,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(d) LM4.2 -- LM4; Count qx99; 2041-2070}',fontsize=11,color='dimgrey')

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

plt.savefig(directoryfigure + 'ComparisonMoistExtremes_LM42_Globe-Land_JJA.png',dpi=300)
