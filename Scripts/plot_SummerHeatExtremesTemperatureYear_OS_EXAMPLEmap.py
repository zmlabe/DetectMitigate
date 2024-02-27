"""
Plot timeseries of a location to show how the net benefit is calculated

Author    : Zachary M. Labe
Date      : 26 February 2024
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

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
directorydata = '/home/Zachary.Labe/Research/DetectMitigate/Data/GridLocationExample/'
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
smoothingFactor = 10
YrThreshN = 10
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
raw = np.load(directorydata + 'gridpointExample_%s.npz' % varcount)
os_raw = raw['gridpoint_osX'][:]
os10ye_raw = raw['gridpoint_os_10yeX'][:]
whereBenefit_raw = raw['minwherebelowMax_X']
lats = raw['lats'][:]
lons = raw['lons'][:]
latqPoint_raw = raw['latqPoint']
lonqPoint_raw = raw['lonqPoint']
latloc = lats[latqPoint_raw]
lonloc = lons[lonqPoint_raw]
benefit_raw = raw['diffbenefit'][:]

threshold = np.load(directorydata + 'gridpointExample_YrThreshold-%s_%s.npz' % (YrThreshN,varcount))
whereBenefit_threshold = threshold['minwherebelowMax_X']
lats = threshold['lats'][:]
lons = threshold['lons'][:]
latqPoint_threshold = threshold['latqPoint']
lonqPoint_threshold = threshold['lonqPoint']
benefit_threshold = threshold['diffbenefit'][:]

filtering = np.load(directorydata + 'gridpointExample_savgolfilter_%s_%s.npz' % (smoothingFactor,varcount))
os_filtering = filtering['gridpoint_osX'][:]
os10ye_filtering = filtering['gridpoint_os_10yeX'][:]
whereBenefit_filtering = filtering['minwherebelowMax_X']
latqPoint_filtering = filtering['latqPoint']
lonqPoint_filtering = filtering['lonqPoint']
benefit_filtering = filtering['diffbenefit'][:]

###############################################################################
###############################################################################
###############################################################################
### Plot figure
lon2,lat2 = np.meshgrid(lons,lats)

### Select map type
style = 'US'

if style == 'ortho':
    m = Basemap(projection='ortho',lon_0=270,
                lat_0=50,resolution='h',round=True,area_thresh=10000)
elif style == 'polar':
    m = Basemap(projection='npstere',boundinglat=67,lon_0=270,resolution='h',round=True,area_thresh=10000)
elif style == 'global':
    m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
elif style == 'US':
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
                area_thresh=10000)
    
### Colorbar limits
if any([variq == 'TMAX',variq == 'T2M']):
    barlim = np.arange(0,51,10)
    limit = np.arange(0,51,1)
    barlim2 = np.arange(0,31,5)
    limit2 = np.arange(0,31,1)
    cmap = cmr.fall_r 
if variq == 'TMAX':
    if varcount == 'count90':
        label = r'\textbf{Years [+10 years]}' 
    elif varcount == 'count95':
        label = r'\textbf{Years [+10 years]}' 
elif variq == 'T2M':
    if varcount == 'count90':
        label = r'\textbf{Years [+10 years]}' 
    elif varcount == 'count95':
        label = r'\textbf{Years [+10 years]}'

### Map world map
fig = plt.figure(figsize=(10,3.5))
ax = plt.subplot(131)
for txt in fig.texts:
    txt.set_visible(False)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

### Make the plot continuous
benefit_rawm = benefit_raw.copy()
benefit_rawm[np.where(benefit_rawm > -10)] = np.nan
benefit_rawmask = (benefit_rawm*-1) - 10

# cs = m.contourf(lon2,lat2,benefit_rawmask,limit2,
#                   extend='both',latlon=True)
cs = m.pcolormesh(lon2,lat2,benefit_rawmask,vmin=0,vmax=30,latlon=True)  
m.scatter(lonloc,latloc,marker='*',color='aqua',latlon=True,s=35) 
cs.set_cmap(cmap)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
m.drawcoastlines(color='darkgrey',linewidth=0.5,zorder=12)
m.drawstates(color='darkgrey',linewidth=0.5,zorder=12)
m.drawcountries(color='darkgrey',linewidth=1,zorder=12)

plt.title(r'\textbf{[a] Raw Data Overshoot Benefit}',fontsize=11,color='k')

###############################################################################
ax = plt.subplot(132)
for txt in fig.texts:
    txt.set_visible(False)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

### Make the plot continuous
benefit_thresholdm = benefit_threshold.copy()
benefit_thresholdm[np.where(benefit_thresholdm > -10)] = np.nan
benefit_thresholdmask = (benefit_thresholdm*-1) - 10

# cs = m.contourf(lon2,lat2,benefit_thresholdmask,limit2,
#                 extend='both',latlon=True)    
cs = m.pcolormesh(lon2,lat2,benefit_thresholdmask,vmin=0,vmax=30,latlon=True)  
m.scatter(lonloc,latloc,marker='*',color='aqua',latlon=True,s=35)                     
cs.set_cmap(cmap)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
m.drawcoastlines(color='darkgrey',linewidth=0.5,zorder=12)
m.drawstates(color='darkgrey',linewidth=0.5,zorder=12)
m.drawcountries(color='darkgrey',linewidth=1,zorder=12)

plt.title(r'\textbf{[b] Threshold [$\geq$%s] Overshoot Benefit}' % YrThreshN,fontsize=11,color='k')

###############################################################################
ax = plt.subplot(133)
for txt in fig.texts:
    txt.set_visible(False)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
benefit_filteringm = benefit_filtering.copy()
benefit_filteringm[np.where(benefit_filteringm > -10)] = np.nan
benefit_filteringmask = (benefit_filteringm*-1) - 10

### Make the plot continuous
# cs = m.contourf(lon2,lat2,benefit_filteringmask,limit2,
#                 extend='both',latlon=True)     
cs = m.pcolormesh(lon2,lat2,benefit_filteringmask,vmin=0,vmax=30,latlon=True)  
m.scatter(lonloc,latloc,marker='*',color='aqua',latlon=True,s=35)                 
cs.set_cmap(cmap)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
m.drawcoastlines(color='darkgrey',linewidth=0.5,zorder=12)
m.drawstates(color='darkgrey',linewidth=0.5,zorder=12)
m.drawcountries(color='darkgrey',linewidth=1,zorder=12)

plt.title(r'\textbf{[c] Smoothed [10 year] Overshoot Benefit}',fontsize=11,color='k')

###############################################################################
cbar_ax1 = fig.add_axes([0.35,0.08,0.3,0.03])                
cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='k',labelpad=1.4)  
cbar1.set_ticks(barlim2)
cbar1.set_ticklabels(list(map(str,barlim2)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()

ax.annotate(r'\textbf{ADDED BENEFIT BY MITIGATING 10 YEARS EARLIER}',xy=(0,0),xytext=(0.5,0.153),
              textcoords='figure fraction',color='dimgrey',fontsize=16,
              rotation=0,ha='center',va='center')

plt.savefig(directoryfigure + 'BenefitTimeSeries_%s_ExampleMap_%s-%s_%s.png' % (variq,latloc,lonloc,monthlychoice),dpi=300)
