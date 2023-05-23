"""
Calculate trend for OS 

Author    : Zachary M. Labe
Date      : 22 May 2023
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

variablesall = ['PRECT']
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
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
dataset_obs = 'ERA5_MEDS'
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
spear_os,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_os_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Calculate epochs
os_yr = np.where((years == 2040))[0][0]
os_10ye_yr = np.where((years == 2031))[0][0]

### Calculate ensemble means
spear_osm = np.nanmean(spear_os,axis=0)
spear_osm_10ye = np.nanmean(spear_os_10ye,axis=0)

### First epoch around overshoot beginning
spear_os1 = np.nanmean(spear_osm[os_yr-5:os_yr+5,:,:],axis=0)
spear_os1_10ye = np.nanmean(spear_osm_10ye[os_10ye_yr-5:os_10ye_yr+5,:,:],axis=0)

### Second epoch around overshoot ending 50 years later
end = 10
spear_os2 = np.nanmean(spear_osm[end+os_yr-5:end+os_yr+5,:,:],axis=0)
spear_os2_10ye = np.nanmean(spear_osm_10ye[end+os_10ye_yr-5:end+os_10ye_yr+5,:,:],axis=0)

### Calculate changes in epochs
change_os = spear_os2 - spear_os1
change_os_10ye = spear_os2_10ye - spear_os1_10ye

###############################################################################
###############################################################################
###############################################################################
### Plot figure

### Define parameters (dark)
def setcolor(x, color):
     for m in x:
         for t in x[m][1]:
             t.set_color(color)

### Select map type
style = 'global'

if style == 'ortho':
    m = Basemap(projection='ortho',lon_0=-90,
                lat_0=70,resolution='l',round=True)
elif style == 'polar':
    m = Basemap(projection='npstere',boundinglat=67,lon_0=270,resolution='l',round =True)
elif style == 'global':
    m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)

### Map world map
fig = plt.figure(figsize=(10,4))
ax = plt.subplot(121)
for txt in fig.texts:
    txt.set_visible(False)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='dimgrey',linewidth=0.7)

### Colorbar limits
if variq == 'T2M':
    barlim = np.arange(-2,2,1)
    limit = np.arange(-2,2.1,0.1)
elif variq == 'PRECT':
    barlim = np.arange(-1,1.1,0.5)
    limit = np.arange(-1,1.01,0.05)

### Make the plot continuous
cs = m.contourf(lon2,lat2,change_os,limit,
                  extend='both',latlon=True)
                
if variq == 'T2M':
    cmap = cmocean.cm.balance    
elif variq == 'PRECT':
    cmap = cmr.seasons_r       
cs.set_cmap(cmap)

plt.title(r'\textbf{OS in %s years}' % end,fontsize=11,color='k')

###############################################################################
ax = plt.subplot(122)
for txt in fig.texts:
    txt.set_visible(False)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='dimgrey',linewidth=0.7)

### Make the plot continuous
cs = m.contourf(lon2,lat2,change_os_10ye,limit,
                extend='both',latlon=True)
                         
cs.set_cmap(cmap)

plt.title(r'\textbf{OS_10ye in %s years}' % end,fontsize=11,color='k')

cbar_ax1 = fig.add_axes([0.35,0.12,0.3,0.03])                
cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
if variq == 'PRECT':
    label = r'\textbf{PRECIPITATION CHANGE [mm/day]}' 
elif variq == 'T2M':
    label = r'\textbf{TEMPERATURE CHANGE [$^{\circ}$C]}' 
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('darkgrey')

### Save figure 
plt.tight_layout()   
plt.savefig(directoryfigure + 'changeAfterOS_%s_in_%syrs.png' % (variq,end),dpi=300)
