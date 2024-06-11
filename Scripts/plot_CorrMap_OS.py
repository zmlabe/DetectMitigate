"""
Plot map of correlations with mean heatwaves

Author    : Zachary M. Labe
Date      : 10 June 2024
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

variq = 'WA'
varcount = 'count90'
numOfEns = 30
numOfEns_10ye = 30
yearsh = np.arange(1921,2014+1,1)
years = np.arange(1921,2100+1)
yearsf = np.arange(2015,2100+1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario']
slicemonthnamen = ['JJA']
monthlychoice = slicemonthnamen[0]

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
### Read in data
lat_bounds,lon_bounds = UT.regions('US')
spear_mALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_hALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osmALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)

### Mask over the USA
spear_m,maskobs = dSS.mask_CONUS(spear_mALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_h,maskobs = dSS.mask_CONUS(spear_hALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
spear_osm,maskobs = dSS.mask_CONUS(spear_osmALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)

yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)

spear_ah = spear_h - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_am = spear_m - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_aosm = spear_osm - climoh_spear[np.newaxis,np.newaxis,:,:]

###############################################################################
###############################################################################
###############################################################################
### Read in data for OS daily extremes
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + 'TMAX' + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
count90 = data.variables[varcount][:,-86:,:,:]
data.close()

### Read in SPEAR_MED_SSP534OS
directorydatahHEAT = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_osHEAT = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + 'TMAX' + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
filename_osHEAT = directorydatahHEAT + name_osHEAT
data_osHEAT = Dataset(filename_osHEAT)
count90_osHEAT = data_osHEAT.variables[varcount][:,4:,:,:] # Need to start in 2015, not 2011
latus = data_osHEAT.variables['lat'][:]
lonus = data_osHEAT.variables['lon'][:]
data_osHEAT.close()

### Meshgrid for the CONUS
lonus2,latus2 = np.meshgrid(lonus,latus)

### Calculate mean timeseries over CONUS
mean_HEAT = UT.calc_weightedAve(count90,latus2)
mean_osHEAT = UT.calc_weightedAve(count90_osHEAT,latus2)
mean_varh = UT.calc_weightedAve(spear_ah,latus2)
mean_varm = UT.calc_weightedAve(spear_am,latus2)
mean_varos = UT.calc_weightedAve(spear_aosm,latus2)

### Calculate correlations
corr_m = np.empty((count90_osHEAT.shape[0],latus2.shape[0],lonus2.shape[1]))
corr_os = np.empty((count90_osHEAT.shape[0],latus2.shape[0],lonus2.shape[1]))
pval_m = np.empty((count90_osHEAT.shape[0],latus2.shape[0],lonus2.shape[1]))
pval_os = np.empty((count90_osHEAT.shape[0],latus2.shape[0],lonus2.shape[1]))
for e in range(count90_osHEAT.shape[0]):
    for i in range(latus2.shape[0]):
        for j in range(lonus2.shape[1]):
            
            if np.isfinite(np.max(spear_am[e,:,i,j])) == True:
                corr_m[e,i,j],pval_m[e,i,j] = sts.pearsonr(mean_HEAT[e,:],spear_am[e,:,i,j])
                corr_os[e,i,j],pval_os[e,i,j] = sts.pearsonr(mean_osHEAT[e,:],spear_aosm[e,:,i,j])
            else:
                corr_m[e,i,j] = np.nan
                pval_m[e,i,j] = np.nan
                corr_os[e,i,j] = np.nan
                pval_os[e,i,j] = np.nan

### Calculate ensemble means
meancorr_m = np.nanmean(corr_m,axis=0)
meancorr_os = np.nanmean(corr_os,axis=0)

###############################################################################
###############################################################################
###############################################################################               
### Plot Figure
fig = plt.figure(figsize=(10,4))
for txt in fig.texts:
    txt.set_visible(False)

### Colorbar limits
barlim = np.round(np.arange(-1,1.1,0.2),2)
limit = np.arange(-1,1.01,0.05)
label = r'\textbf{Correlation - %s}' % variq 

ax = plt.subplot(121)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
            area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lonus2,latus2,meancorr_m,limit,
                  extend='both',latlon=True)
# cs3 = m.contourf(lon2,lat2,pval_os,colors='None',hatches=['.....'],latlon=True)
                
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) SSP5-8.5}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(122)
    
### Select map type
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
            area_thresh=10000)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lonus2,latus2,meancorr_os,limit,
                  extend='both',latlon=True)
# cs3 = m.contourf(lon2,lat2,pval_os_10ye,colors='None',hatches=['.....'],latlon=True)
              
cmap = cmocean.cm.balance
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) SSP5-3.4OS}',fontsize=11,color='dimgrey')

###############################################################################
cbar_ax1 = fig.add_axes([0.355,0.08,0.3,0.03])                
cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.13)

plt.savefig(directoryfigure + 'CorrMap_os_%s.png' % (variq),dpi=300)
