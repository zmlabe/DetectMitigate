"""
Calculate trend for soil moisture in OS runs

Author    : Zachary M. Labe
Date      : 2 May 2024
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

variq = 'water_soil'
variablesglobe = 'T2M'
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
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'Globe'

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
### Read in data
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_m,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_h,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osm,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)

### Units conversion
spear_m = spear_m/1000.
spear_h = spear_h/1000.
spear_osm = spear_osm/1000.
spear_osm_10ye = spear_osm_10ye/1000.

spear_mt,lats,lons = read_primary_dataset('T2M','SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_ht,lats,lons = read_primary_dataset('T2M','SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osmt,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10yet,lats,lons = read_primary_dataset('T2M','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)
climoh_speart = np.nanmean(np.nanmean(spear_ht[:,yearq,:,:],axis=1),axis=0)

spear_ah = spear_h - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_am = spear_m - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_aosm = spear_osm - climoh_spear[np.newaxis,np.newaxis,:,:]
spear_aosm_10ye = spear_osm_10ye - climoh_spear[np.newaxis,np.newaxis,:,:]

spear_aht = spear_ht - climoh_speart[np.newaxis,np.newaxis,:,:]
spear_amt = spear_mt - climoh_speart[np.newaxis,np.newaxis,:,:]
spear_aosmt = spear_osmt - climoh_speart[np.newaxis,np.newaxis,:,:]
spear_aosm_10yet = spear_osm_10yet - climoh_speart[np.newaxis,np.newaxis,:,:]

### Calculate global average in SPEAR_MED
lon2,lat2 = np.meshgrid(lons,lats)
spear_ah_globeht = UT.calc_weightedAve(spear_aht,lat2)
spear_am_globeht = UT.calc_weightedAve(spear_amt,lat2)
spear_osm_globeht = UT.calc_weightedAve(spear_aosmt,lat2)
spear_osm_10ye_globeht = UT.calc_weightedAve(spear_aosm_10yet,lat2)

### Calculate GWL for ensemble means
gwl_spearht = np.nanmean(spear_ah_globeht,axis=0)
gwl_spearft = np.nanmean(spear_am_globeht,axis=0)
gwl_ost = np.nanmean(spear_osm_globeht,axis=0)
gwl_os_10yet = np.nanmean(spear_osm_10ye_globeht,axis=0)

### Combined gwl
gwl_allt = np.append(gwl_spearht,gwl_spearft,axis=0)

### Calculate maximum warming
yrmax_allssp585 = np.argmax(gwl_allt)
yrmax_os = np.argmax(gwl_ost)
yrmax_os_10ye = np.argmax(gwl_os_10yet)

### Slice trend periods for only 30 years at a time
trendperiod_os = spear_aosm[:,yrmax_os:yrmax_os+30,:,:]
trendperiod_os_10ye = spear_aosm_10ye[:,yrmax_os_10ye:yrmax_os_10ye+30,:,:]
trendperiod_osm = np.nanmean(trendperiod_os[:,:,:,:],axis=0)
trendperiod_os_10yem = np.nanmean(trendperiod_os_10ye[:,:,:,:],axis=0)

###############################################################################
###############################################################################
###############################################################################
### Calculate trends
trend_os = calcTrend(trendperiod_os)
trend_os10ye = calcTrend(trendperiod_os_10ye)

### Calculate statistical test for ensemble mean - OS
pval_os = np.empty((trendperiod_osm.shape[1],trendperiod_osm.shape[2]))
h_os = np.empty((trendperiod_osm.shape[1],trendperiod_osm.shape[2]))
for i in range(trendperiod_osm.shape[1]):
    for j in range(trendperiod_osm.shape[2]):
        trendagain_os,h_os[i,j],pval_os[i,j],z_os = UT.mk_test(trendperiod_osm[:,i,j],0.05)
        
pval_os[np.where(pval_os == 1.)] = 0.
pval_os[np.where(np.isnan(pval_os))] = 1.
pval_os[np.where(pval_os == 0.)] = np.nan

### Calculate statistical test for ensemble mean - OS_10ye
pval_os_10ye = np.empty((trendperiod_os_10yem.shape[1],trendperiod_os_10yem.shape[2]))
h_os_10ye = np.empty((trendperiod_os_10yem.shape[1],trendperiod_os_10yem.shape[2]))
for i in range(trendperiod_os_10yem.shape[1]):
    for j in range(trendperiod_os_10yem.shape[2]):
        trendagain_10ye,h_os_10ye[i,j],pval_os_10ye[i,j],z_os = UT.mk_test(trendperiod_os_10yem[:,i,j],0.05)
        
# pval_os_10ye[np.where(pval_os_10ye == 1.)] = 0.
# pval_os_10ye[np.where(np.isnan(pval_os_10ye))] = 1.
# pval_os_10ye[np.where(pval_os_10ye == 0.)] = np.nan

### Calculate ensemble means
trendmean_os = np.nanmean(trend_os[:,:,:],axis=0)
trendmean_os10ye = np.nanmean(trend_os10ye[:,:,:],axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot figure

fig = plt.figure(figsize=(8,3))
for txt in fig.texts:
    txt.set_visible(False)

### Colorbar limits
barlim = np.round(np.arange(-0.1,0.11,0.1),2)
limit = np.arange(-0.1,0.101,0.001)
label = r'\textbf{Trend in soil moisture [m/decade] - 30 years after peak warming}' 

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
cs = m.contourf(lon2,lat2,trendmean_os,limit,
                  extend='both',latlon=True)
cs3 = m.contourf(lon2,lat2,pval_os,colors='None',hatches=['.....'],latlon=True)
                
cmap = cmr.seasons_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) SSP5-3.4OS}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,trendmean_os10ye,limit,
                  extend='both',latlon=True)
cs3 = m.contourf(lon2,lat2,pval_os_10ye,colors='None',hatches=['.....'],latlon=True)
              
cmap = cmr.seasons_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) SSP5-3.4OS_10ye}',fontsize=11,color='dimgrey')

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

plt.savefig(directoryfigure + 'Trends_OS_SoilMoisture_%s_CONUS.png' % (monthlychoice),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Slice trend periods for the last 30 years
trendperiod_os = spear_aosm[:,-30:,:,:]
trendperiod_os_10ye = spear_aosm_10ye[:,-30:,:,:]
trendperiod_osm = np.nanmean(trendperiod_os[:,:,:,:],axis=0)
trendperiod_os_10yem = np.nanmean(trendperiod_os_10ye[:,:,:,:],axis=0)

###############################################################################
###############################################################################
###############################################################################
### Calculate trends
trend_os = calcTrend(trendperiod_os)
trend_os10ye = calcTrend(trendperiod_os_10ye)

### Calculate statistical test for ensemble mean - OS
pval_os = np.empty((trendperiod_osm.shape[1],trendperiod_osm.shape[2]))
h_os = np.empty((trendperiod_osm.shape[1],trendperiod_osm.shape[2]))
for i in range(trendperiod_osm.shape[1]):
    for j in range(trendperiod_osm.shape[2]):
        trendagain_os,h_os[i,j],pval_os[i,j],z_os = UT.mk_test(trendperiod_osm[:,i,j],0.05)
        
# pval_os[np.where(pval_os == 1.)] = 0.
# pval_os[np.where(np.isnan(pval_os))] = 1.
# pval_os[np.where(pval_os == 0.)] = np.nan

### Calculate statistical test for ensemble mean - OS_10ye
pval_os_10ye = np.empty((trendperiod_os_10yem.shape[1],trendperiod_os_10yem.shape[2]))
h_os_10ye = np.empty((trendperiod_os_10yem.shape[1],trendperiod_os_10yem.shape[2]))
for i in range(trendperiod_os_10yem.shape[1]):
    for j in range(trendperiod_os_10yem.shape[2]):
        trendagain_10ye,h_os_10ye[i,j],pval_os_10ye[i,j],z_os = UT.mk_test(trendperiod_os_10yem[:,i,j],0.05)
        
# pval_os_10ye[np.where(pval_os_10ye == 1.)] = 0.
# pval_os_10ye[np.where(np.isnan(pval_os_10ye))] = 1.
# pval_os_10ye[np.where(pval_os_10ye == 0.)] = np.nan

### Calculate ensemble means
trendmean_os = np.nanmean(trend_os[:,:,:],axis=0)
trendmean_os10ye = np.nanmean(trend_os10ye[:,:,:],axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot figure

fig = plt.figure(figsize=(8,3))
for txt in fig.texts:
    txt.set_visible(False)

### Colorbar limits
barlim = np.round(np.arange(-0.1,0.11,0.1),2)
limit = np.arange(-0.1,0.101,0.001)
label = r'\textbf{Trend in soil moisture [m/decade] - 2071-2100}' 

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
cs = m.contourf(lon2,lat2,trendmean_os,limit,
                  extend='both',latlon=True)
cs3 = m.contourf(lon2,lat2,pval_os,colors='None',hatches=['.....'],latlon=True)
                
cmap = cmr.seasons_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(a) SSP5-3.4OS}',fontsize=11,color='dimgrey')

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
cs = m.contourf(lon2,lat2,trendmean_os10ye,limit,
                  extend='both',latlon=True)
cs3 = m.contourf(lon2,lat2,pval_os_10ye,colors='None',hatches=['.....'],latlon=True)
              
cmap = cmr.seasons_r
cs.set_cmap(cmap)

plt.title(r'\textbf{(b) SSP5-3.4OS_10ye}',fontsize=11,color='dimgrey')

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

plt.savefig(directoryfigure + 'Trends_OS_SoilMoisture_%s_CONUS_last30yrs.png' % (monthlychoice),dpi=300)

