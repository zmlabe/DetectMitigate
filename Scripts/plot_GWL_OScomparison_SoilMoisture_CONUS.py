"""
Calculate trend for OS for water_soil for just the CONUS

Author    : Zachary M. Labe
Date      : 14 March 2024
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

variablesall = ['water_soil']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
years = np.arange(2015,2100+1)
yearsh = np.arange(1921,2014+1,1)
yearsall = np.arange(1921,2100+1,1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
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
def findNearestValueIndex(array,value):
    index = (np.abs(array-value)).argmin()
    return index
###############################################################################
###############################################################################
###############################################################################
### Get data
selectGWL = 1.7
selectGWLn = '%s' % (int(selectGWL*10))
yrplus = 3

if variq == 'T2M':
    lat_bounds,lon_bounds = UT.regions(reg_name)
    spear_m,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_h,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_osm,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
    spear_osm_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
    lon2,lat2 = np.meshgrid(lons,lats)
    
    yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
    climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)
    
    spear_ah = spear_h - climoh_spear[np.newaxis,np.newaxis,:,:]
    spear_am = spear_m - climoh_spear[np.newaxis,np.newaxis,:,:]
    spear_aosm = spear_osm - climoh_spear[np.newaxis,np.newaxis,:,:]
    spear_aosm_10ye = spear_osm_10ye - climoh_spear[np.newaxis,np.newaxis,:,:]
    
    ### Calculate global average in SPEAR_MED
    lon2,lat2 = np.meshgrid(lons,lats)
    spear_ah_globeh = UT.calc_weightedAve(spear_ah,lat2)
    spear_am_globeh = UT.calc_weightedAve(spear_am,lat2)
    spear_osm_globeh = UT.calc_weightedAve(spear_aosm,lat2)
    spear_osm_10ye_globeh = UT.calc_weightedAve(spear_aosm_10ye,lat2)
    
    ### Calculate GWL for ensemble means
    gwl_spearh = np.nanmean(spear_ah_globeh,axis=0)
    gwl_spearf = np.nanmean(spear_am_globeh,axis=0)
    gwl_os = np.nanmean(spear_osm_globeh,axis=0)
    gwl_os_10ye = np.nanmean(spear_osm_10ye_globeh,axis=0)
    
    ### Combined gwl
    gwl_all = np.append(gwl_spearh,gwl_spearf,axis=0)
    
    ### Calculate overshoot times
    os_yr = np.where((years == 2040))[0][0]
    os_10ye_yr = np.where((years == 2031))[0][0]
    
    ### Find year of selected GWL
    ssp_GWL = findNearestValueIndex(gwl_spearf,selectGWL)
    
    os_first_GWL = findNearestValueIndex(gwl_os[:os_yr],selectGWL)
    os_second_GWL = findNearestValueIndex(gwl_os[os_yr:],selectGWL)+(len(years)-len(years[os_yr:]))
    
    os_10ye_first_GWL = findNearestValueIndex(gwl_os_10ye[:os_yr],selectGWL) # need to account for further warming after 2031 to reach 1.5C
    os_10ye_second_GWL = findNearestValueIndex(gwl_os_10ye[os_yr:],selectGWL)+(len(years)-len(years[os_yr:])) # need to account for further warming after 2031 to reach 1.5C
    
    ### Epochs for +- years around selected GWL
    climatechange_GWL = np.nanmean(spear_am[:,ssp_GWL-yrplus:ssp_GWL+yrplus,:,:],axis=(0,1))
    os_GWL = np.nanmean(spear_aosm[:,os_second_GWL-yrplus:os_second_GWL+yrplus,:,:],axis=(0,1))
    os_10ye_GWL = np.nanmean(spear_aosm_10ye[:,os_10ye_second_GWL-yrplus:os_10ye_second_GWL+yrplus,:,:],axis=(0,1))
    
    ### Differences at selected GWL
    diff_os = os_GWL - climatechange_GWL
    diff_os_10ye = os_10ye_GWL - climatechange_GWL
    
    ### Calculate statistical significance (FDR)
    alpha_f = 0.05
    varx_os = np.nanmean(spear_am[:,ssp_GWL-yrplus:ssp_GWL+yrplus,:,:],axis=(1))
    vary_os = np.nanmean(spear_aosm[:,os_second_GWL-yrplus:os_second_GWL+yrplus,:,:],axis=(1))
    vary_os_10ye = np.nanmean(spear_aosm_10ye[:,os_10ye_second_GWL-yrplus:os_10ye_second_GWL+yrplus,:,:],axis=(1))
    pval_os = UT.calc_FDR_ttest(varx_os,vary_os,alpha_f)
    pval_os_10ye = UT.calc_FDR_ttest(varx_os,vary_os_10ye,alpha_f)
    
else:
    lat_bounds,lon_bounds = UT.regions(reg_name)
    spear_m,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_h,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_osm,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
    spear_osm_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)

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
    
    ### Calculate overshoot times
    os_yr = np.where((years == 2040))[0][0]
    os_10ye_yr = np.where((years == 2031))[0][0]
    
    ### Find year of selected GWL
    ssp_GWLt = findNearestValueIndex(gwl_spearft,selectGWL)
    ssp_GWL = ssp_GWLt
    
    os_first_GWLt = findNearestValueIndex(gwl_ost[:os_yr],selectGWL)
    os_second_GWLt = findNearestValueIndex(gwl_ost[os_yr:],selectGWL)+(len(years)-len(years[os_yr:]))
    os_first_GWL = os_first_GWLt
    os_second_GWL = os_second_GWLt
    
    os_10ye_first_GWLt = findNearestValueIndex(gwl_os_10yet[:os_yr],selectGWL) # need to account for further warming after 2031 to reach 1.5C
    os_10ye_second_GWLt = findNearestValueIndex(gwl_os_10yet[os_yr:],selectGWL)+(len(years)-len(years[os_yr:])) # need to account for further warming after 2031 to reach 1.5C
    os_10ye_first_GWL = os_10ye_first_GWLt
    os_10ye_second_GWL = os_10ye_second_GWLt 
    
    ### Epochs for +- years around selected GWL
    climatechange_GWL = np.nanmean(spear_am[:,ssp_GWLt-yrplus:ssp_GWLt+yrplus,:,:],axis=(0,1))
    os_GWL = np.nanmean(spear_aosm[:,os_second_GWLt-yrplus:os_second_GWLt+yrplus,:,:],axis=(0,1))
    os_10ye_GWL = np.nanmean(spear_aosm_10ye[:,os_10ye_second_GWLt-yrplus:os_10ye_second_GWLt+yrplus,:,:],axis=(0,1))
    
    ### Differences at selected GWL
    diff_os = os_GWL - climatechange_GWL
    diff_os_10ye = os_10ye_GWL - climatechange_GWL
    
    ### Calculate statistical significance (FDR)
    alpha_f = 0.05
    varx_os = np.nanmean(spear_am[:,ssp_GWL-yrplus:ssp_GWL+yrplus,:,:],axis=(1))
    vary_os = np.nanmean(spear_aosm[:,os_second_GWL-yrplus:os_second_GWL+yrplus,:,:],axis=(1))
    vary_os_10ye = np.nanmean(spear_aosm_10ye[:,os_10ye_second_GWL-yrplus:os_10ye_second_GWL+yrplus,:,:],axis=(1))
    pval_os = UT.calc_FDR_ttest(varx_os,vary_os,alpha_f)
    pval_os_10ye = UT.calc_FDR_ttest(varx_os,vary_os_10ye,alpha_f)

###############################################################################
###############################################################################
###############################################################################
### Plot figure

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
barlim = np.arange(-0.5,0.51,0.25)
limit = np.arange(-0.5,0.51,0.01)
barlim2 = np.arange(-0.5,0.51,0.25)
limit2 = np.arange(-0.5,0.51,0.01)  
label = r'\textbf{SOIL MOISTURE CHANGE [m]}' 

### Map world map
fig = plt.figure(figsize=(10,4))
ax = plt.subplot(231)
for txt in fig.texts:
    txt.set_visible(False)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,climatechange_GWL/1000,limit,
                  extend='both',latlon=True)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)  
                
cmap = cmr.seasons_r    
cs.set_cmap(cmap)

plt.title(r'\textbf{(a); %s$^{\circ}$C [%s] for SSP5-8.5}' % (selectGWL,years[ssp_GWL]),fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(232)
for txt in fig.texts:
    txt.set_visible(False)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,os_GWL/1000,limit,
                extend='both',latlon=True)                        
cs.set_cmap(cmap)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)  

plt.title(r'\textbf{(b); %s$^{\circ}$C [%s] for SSP5-3.4OS}' % (selectGWL,years[os_second_GWL]),fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(233)
for txt in fig.texts:
    txt.set_visible(False)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs = m.contourf(lon2,lat2,os_10ye_GWL/1000,limit,
                extend='both',latlon=True)                        
cs.set_cmap(cmap)

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)  

plt.title(r'\textbf{(c); %s$^{\circ}$C [%s] for SSP5-3.4OS_10ye}' % (selectGWL,years[os_10ye_second_GWL]),fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(235)
for txt in fig.texts:
    txt.set_visible(False)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs2 = m.contourf(lon2,lat2,diff_os/1000,limit2,
                extend='both',latlon=True) 
cs2.set_cmap(cmap)

pval_os[np.where(np.isnan(pval_os))] = 0.     
pval_os[np.where(pval_os == 1)] = np.nan   
pval_os[np.where(pval_os == 0)] = 1. 
cs3 = m.contourf(lon2,lat2,pval_os,colors='None',hatches=['/////////'],latlon=True)   

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)               

plt.title(r'\textbf{(d); (b) minus (a)}',fontsize=11,color='dimgrey')

###############################################################################
ax = plt.subplot(236)
for txt in fig.texts:
    txt.set_visible(False)

circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)
m.drawcoastlines(color='darkgrey',linewidth=1)
m.drawstates(color='darkgrey',linewidth=0.5)
m.drawcountries(color='darkgrey',linewidth=0.5)

### Make the plot continuous
cs2 = m.contourf(lon2,lat2,diff_os_10ye/1000,limit2,
                extend='both',latlon=True)   
cs2.set_cmap(cmap) 
 
pval_os_10ye[np.where(np.isnan(pval_os_10ye))] = 0.     
pval_os_10ye[np.where(pval_os_10ye == 1)] = np.nan   
pval_os_10ye[np.where(pval_os_10ye == 0)] = 1. 
cs3 = m.contourf(lon2,lat2,pval_os_10ye,colors='None',hatches=['/////////'],latlon=True)      

m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgray',lakes=False,zorder=11)                 

plt.title(r'\textbf{(e); (c) minus (a)}',fontsize=11,color='dimgrey')

cbar_axg = fig.add_axes([0.94,0.61,0.013,0.25])                
cbarg = fig.colorbar(cs,cax=cbar_axg,orientation='vertical',
                    extend='both',extendfrac=0.07,drawedges=False) 
cbarg.set_label(label,fontsize=8,color='k',labelpad=12)
cbarg.set_ticks(barlim)
cbarg.set_ticklabels(list(map(str,np.round(barlim,2)))) 
cbarg.ax.tick_params(axis='y', size=.01,labelsize=7)
cbarg.outline.set_edgecolor('dimgrey')

cbar_ax = fig.add_axes([0.94,0.14,0.013,0.25])                
cbar = fig.colorbar(cs2,cax=cbar_ax,orientation='vertical',
                    extend='both',extendfrac=0.07,drawedges=False) 
cbar.set_label(label,fontsize=8,color='k',labelpad=8)  
cbar.set_ticks(barlim2)
cbar.set_ticklabels(list(map(str,np.round(barlim2,2)))) 
cbar.ax.tick_params(axis='y', size=.01,labelsize=7)
cbar.outline.set_edgecolor('dimgrey')  

### Save figure 
plt.tight_layout()   
fig.subplots_adjust(right=0.93)
plt.savefig(directoryfigure + 'GWL-%s_%s_%s_CONUS.png' % (selectGWLn,variq,seasons[0]),dpi=300)
