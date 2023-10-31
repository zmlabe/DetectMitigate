"""
Evaluate composites during the transition periods

Author     : Zachary M. Labe
Date       : 27 October 2023
Version    : 4
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
from sklearn.metrics import accuracy_score
import cmasher as cmr
import cmocean
from scipy.ndimage import gaussian_filter
import calc_Utilities as UT
import scipy.stats as sts
import scipy.stats.mstats as mstats
import calc_dataFunctions as df

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
seasons = ['annual']
slicemonthnamen = ['ANNUAL']
monthlychoice = seasons[0]
reg_name = 'Globe'

### Figure parameters
allvariables = ['Temperature','Temperature','Temperature','Precipitation','Precipitation','Precipitation']

###############################################################################
###############################################################################
###############################################################################
### Read in climate models      
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons 

### Read in data
lat_bounds,lon_bounds = UT.regions(reg_name)
os_t2m,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
os10ye_t2m,lats,lons = read_primary_dataset('T2M','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)

os_precip,lats,lons = read_primary_dataset('PRECT','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
os10ye_precip,lats,lons = read_primary_dataset('PRECT','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)

lon2,lat2 = np.meshgrid(lons,lats)

### Calculate ensemble means
os_t2m_mean = np.nanmean(os_t2m[:,:,:,:],axis=0)
os10ye_t2m_mean = np.nanmean(os_t2m[:,:,:,:],axis=0)
os_precip_mean = np.nanmean(os_precip[:,:,:,:],axis=0)
os10ye_precip_mean = np.nanmean(os_precip[:,:,:,:],axis=0)

fig = plt.figure(figsize=(10,4.5))
transitionsLength = 6
for pp in range(transitionsLength):
    ### Slice periods and take difference (+- 5 years)
    years = np.arange(2015,2100+1,1)
    sliceperiod = 5
    
    if pp == 0:
        variq = 'T2M'
        yearq_1 = np.where((years == 2052))[0][0]
        yearq_2 = np.where((years == 2063))[0][0]
        
        lrp_1 = os_t2m_mean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = os_t2m_mean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################
    elif pp == 1:
        variq = 'T2M'
        yearq_1 = np.where((years == 2048))[0][0]
        yearq_2 = np.where((years == 2056))[0][0]
        
        lrp_1 = os10ye_t2m_mean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = os10ye_t2m_mean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################        
    elif pp == 2:
        variq = 'T2M'
        yearq_1 = np.where((years == 2073))[0][0]
        yearq_2 = np.where((years == 2084))[0][0]
        
        lrp_1 = os10ye_t2m_mean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = os10ye_t2m_mean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################       
    elif pp == 3:
        variq == 'PRECT'
        yearq_1 = np.where((years == 2049))[0][0]
        yearq_2 = np.where((years == 2064))[0][0]
        
        lrp_1 = os_precip_mean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = os_precip_mean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################        
    elif pp == 4:
        variq = 'PRECT'
        yearq_1 = np.where((years == 2045))[0][0]
        yearq_2 = np.where((years == 2051))[0][0]
        
        lrp_1 = os10ye_precip_mean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = os10ye_precip_mean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################        
    elif pp == 5:
        variq = 'PRECT'
        yearq_1 = np.where((years == 2072))[0][0]
        yearq_2 = np.where((years == 2086))[0][0]
        
        lrp_1 = os10ye_precip_mean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = os10ye_precip_mean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################        

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plot subplot of LRP means OS
    limitt = np.arange(-2,2.01,0.05)
    barlimt = np.round(np.arange(-2,3,2),2)
    labelt = r'\textbf{$^{\circ}$C}'
    
    limitp = np.arange(-0.5,0.501,0.01)
    barlimp = np.round(np.arange(-0.5,0.501,0.5),2)
    labelp = r'\textbf{mm/day}'
    
    if pp < 3:
        limit = limitt
        barlim = barlimt
        label = labelt
        cmap = cmocean.cm.balance
    else:
        limit = limitp
        barlim = barlimp
        label = labelp
        cmap = cmr.seasons_r
    
    ### Retrieve variable
    var = lrp_diffmean
    
    ax1 = plt.subplot(2,3,pp+1)
    m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
    m.drawcoastlines(color='dimgrey',linewidth=0.35)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    x,y = m(lon2,lat2)
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    cs1.set_cmap(cmap) 
    
    plt.title(r'\textbf{%s-%s minus %s-%s}' % (min(years[yearq_2:yearq_2+sliceperiod]),max(years[yearq_2:yearq_2+sliceperiod]),min(years[yearq_1-(sliceperiod-1):yearq_1+1]),max(years[yearq_1-(sliceperiod-1):yearq_1+1])),fontsize=10,color='k')

    if any([pp==0,pp==3]):
        ax1.annotate(r'\textbf{%s}' % (allvariables[pp]),xy=(0,0),xytext=(-0.05,0.50),
                  textcoords='axes fraction',color='k',fontsize=15,
                  rotation=90,ha='center',va='center')
            
    ax1.annotate(r'\textbf{[%s]}' % letters[pp],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=8,
                  rotation=330,ha='center',va='center')
    
    if pp == 2:
        cbar_ax1 = fig.add_axes([0.92,0.57,0.013,0.25])                 
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='vertical',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=10,color='k',labelpad=4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='y', size=.01,labelsize=7)
        cbar1.outline.set_edgecolor('dimgrey')
    elif pp == 5:
        cbar_ax1 = fig.add_axes([0.92,0.16,0.013,0.25])                 
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='vertical',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=10,color='k',labelpad=5.5)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='y', size=.01,labelsize=7)
        cbar1.outline.set_edgecolor('dimgrey')
   
ax1.annotate(r'\textbf{SPEAR_MED_SSP534OS}',xy=(0,0),xytext=(-1.52,2.7),
              textcoords='axes fraction',color='dimgrey',fontsize=18,
              rotation=0,ha='center',va='center')
ax1.annotate(r'\textbf{SPEAR_MED_SSP534OS_10ye}',xy=(0,0),xytext=(-0.02,2.7),
              textcoords='axes fraction',color='dimgrey',fontsize=18,
              rotation=0,ha='center',va='center')

plt.tight_layout()
plt.subplots_adjust(wspace=0.02,hspace=0.06,bottom=0.1)

plt.savefig(directoryfigure + 'MSFigure_ANN_Composites_Transitions.png',dpi=600)
