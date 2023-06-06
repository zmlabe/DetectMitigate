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
selectGWL = 1.5
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
    climoh_spear_zonal = np.nanmean(climoh_spear[:,:],axis=1)
    
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
    
    ### Calculate zonal means
    climatechange_GWL_zonal = np.nanmean(climatechange_GWL[:,:],axis=1)
    os_GWL_zonal = np.nanmean(os_GWL[:,:],axis=1)
    os_10ye_GWL_zonal = np.nanmean(os_10ye_GWL[:,:],axis=1)
else:
    lat_bounds,lon_bounds = UT.regions(reg_name)
    spear_m,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_h,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_osm,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
    spear_osm_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
    lon2,lat2 = np.meshgrid(lons,lats)
    
    spear_mt,lats,lons = read_primary_dataset('T2M','SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_ht,lats,lons = read_primary_dataset('T2M','SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_osmt,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
    spear_osm_10yet,lats,lons = read_primary_dataset('T2M','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
    
    yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
    climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)
    climoh_spear_zonal = np.nanmean(climoh_spear[:,:],axis=1)
    climoh_speart = np.nanmean(np.nanmean(spear_ht[:,yearq,:,:],axis=1),axis=0)
    climoh_spear_zonalt = np.nanmean(climoh_speart[:,:],axis=1)
    
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
    
    ### Calculate zonal means
    climatechange_GWL_zonal = np.nanmean(climatechange_GWL[:,:],axis=1)
    os_GWL_zonal = np.nanmean(os_GWL[:,:],axis=1)
    os_10ye_GWL_zonal = np.nanmean(os_10ye_GWL[:,:],axis=1)

###############################################################################
###############################################################################
###############################################################################               
### Plot Figure
### Adjust axes in time series plots 
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
        
if variq == 'T2M':
    fig = plt.figure()
    ax = plt.subplot(111)
    
    adjust_spines(ax, ['left', 'bottom'])            
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4.,width=2,which='major',color='dimgrey')
    ax.tick_params(axis='x',labelsize=6,pad=1.5)
    ax.tick_params(axis='y',labelsize=6,pad=1.5)
    ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.8)
    
    plt.plot(lats,climatechange_GWL_zonal,linestyle='-',color='maroon',
              clip_on=False,zorder=1,label=r'\textbf{%s$^{\circ}$C [%s] for SSP5-8.5}' % (selectGWL,years[ssp_GWL]),linewidth=2)
    plt.plot(lats,os_GWL_zonal,linestyle='-',color='darkslategrey',
              clip_on=False,zorder=1,label=r'\textbf{%s$^{\circ}$C [%s] for SSP5-3.4OS}' % (selectGWL,years[os_second_GWL]),linewidth=2)
    plt.plot(lats,os_10ye_GWL_zonal,linestyle='--',color='teal',
              clip_on=False,zorder=1,label=r'\textbf{%s$^{\circ}$C [%s] for SSP5-3.4OS_10ye}' % (selectGWL,years[os_10ye_second_GWL]),
              dashes=(1,0.3),linewidth=2)
    
    leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
          bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
          handlelength=1,handletextpad=0.5)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    plt.xticks(np.arange(-90,91,10),np.arange(-90,91,10))
    plt.yticks(np.round(np.arange(-18,18.1,1),2),np.round(np.arange(-18,18.1,1),2))
    plt.xlim([-90,90])
    plt.ylim([0,5])
    
    plt.ylabel(r'\textbf{Near-Surface Temperature Anomaly [$^{\circ}$C; 1921-1950]}',
                fontsize=10,color='k')
    plt.xlabel(r'\textbf{Latitude [$^{\circ}$]}',color='k',fontsize=10)
    
    plt.tight_layout()
    plt.savefig(directoryfigure + 'GWL-%s_ZONAL_%s.png' % (selectGWLn,variq),dpi=300)
    
elif variq == 'PRECT':
    fig = plt.figure()
    ax = plt.subplot(111)
    
    adjust_spines(ax, ['left', 'bottom'])            
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4.,width=2,which='major',color='dimgrey')
    ax.tick_params(axis='x',labelsize=6,pad=1.5)
    ax.tick_params(axis='y',labelsize=6,pad=1.5)
    ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.8)
    
    plt.plot(lats,climatechange_GWL_zonal,linestyle='-',color='maroon',
              clip_on=False,zorder=1,label=r'\textbf{%s$^{\circ}$C [%s] for SSP5-8.5}' % (selectGWL,years[ssp_GWL]),linewidth=2)
    plt.plot(lats,os_GWL_zonal,linestyle='-',color='darkslategrey',
              clip_on=False,zorder=1,label=r'\textbf{%s$^{\circ}$C [%s] for SSP5-3.4OS}' % (selectGWL,years[os_second_GWL]),linewidth=2)
    plt.plot(lats,os_10ye_GWL_zonal,linestyle='--',color='teal',
              clip_on=False,zorder=1,label=r'\textbf{%s$^{\circ}$C [%s] for SSP5-3.4OS_10ye}' % (selectGWL,years[os_10ye_second_GWL]),
              dashes=(1,0.3),linewidth=2)
    
    leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
          bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
          handlelength=1,handletextpad=0.5)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    plt.xticks(np.arange(-90,91,10),np.arange(-90,91,10))
    plt.yticks(np.round(np.arange(-2,3,0.1),2),np.round(np.arange(-2,3,0.1),2))
    plt.xlim([-90,90])
    plt.ylim([-0.2,0.6])
    
    plt.ylabel(r'\textbf{Precipitation Anomaly [mm/day; 1921-1950]}',
                fontsize=10,color='k')
    plt.xlabel(r'\textbf{Latitude [$^{\circ}$]}',color='k',fontsize=10)
    
    plt.tight_layout()
    plt.savefig(directoryfigure + 'GWL-%s_ZONAL_%s.png' % (selectGWLn,variq),dpi=300)
 
###############################################################################
###############################################################################
###############################################################################
if variq == 'PRECT':
    lat_bounds,lon_bounds = UT.regions(reg_name)
    spear_m,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_h,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_osm,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
    spear_osm_10ye,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
    lon2,lat2 = np.meshgrid(lons,lats)
    
    spear_mt,lats,lons = read_primary_dataset('T2M','SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_ht,lats,lons = read_primary_dataset('T2M','SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    spear_osmt,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
    spear_osm_10yet,lats,lons = read_primary_dataset('T2M','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
    
    yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
    climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)
    climoh_spear_zonal = np.nanmean(climoh_spear[:,:],axis=1)
    climoh_speart = np.nanmean(np.nanmean(spear_ht[:,yearq,:,:],axis=1),axis=0)
    climoh_spear_zonalt = np.nanmean(climoh_speart[:,:],axis=1)
    
    spear_ah = (spear_h - climoh_spear[np.newaxis,np.newaxis,:,:])/climoh_spear[np.newaxis,np.newaxis,:,:]
    spear_am = (spear_m - climoh_spear[np.newaxis,np.newaxis,:,:])/climoh_spear[np.newaxis,np.newaxis,:,:]
    spear_aosm = (spear_osm - climoh_spear[np.newaxis,np.newaxis,:,:])/climoh_spear[np.newaxis,np.newaxis,:,:]
    spear_aosm_10ye = (spear_osm_10ye - climoh_spear[np.newaxis,np.newaxis,:,:])/climoh_spear[np.newaxis,np.newaxis,:,:]
    
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
    
    ### Calculate zonal means (percent change)
    climatechange_GWL_zonal = np.nanmean(climatechange_GWL[:,:],axis=1) * 100.
    os_GWL_zonal = np.nanmean(os_GWL[:,:],axis=1) * 100.
    os_10ye_GWL_zonal = np.nanmean(os_10ye_GWL[:,:],axis=1) * 100.
    
    ### Calculate percent change of zonal mean for precipitation
    fig = plt.figure()
    ax = plt.subplot(111)
    
    adjust_spines(ax, ['left', 'bottom'])            
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4.,width=2,which='major',color='dimgrey')
    ax.tick_params(axis='x',labelsize=6,pad=1.5)
    ax.tick_params(axis='y',labelsize=6,pad=1.5)
    ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.8)
    
    plt.plot(lats,climatechange_GWL_zonal,linestyle='-',color='maroon',
              clip_on=False,zorder=1,label=r'\textbf{%s$^{\circ}$C [%s] for SSP5-8.5}' % (selectGWL,years[ssp_GWL]),linewidth=2)
    plt.plot(lats,os_GWL_zonal,linestyle='-',color='darkslategrey',
              clip_on=False,zorder=1,label=r'\textbf{%s$^{\circ}$C [%s] for SSP5-3.4OS}' % (selectGWL,years[os_second_GWL]),linewidth=2)
    plt.plot(lats,os_10ye_GWL_zonal,linestyle='--',color='teal',
              clip_on=False,zorder=1,label=r'\textbf{%s$^{\circ}$C [%s] for SSP5-3.4OS_10ye}' % (selectGWL,years[os_10ye_second_GWL]),
              dashes=(1,0.3),linewidth=2)
    
    leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
          bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
          handlelength=1,handletextpad=0.5)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    plt.xticks(np.arange(-90,91,10),np.arange(-90,91,10))
    plt.yticks(np.round(np.arange(-200,200,10),2),np.round(np.arange(-200,200,10),2))
    plt.xlim([-90,90])
    plt.ylim([-10,50])
    
    plt.ylabel(r'\textbf{Precipitation Change [$\%$; 1921-1950]}',
                fontsize=10,color='k')
    plt.xlabel(r'\textbf{Latitude [$^{\circ}$]}',color='k',fontsize=10)
    
    plt.tight_layout()
    plt.savefig(directoryfigure + 'GWL-%s-PercChange_ZONAL_%s.png' % (selectGWLn,variq),dpi=300)
