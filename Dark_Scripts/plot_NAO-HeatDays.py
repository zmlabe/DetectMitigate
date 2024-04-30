"""
Calculate heat extremes regressed on the NAO index

Author    : Zachary M. Labe
Date      :9 October 2023
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
from scipy.signal import savgol_filter

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['SLP']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
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
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye']
dataset_obs = 'ERA5_MEDS'
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'Globe'
detrend = True

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
def regressData(x,y,runnamesm):    
    print('\n>>> Using regressData function! \n')
    
    if y.ndim == 5: # 5D array
        slope = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        intercept = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        rvalue = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        pvalue = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        stderr = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        for model in range(y.shape[0]):
            print('Completed: Regression for %s!' % runnamesm[model])
            for ens in range(y.shape[1]):
                for i in range(y.shape[3]):
                    for j in range(y.shape[4]):
                        ### 1D time series for regression
                        xx = x
                        yy = y[model,ens,:,i,j]
                        
                        ### Mask data for nans
                        mask = ~np.isnan(xx) & ~np.isnan(yy)
                        varx = xx[mask]
                        vary = yy[mask]
                        
                        ### Calculate regressions
                        slope[model,ens,i,j],intercept[model,ens,i,j], \
                        rvalue[model,ens,i,j],pvalue[model,ens,i,j], \
                        stderr[model,ens,i,j] = sts.linregress(varx,vary)
                        
    if y.ndim == 4: # 4D array
        slope = np.empty((y.shape[0],y.shape[2],y.shape[3]))
        intercept = np.empty((y.shape[0],y.shape[2],y.shape[3],))
        rvalue = np.empty((y.shape[0],y.shape[2],y.shape[3]))
        pvalue = np.empty((y.shape[0],y.shape[2],y.shape[3],))
        stderr = np.empty((y.shape[0],y.shape[2],y.shape[3]))
        for model in range(y.shape[0]):
            print('Completed: Regression for %s!' % runnamesm[model])
            for i in range(y.shape[2]):
                for j in range(y.shape[3]):
                    ### 1D time series for regression
                    xx = x
                    yy = y[model,:,i,j]
                    
                    ### Mask data for nans
                    mask = ~np.isnan(xx) & ~np.isnan(yy)
                    varx = xx[mask]
                    vary = yy[mask]
                        
                    ### Calculate regressions
                    slope[model,i,j],intercept[model,i,j], \
                    rvalue[model,i,j],pvalue[model,i,j], \
                    stderr[model,i,j] = sts.linregress(varx,vary)
                    
    elif y.ndim == 3: #3D array
        slope = np.empty((y.shape[1],y.shape[2]))
        intercept = np.empty((y.shape[1],y.shape[2]))
        rvalue = np.empty((y.shape[1],y.shape[2]))
        pvalue = np.empty((y.shape[1],y.shape[2]))
        stderr = np.empty((y.shape[1],y.shape[2]))
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                ### 1D time series for regression
                xx = x
                yy = y[:,i,j]
                
                ### Mask data for nans
                mask = ~np.isnan(xx) & ~np.isnan(yy)
                varx = xx[mask]
                vary = yy[mask]
                        
                ### Calculate regressions
                slope[i,j],intercept[i,j],rvalue[i,j], \
                pvalue[i,j],stderr[i,j] = sts.linregress(varx,vary)
                        
    print('>>> Completed: Finished regressData function!')
    return slope,intercept,rvalue,pvalue,stderr
###############################################################################
###############################################################################
##############################################################################
### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_mr,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
spear_osmr,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10yer,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Remove ensemble mean
if detrend == True:
    spear_m_mean = np.nanmean(spear_mr[:,:,:,:],axis=0)
    spear_osm_mean = np.nanmean(spear_osmr[:,:,:,:],axis=0)
    spear_osm10ye_mean = np.nanmean(spear_osm_10yer[:,:,:,:],axis=0)
    
    spear_m = spear_mr - spear_m_mean
    spear_osm = spear_osmr - spear_osm_mean
    spear_osm_10ye = spear_osm_10yer - spear_osm10ye_mean
else:
    spear_m = spear_mr
    spear_osm = spear_osmr
    spear_osm_10ye = spear_osm_10yer

### Calculate anomalies
yearq = np.where((years >= 2015) & (years <= 2044))[0]
climo_spear = np.nanmean(spear_m[:,yearq,:,:],axis=1)
climo_osspear = np.nanmean(spear_osm[:,yearq,:,:],axis=1)
climo_os10yespear = np.nanmean(spear_osm_10ye[:,yearq,:,:],axis=1)

spear_am = spear_m - climo_spear[:,np.newaxis,:,:]
spear_aosm = spear_osm - climo_osspear[:,np.newaxis,:,:]
spear_aosm_10ye = spear_osm_10ye - climo_os10yespear[:,np.newaxis,:,:]

###############################################################################
### Calculate region 1
lonq1_z1 = np.where((lons >=0) & (lons <=60))[0]
lonq2_z1 = np.where((lons >= 270) & (lons <= 360))[0]
lonq_z1 = np.append(lonq1_z1,lonq2_z1)
latq_z1 = np.where((lats >=20) & (lats <=55))[0]
lon2_z1,lat2_z1 = np.meshgrid(lons[lonq_z1],lats[latq_z1])

spear_am_z11 = spear_am[:,:,latq_z1,:]
spear_am_z1 = spear_am_z11[:,:,:,lonq_z1]

spear_aosm_z11 = spear_aosm[:,:,latq_z1,:]
spear_aosm_z1 = spear_aosm_z11[:,:,:,lonq_z1]

spear_aosm_10ye_z11 = spear_aosm_10ye[:,:,latq_z1,:]
spear_aosm_10ye_z1 = spear_aosm_10ye_z11[:,:,:,lonq_z1]

### Calculate index for z1
ave_spear_z1 = UT.calc_weightedAve(spear_am_z1,lat2_z1)
ave_spear_os_z1 = UT.calc_weightedAve(spear_aosm_z1,lat2_z1)
ave_spear_os10ye_z1 = UT.calc_weightedAve(spear_aosm_10ye_z1,lat2_z1)

###############################################################################
### Calculate region 2
lonq1_z2 = np.where((lons >=0) & (lons <=60))[0]
lonq2_z2 = np.where((lons >= 270) & (lons <= 360))[0]
lonq_z2 = np.append(lonq1_z2,lonq2_z2)
latq_z2 = np.where((lats >=55) & (lats <=90))[0]
lon2_z2,lat2_z2 = np.meshgrid(lons[lonq_z2],lats[latq_z2])

spear_am_z22 = spear_am[:,:,latq_z2,:]
spear_am_z2 = spear_am_z22[:,:,:,lonq_z2]

spear_aosm_z22 = spear_aosm[:,:,latq_z2,:]
spear_aosm_z2 = spear_aosm_z22[:,:,:,lonq_z2]

spear_aosm_10ye_z22 = spear_aosm_10ye[:,:,latq_z2,:]
spear_aosm_10ye_z2 = spear_aosm_10ye_z22[:,:,:,lonq_z2]

### Calculate index for z2
ave_spear_z2 = UT.calc_weightedAve(spear_am_z2,lat2_z2)
ave_spear_os_z2 = UT.calc_weightedAve(spear_aosm_z2,lat2_z2)
ave_spear_os10ye_z2 = UT.calc_weightedAve(spear_aosm_10ye_z2,lat2_z2)

ave_spear = ave_spear_z1 - ave_spear_z2
ave_spear_os = ave_spear_os_z1 - ave_spear_os_z2
ave_spear_os10ye = ave_spear_os10ye_z1 - ave_spear_os10ye_z2

### Save output
directorydata = '/work/Zachary.Labe/Data/ClimateIndices/NAO/'
if detrend == True:
    np.savez(directorydata + 'NAO_SLP_SPEAR-MED_%s_2015-2100_detrend.npz' % monthlychoice,
                  spear_unfilter=ave_spear,years=years)
    np.savez(directorydata + 'NAO_SLP_SPEAR-MED-SSP534OS_%s_2015-2100_detrend.npz' % monthlychoice,
                  spear_os_unfilter=ave_spear_os,years=years)
    np.savez(directorydata + 'NAO_SLP_SPEAR-MED-SSP534OS-10ye_%s_2015-2100_detrend.npz' % monthlychoice,
                  spear_os10ye_unfilter=ave_spear_os10ye,years=years)
elif detrend == False:
    np.savez(directorydata + 'NAO_SLP_SPEAR-MED_%s_2015-2100.npz' % monthlychoice,
                  spear_unfilter=ave_spear,years=years)
    np.savez(directorydata + 'NAO_SLP_SPEAR-MED-SSP534OS_%s_2015-2100.npz' % monthlychoice,
                  spear_os_unfilter=ave_spear_os,years=years)
    np.savez(directorydata + 'NAO_SLP_SPEAR-MED-SSP534OS-10ye_%s_2015-2100.npz' % monthlychoice,
                  spear_os10ye_unfilter=ave_spear_os10ye,years=years)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in heat days
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
varcount = 'count90'
variq = 'TMAX'

name = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED' + '.nc'
filename = directorydatah + name
data = Dataset(filename)
latus = data.variables['lat'][:]
lonus = data.variables['lon'][:]
count90 = data.variables[varcount][:,-86:,:,:]
data.close()

### Read in SPEAR_MED_SSP534OS
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
filename_os = directorydatah + name_os
data_os = Dataset(filename_os)
count90_os = data_os.variables[varcount][:,4:,:,:]
data_os.close()

### Read in SPEAR_MED_SSP534OS_10ye
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os10ye = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
filename_os10ye = directorydatah + name_os10ye
data_os10ye = Dataset(filename_os10ye)
count90_os10ye_1 = data_os10ye.variables[varcount][:]
data_os10ye.close()

count90_os10ye = np.append(count90_os[:,:count90_os.shape[1]-count90_os10ye_1.shape[1],:,:],count90_os10ye_1,axis=1)

### Remove ensemble mean
if detrend == True:
    count90_mean = np.nanmean(count90[:,:,:,:],axis=0)
    count90_os_mean = np.nanmean(count90_os[:,:,:,:],axis=0)
    count90_os10ye_mean = np.nanmean(count90_os10ye[:,:,:,:],axis=0)
    
    count90_m = count90 - count90_mean
    count90_osm = count90_os - count90_os_mean
    count90_osm_10ye = count90_os10ye - count90_os10ye_mean
else:
    count90_m = count90
    count90_osm = count90_os
    count90_osm_10ye = count90_os10ye

### Calculate anomalies
yearq = np.where((years >= 2015) & (years <= 2044))[0]
climo_spearho = np.nanmean(count90_m[:,yearq,:,:],axis=1)
climo_osspearho = np.nanmean(count90_osm[:,yearq,:,:],axis=1)
climo_os10yespearho = np.nanmean(count90_osm_10ye[:,yearq,:,:],axis=1)

spear_amho = count90_m - climo_spearho[:,np.newaxis,:,:]
spear_aosmho = count90_osm - climo_osspearho[:,np.newaxis,:,:]
spear_aosm_10yeho = count90_osm_10ye - climo_os10yespearho[:,np.newaxis,:,:]

###############################################################################
###############################################################################
###############################################################################
### Regress heatwave day anomalies onto NAO index
ensRegr = np.empty((len(spear_amho),lats.shape[0],lons.shape[0]))
ensPVal = np.empty((len(spear_amho),lats.shape[0],lons.shape[0]))
ensRVal = np.empty((len(spear_amho),lats.shape[0],lons.shape[0]))
for ee in range(len(spear_amho)):
    slope,intercept,rvalue,pvalue,stderr = regressData(ave_spear[ee,:],spear_amho[ee,:,:,:],'SPEAR_MED')
    ensRegr[ee,:,:] = slope
    ensPVal[ee,:,:] = pvalue
    ensRVal[ee,:,:] = rvalue
    print('>>> Finished calculating ensemble-%s for regression of %s!' % (ee+1,'SPEAR_MED'))

### Calculate ensemble means
slopeM = np.nanmean(ensRegr,axis=0)
pvalueM = np.nanmean(ensPVal,axis=0)

### Significant at 95% confidence level
pvalueM[np.where(pvalueM >= 0.05)] = np.nan
pvalueM[np.where(pvalueM < 0.05)] = 1.

###############################################################################
ensRegr_os = np.empty((len(spear_aosmho),lats.shape[0],lons.shape[0]))
ensPVal_os = np.empty((len(spear_aosmho),lats.shape[0],lons.shape[0]))
ensRVal_os = np.empty((len(spear_aosmho),lats.shape[0],lons.shape[0]))
for ee in range(len(spear_aosmho)):
    slope_os,intercept_os,rvalue_os,pvalue_os,stderr_os = regressData(ave_spear_os[ee,:],spear_aosmho[ee,:,:,:],'SPEAR_MED_SSP534OS')
    ensRegr_os[ee,:,:] = slope_os
    ensPVal_os[ee,:,:] = pvalue_os
    ensRVal_os[ee,:,:] = rvalue_os
    print('>>> Finished calculating ensemble-%s for regression of %s!' % (ee+1,'SPEAR_MED_SSP534OS'))

### Calculate ensemble means
slopeM_os = np.nanmean(ensRegr_os,axis=0)
pvalueM_os = np.nanmean(ensPVal_os,axis=0)

### Significant at 95% confidence level
pvalueM_os[np.where(pvalueM_os >= 0.05)] = np.nan
pvalueM_os[np.where(pvalueM_os < 0.05)] = 1.

###############################################################################
ensRegr_os10ye = np.empty((len(spear_aosm_10yeho),lats.shape[0],lons.shape[0]))
ensPVal_os10ye = np.empty((len(spear_aosm_10yeho),lats.shape[0],lons.shape[0]))
ensRVal_os10ye = np.empty((len(spear_aosm_10yeho),lats.shape[0],lons.shape[0]))
for ee in range(len(spear_aosm_10yeho)):
    slope_os10ye,intercept_os10ye,rvalue_os10ye,pvalue_os10ye,stderr_os10ye = regressData(ave_spear_os10ye[ee,:],spear_aosm_10yeho[ee,:,:,:],'SPEAR_MED_SSP534OS_10ye')
    ensRegr_os10ye[ee,:,:] = slope_os10ye
    ensPVal_os10ye[ee,:,:] = pvalue_os10ye
    ensRVal_os10ye[ee,:,:] = rvalue_os10ye
    print('>>> Finished calculating ensemble-%s for regression of %s!' % (ee+1,'SPEAR_MED_SSP534OS_10ye'))

### Calculate ensemble means
slopeM_os10ye = np.nanmean(ensRegr_os10ye,axis=0)
pvalueM_os10ye = np.nanmean(ensPVal_os10ye,axis=0)

### Significant at 95% confidence level
pvalueM_os10ye[np.where(pvalueM_os10ye >= 0.05)] = np.nan
pvalueM_os10ye[np.where(pvalueM_os10ye < 0.05)] = 1.

###############################################################################
###############################################################################
###############################################################################
### Plot regression Pattern
limit = np.arange(-4,4.01,0.1)
barlim = np.round(np.arange(-4,5,2),2)

fig = plt.figure(figsize=(10,3))
label = r'\textbf{Regression Coefficients [NAO; days]}'

ax = plt.subplot(131)

var = slopeM
pvar = pvalueM

m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
  
lon2,lat2 = np.meshgrid(lons,lats)

circle = m.drawmapboundary(fill_color='white',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
cs2 = m.contourf(lon2,lat2,pvar,colors='None',hatches=['....'],
             linewidths=0.4,latlon=True)

cs1.set_cmap(cmocean.cm.balance)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
m.drawcoastlines(color='darkgrey',linewidth=0.3,zorder=12)

plt.title(r'\textbf{SPEAR_MED}',fontsize=8,color='k')
ax.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.86,0.97),
              textcoords='axes fraction',color='k',fontsize=6,
              rotation=330,ha='center',va='center')
    

###############################################################################
ax = plt.subplot(132)

var = slopeM_os
pvar = pvalueM_os

m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
  
lon2,lat2 = np.meshgrid(lons,lats)

circle = m.drawmapboundary(fill_color='white',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
cs2 = m.contourf(lon2,lat2,pvar,colors='None',hatches=['....'],
             linewidths=0.4,latlon=True)

cs1.set_cmap(cmocean.cm.balance)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
m.drawcoastlines(color='darkgrey',linewidth=0.3,zorder=12)

plt.title(r'\textbf{SPEAR_MED_SSP534OS}',fontsize=8,color='k')
ax.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.86,0.97),
              textcoords='axes fraction',color='k',fontsize=6,
              rotation=330,ha='center',va='center')
    

###############################################################################
ax = plt.subplot(133)

var = slopeM_os10ye
pvar = pvalueM_os10ye

m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
  
lon2,lat2 = np.meshgrid(lons,lats)

circle = m.drawmapboundary(fill_color='white',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
cs2 = m.contourf(lon2,lat2,pvar,colors='None',hatches=['....'],
             linewidths=0.4,latlon=True)

cs1.set_cmap(cmocean.cm.balance)
m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=False,zorder=11)
m.drawcoastlines(color='darkgrey',linewidth=0.3,zorder=12)

plt.title(r'\textbf{SPEAR_MED_SSP534OS_10ye}',fontsize=8,color='k')
ax.annotate(r'\textbf{[%s]}' % letters[2],xy=(0,0),xytext=(0.86,0.97),
              textcoords='axes fraction',color='k',fontsize=6,
              rotation=330,ha='center',va='center')
    
cbar_ax1 = fig.add_axes([0.402,0.14,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=7,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')
plt.tight_layout()
    
plt.savefig(directoryfigure + 'Regression_NAO-HeatDays_TMAX_%s_2015-2100.png' % (monthlychoice),dpi=300)

