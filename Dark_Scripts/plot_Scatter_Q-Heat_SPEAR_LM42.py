"""
Plot relationship between daily heat extremes and q

Author    : Zachary M. Labe
Date      : 31 August 2023
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
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
months = 'JJA'
lenmon = 3
variq = variablesall[0]
numOfEns = 30
numOfEns_os = 30
numOfEns_LM42 = 3
numOfEns_os10ye = 9
years = np.arange(1921,2100+1)
years_LM42 = np.arange(1921,2070+1)
years_os = np.arange(2011,2100+1)
years_os10ye = np.arange(2031,2100+1)

junedays = np.arange(0,30,1)
julydays = np.arange(0,31,1)
augustdays = np.arange(0,31,1)
dayslength = len(junedays) + len(julydays) + len(augustdays)

reg_name = 'Globe'
lat_bounds,lon_bounds = UT.regions(reg_name)

directoryjune = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/June/'
directoryjuly = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/July/'
directoryaugust = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/August/'

###############################################################################
###############################################################################
###############################################################################
### Read in SPEAR_MED_LM42
TMIN_lm42 = np.full((numOfEns_LM42,years_LM42.shape[0],dayslength,70,144),np.nan)
q_lm42 = np.full((numOfEns_LM42,years_LM42.shape[0],dayslength,70,144),np.nan)
for s in range(numOfEns_LM42):
    data = Dataset(directoryjune + 'June_TMIN_US_SPEAR_MED_LM42p2_test_ens%s.nc' % str(s+1).zfill(2))
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    TMIN_june = data.variables['t_ref_min'][:].reshape(len(years_LM42),len(junedays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryjuly + 'July_TMIN_US_SPEAR_MED_LM42p2_test_ens%s.nc' % str(s+1).zfill(2))
    TMIN_july = data.variables['t_ref_min'][:].reshape(len(years_LM42),len(julydays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryaugust + 'August_TMIN_US_SPEAR_MED_LM42p2_test_ens%s.nc' % str(s+1).zfill(2))
    TMIN_august = data.variables['t_ref_min'][:].reshape(len(years_LM42),len(augustdays),lat.shape[0],lon.shape[0])
    data.close()
    
    ###########################################################################
    data = Dataset(directoryjune + 'June_q_US_SPEAR_MED_LM42p2_test_ens%s.nc' % str(s+1).zfill(2))
    q_june = data.variables['q_ref'][:].reshape(len(years_LM42),len(junedays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryjuly + 'July_q_US_SPEAR_MED_LM42p2_test_ens%s.nc' % str(s+1).zfill(2))
    q_july = data.variables['q_ref'][:].reshape(len(years_LM42),len(julydays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryaugust + 'August_q_US_SPEAR_MED_LM42p2_test_ens%s.nc' % str(s+1).zfill(2))
    q_august = data.variables['q_ref'][:].reshape(len(years_LM42),len(augustdays),lat.shape[0],lon.shape[0])
    data.close()
    
    ###########################################################################
    TMIN_lm42[s,:,:,:,:] = np.concatenate([TMIN_june,TMIN_july,TMIN_august],axis=1)
    q_lm42[s,:,:,:,:] = np.concatenate([q_june,q_july,q_august],axis=1)
    
###############################################################################
###############################################################################
###############################################################################    
### Read in SPEAR_MED
TMIN_spear = np.full((numOfEns,years.shape[0],dayslength,70,144),np.nan)
q_spear = np.full((numOfEns,years.shape[0],dayslength,70,144),np.nan)
for s in range(numOfEns_os):
    data = Dataset(directoryjune + 'June_TMIN_US_SPEAR_MED_ens%s.nc' % str(s+1).zfill(2))
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    TMIN_june = data.variables['t_ref_min'][:].reshape(len(years),len(junedays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryjuly + 'July_TMIN_US_SPEAR_MED_ens%s.nc' % str(s+1).zfill(2))
    TMIN_july = data.variables['t_ref_min'][:].reshape(len(years),len(julydays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryaugust + 'August_TMIN_US_SPEAR_MED_ens%s.nc' % str(s+1).zfill(2))
    TMIN_august = data.variables['t_ref_min'][:].reshape(len(years),len(augustdays),lat.shape[0],lon.shape[0])
    data.close()
    
    ###########################################################################
    data = Dataset(directoryjune + 'June_q_US_SPEAR_MED_ens%s.nc' % str(s+1).zfill(2))
    q_june = data.variables['q_ref'][:].reshape(len(years),len(junedays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryjuly + 'July_q_US_SPEAR_MED_ens%s.nc' % str(s+1).zfill(2))
    q_july = data.variables['q_ref'][:].reshape(len(years),len(julydays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryaugust + 'August_q_US_SPEAR_MED_ens%s.nc' % str(s+1).zfill(2))
    q_august = data.variables['q_ref'][:].reshape(len(years),len(augustdays),lat.shape[0],lon.shape[0])
    data.close()
    
    ###########################################################################
    TMIN_spear[s,:,:,:,:] = np.concatenate([TMIN_june,TMIN_july,TMIN_august],axis=1)
    q_spear[s,:,:,:,:] = np.concatenate([q_june,q_july,q_august],axis=1)

###############################################################################
###############################################################################
###############################################################################    
### Read in SPEAR_MED_SSP534OS
TMIN_os = np.full((numOfEns_os,years_os.shape[0],dayslength,70,144),np.nan)
q_os = np.full((numOfEns_os,years_os.shape[0],dayslength,70,144),np.nan)
for s in range(numOfEns_os):
    data = Dataset(directoryjune + 'June_TMIN_US_SPEAR_MED_SSP534OS_ens%s.nc' % str(s+1).zfill(2))
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    TMIN_june = data.variables['t_ref_min'][:].reshape(len(years_os),len(junedays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryjuly + 'July_TMIN_US_SPEAR_MED_SSP534OS_ens%s.nc' % str(s+1).zfill(2))
    TMIN_july = data.variables['t_ref_min'][:].reshape(len(years_os),len(julydays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryaugust + 'August_TMIN_US_SPEAR_MED_SSP534OS_ens%s.nc' % str(s+1).zfill(2))
    TMIN_august = data.variables['t_ref_min'][:].reshape(len(years_os),len(augustdays),lat.shape[0],lon.shape[0])
    data.close()
    
    ###########################################################################
    data = Dataset(directoryjune + 'June_q_US_SPEAR_MED_SSP534OS_ens%s.nc' % str(s+1).zfill(2))
    q_june = data.variables['q_ref'][:].reshape(len(years_os),len(junedays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryjuly + 'July_q_US_SPEAR_MED_SSP534OS_ens%s.nc' % str(s+1).zfill(2))
    q_july = data.variables['q_ref'][:].reshape(len(years_os),len(julydays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryaugust + 'August_q_US_SPEAR_MED_SSP534OS_ens%s.nc' % str(s+1).zfill(2))
    q_august = data.variables['q_ref'][:].reshape(len(years_os),len(augustdays),lat.shape[0],lon.shape[0])
    data.close()
    
    ###########################################################################
    TMIN_os[s,:,:,:,:] = np.concatenate([TMIN_june,TMIN_july,TMIN_august],axis=1)
    q_os[s,:,:,:,:] = np.concatenate([q_june,q_july,q_august],axis=1)
    
###############################################################################
###############################################################################
###############################################################################    
### Read in SPEAR_MED_SSP534OS_10ye
TMIN_os10ye = np.full((numOfEns_os10ye,years_os10ye.shape[0],dayslength,70,144),np.nan)
q_os10ye = np.full((numOfEns_os10ye,years_os10ye.shape[0],dayslength,70,144),np.nan)
for s in range(numOfEns_os10ye):
    data = Dataset(directoryjune + 'June_TMIN_US_SPEAR_MED_SSP534OS_10ye_ens%s.nc' % str(s+1).zfill(2))
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    TMIN_june = data.variables['t_ref_min'][:].reshape(len(years_os10ye),len(junedays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryjuly + 'July_TMIN_US_SPEAR_MED_SSP534OS_10ye_ens%s.nc' % str(s+1).zfill(2))
    TMIN_july = data.variables['t_ref_min'][:].reshape(len(years_os10ye),len(julydays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryaugust + 'August_TMIN_US_SPEAR_MED_SSP534OS_10ye_ens%s.nc' % str(s+1).zfill(2))
    TMIN_august = data.variables['t_ref_min'][:].reshape(len(years_os10ye),len(augustdays),lat.shape[0],lon.shape[0])
    data.close()
    
    ###########################################################################
    data = Dataset(directoryjune + 'June_q_US_SPEAR_MED_SSP534OS_10ye_ens%s.nc' % str(s+1).zfill(2))
    q_june = data.variables['q_ref'][:].reshape(len(years_os10ye),len(junedays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryjuly + 'July_q_US_SPEAR_MED_SSP534OS_10ye_ens%s.nc' % str(s+1).zfill(2))
    q_july = data.variables['q_ref'][:].reshape(len(years_os10ye),len(julydays),lat.shape[0],lon.shape[0])
    data.close()
    
    data = Dataset(directoryaugust + 'August_q_US_SPEAR_MED_SSP534OS_10ye_ens%s.nc' % str(s+1).zfill(2))
    q_august = data.variables['q_ref'][:].reshape(len(years_os10ye),len(augustdays),lat.shape[0],lon.shape[0])
    data.close()
    
    ###########################################################################
    TMIN_os10ye[s,:,:,:,:] = np.concatenate([TMIN_june,TMIN_july,TMIN_august],axis=1)
    q_os10ye[s,:,:,:,:] = np.concatenate([q_june,q_july,q_august],axis=1)

###############################################################################
###############################################################################
###############################################################################   
### Mask over the United States
lon2,lat2 = np.meshgrid(lon,lat)

if reg_name == 'US':
    data_obsnan = np.full([1,lat.shape[0],lon.shape[0]],np.nan)
    
    TMIN_lm42_mask,data_obsnan = dSS.mask_CONUS(TMIN_lm42,data_obsnan,'MEDS',lat_bounds,lon_bounds)
    q_lm42_mask,data_obsnan = dSS.mask_CONUS(q_lm42,data_obsnan,'MEDS',lat_bounds,lon_bounds)
    
    TMIN_spear_mask,data_obsnan = dSS.mask_CONUS(TMIN_spear,data_obsnan,'MEDS',lat_bounds,lon_bounds)
    q_spear_mask,data_obsnan = dSS.mask_CONUS(q_spear,data_obsnan,'MEDS',lat_bounds,lon_bounds)
    
    TMIN_os_mask,data_obsnan = dSS.mask_CONUS(TMIN_os,data_obsnan,'MEDS',lat_bounds,lon_bounds)
    q_os_mask,data_obsnan = dSS.mask_CONUS(q_os,data_obsnan,'MEDS',lat_bounds,lon_bounds)
    
    TMIN_os10ye_mask,data_obsnan = dSS.mask_CONUS(TMIN_os10ye,data_obsnan,'MEDS',lat_bounds,lon_bounds)
    q_os10ye_mask,data_obsnan = dSS.mask_CONUS(q_os10ye,data_obsnan,'MEDS',lat_bounds,lon_bounds)
else:
    TMIN_lm42_mask = TMIN_lm42
    q_lm42_mask = q_lm42
    TMIN_spear_mask = TMIN_spear
    q_spear_mask = q_spear
    TMIN_os_mask = TMIN_os
    q_os_mask = q_os
    TMIN_os10ye_mask = TMIN_os10ye
    q_os10ye_mask = q_os10ye

##############################################################################
##############################################################################
##############################################################################
### Calculate means for JJA
TMIN_meanJJA_lm42 = np.nanmean(TMIN_lm42_mask[:,:,:,:,:],axis=2)
TMIN_meanJJA_spear = np.nanmean(TMIN_spear_mask[:,:,:,:,:],axis=2)
TMIN_meanJJA_os = np.nanmean(TMIN_os_mask[:,:,:,:,:],axis=2)
TMIN_meanJJA_os10ye = np.nanmean(TMIN_os10ye_mask[:,:,:,:,:],axis=2)

q_meanJJA_lm42 = np.nanmean(q_lm42_mask[:,:,:,:,:],axis=2)
q_meanJJA_spear = np.nanmean(q_spear_mask[:,:,:,:,:],axis=2)
q_meanJJA_os = np.nanmean(q_os_mask[:,:,:,:,:],axis=2)
q_meanJJA_os10ye = np.nanmean(q_os10ye_mask[:,:,:,:,:],axis=2)

##############################################################################
##############################################################################
##############################################################################
### Evalute future extremes
yearsq = np.where((years >= 2015) & (years <= 2070))[0]
yearsq_lm42 = np.where((years_LM42 >= 2015) & (years_LM42 <= 2070))[0]
yearsq_spear = np.where((years >= 2015) & (years <= 2070))[0]
yearsq_os = np.where((years_os >= 2015) & (years_os <= 2070))[0]
yearsq_os10ye = np.where((years_os10ye >= 2015) & (years_os10ye <= 2070))[0]

TMIN_meanJJA_lm42_fut = TMIN_meanJJA_lm42[:,yearsq_lm42,:,:]
TMIN_meanJJA_spear_fut = TMIN_meanJJA_spear[:,yearsq_spear,:,:]
TMIN_meanJJA_os_fut = TMIN_meanJJA_os[:,yearsq_os,:,:]
TMIN_meanJJA_os10ye_fut = TMIN_meanJJA_os10ye[:,yearsq_os10ye,:,:]

q_meanJJA_lm42_fut = q_meanJJA_lm42[:,yearsq_lm42,:,:]
q_meanJJA_spear_fut = q_meanJJA_spear[:,yearsq_spear,:,:]
q_meanJJA_os_fut = q_meanJJA_os[:,yearsq_os,:,:]
q_meanJJA_os10ye_fut = q_meanJJA_os10ye[:,yearsq_os10ye,:,:]

##############################################################################
##############################################################################
##############################################################################
### Calculate ensemble means
TMIN_ensmean_lm42 = np.nanmean(TMIN_meanJJA_lm42_fut,axis=0)
TMIN_ensmean_spear = np.nanmean(TMIN_meanJJA_spear_fut,axis=0)
TMIN_ensmean_os = np.nanmean(TMIN_meanJJA_os_fut,axis=0)
TMIN_ensmean_os10ye = np.nanmean(TMIN_meanJJA_os10ye_fut,axis=0)

q_ensmean_lm42 = np.nanmean(q_meanJJA_lm42_fut,axis=0)
q_ensmean_spear = np.nanmean(q_meanJJA_spear_fut,axis=0)
q_ensmean_os = np.nanmean(q_meanJJA_os_fut,axis=0)
q_ensmean_os10ye = np.nanmean(q_meanJJA_os10ye_fut,axis=0)

##############################################################################
##############################################################################
##############################################################################
def netcdfmeanJJA(lats,lons,meanJJA,model,reg_name,vari):
    print('\n>>> Using netcdfHEAT function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    directory = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/MeanJJA/'
    name = 'JJAmean_' + reg_name + '_' + vari + '_' + model + '.nc'
    filename = directory + name
    ncfile = Dataset(filename,'w',format='NETCDF4')
    ncfile.description = 'mean of JJA variable' 
    
    ### Dimensions
    ncfile.createDimension('ensembles',meanJJA.shape[0])
    ncfile.createDimension('years',meanJJA.shape[1])
    ncfile.createDimension('lat',meanJJA.shape[2])
    ncfile.createDimension('lon',meanJJA.shape[3])
    
    ### Variables
    ensembles = ncfile.createVariable('ensembles','f4',('ensembles'))
    years = ncfile.createVariable('years','f4',('years'))
    latitude = ncfile.createVariable('lat','f4',('lat'))
    longitude = ncfile.createVariable('lon','f4',('lon'))
    meanJJAn = ncfile.createVariable('meanJJA','f4',('ensembles','years','lat','lon'))
    
    ### Units
    ncfile.title = 'mean JJA'
    ncfile.instituion = 'NOAA GFDL SPEAR_MED'
    ncfile.references = 'Delworth et al. 2020'
    
    ### Data
    ensembles[:] = np.arange(meanJJA.shape[0])
    years[:] = np.arange(meanJJA.shape[1])
    latitude[:] = lats
    longitude[:] = lons
    meanJJAn[:] = meanJJA
    
    ncfile.close()
    print('*Completed: Created netCDF4 File!')
     
### Save data
netcdfmeanJJA(lat,lon,TMIN_meanJJA_lm42,'SPEAR_MED_LM42p2_test',reg_name,'TMIN')
# netcdfmeanJJA(lat,lon,q_meanJJA_lm42,'SPEAR_MED_LM42p2_test',reg_name,'q')

netcdfmeanJJA(lat,lon,TMIN_meanJJA_spear,'SPEAR_MED',reg_name,'TMIN')
# netcdfmeanJJA(lat,lon,q_meanJJA_spear,'SPEAR_MED',reg_name,'q')

netcdfmeanJJA(lat,lon,TMIN_meanJJA_os,'SPEAR_MED_SSP534OS',reg_name,'TMIN')
# netcdfmeanJJA(lat,lon,q_meanJJA_os,'SPEAR_MED_SSP534OS',reg_name,'q')

netcdfmeanJJA(lat,lon,TMIN_meanJJA_os10ye,'SPEAR_MED_SSP534OS_10ye',reg_name,'TMIN')
# netcdfmeanJJA(lat,lon,q_meanJJA_os10ye,'SPEAR_MED_SSP534OS_10ye',reg_name,'q')
