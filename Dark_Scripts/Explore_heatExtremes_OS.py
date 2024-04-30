"""
Explore heat extremes in the OS runs
 
Author    : Zachary M. Labe
Date      : 21 July 2023
"""

import matplotlib.pyplot as plt
import calc_Utilities as UT
import xarray as xr
import numpy as np
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean
import os, glob
import calc_Stats as dSS
import sys
from netCDF4 import Dataset

import time
startTime = time.time()

### Parameters
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})
directorydata = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
vari = 'TMIN'
variq = 't_ref_min'
resolution = 'MEDS' 
minb = 1981
maxb = 2010
junedays = np.arange(0,30,1)
julydays = np.arange(0,31,1)
augustdays = np.arange(0,31,1)
dayslength = len(junedays) + len(julydays) + len(augustdays)
reg_name = 'Globe'
lat_bounds,lon_bounds = UT.regions(reg_name)
   
def readData(model,reg_name):
    directorydata = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
    
    ### Select model
    if model == 'SPEAR_MED':
        years = np.arange(1921,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS':
        years = np.arange(2011,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP245':
        years = np.arange(2011,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS_10ye':
        years = np.arange(2031,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv':
        years = np.arange(2041,2100+1)
        ENS = 9
    elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv':
        years = np.arange(2041,2100+1)
        ENS = 9
    elif model == 'SPEAR_MED_LM42p2_test':
        years = np.arange(1921,2070+1)
        ENS = 3
    
    if reg_name == 'US':
        daysall = np.empty((ENS,years.shape[0],dayslength,70,144))
    elif reg_name == 'Globe':
        daysall = np.empty((ENS,years.shape[0],dayslength,360,576))
    for e in range(ENS):
        data = Dataset(directorydata + 'June/June_%s_%s_%s_ens%s.nc' % (vari,reg_name,model,str(e+1).zfill(2)))
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
        june = data.variables['%s' % variq][:].reshape(len(years),len(junedays),lat.shape[0],lon.shape[0])
        data.close()
        
        data2 = Dataset(directorydata + 'July/July_%s_%s_%s_ens%s.nc' % (vari,reg_name,model,str(e+1).zfill(2)))
        july = data2.variables['%s' % variq][:].reshape(len(years),len(julydays),lat.shape[0],lon.shape[0])
        data2.close()

        data3 = Dataset(directorydata + 'August/August_%s_%s_%s_ens%s.nc' % (vari,reg_name,model,str(e+1).zfill(2)))
        august = data3.variables['%s' % variq][:].reshape(len(years),len(augustdays),lat.shape[0],lon.shape[0])
        data3.close()        

        ### Convert units if needed
        if any([vari == 'T2M',vari == 'TMAX']):
            daysall[e,:,:,:,:] = np.concatenate([june,july,august],axis=1) - 273.15 # K to C
            print('Completed: Changed units (K to C)!')
        else:
            daysall[e,:,:,:,:] = np.concatenate([june,july,august],axis=1)
            
    ### Meshgrid and mask by CONUS
    lon2,lat2 = np.meshgrid(lon,lat)
    
    if reg_name == 'US':
        data_obsnan = np.full([1,lat.shape[0],lon.shape[0]],np.nan)
        datamask,data_obsnan = dSS.mask_CONUS(daysall,data_obsnan,resolution,lat_bounds,lon_bounds)
    else:
        datamask = daysall
        
    return datamask,lat,lon,years

def calc_heatExtremes(datamask,model,lat,lon,baselineanom):
    
    ### Select model
    if model == 'SPEAR_MED':
        years = np.arange(1921,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS':
        years = np.arange(2011,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP245':
        years = np.arange(2011,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS_10ye':
        years = np.arange(2031,2100+1)
        ENS = 30
    elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv':
        years = np.arange(2041,2100+1)
        ENS = 9
    elif model == 'SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv':
        years = np.arange(2041,2100+1)
        ENS = 9
    elif model == 'SPEAR_MED_LM42p2_test':
        years = np.arange(1921,2070+1)
        ENS = 3
    
    if any([model == 'SPEAR_MED',model == 'SPEAR_MED_LM42p2_test']):
        ### Calculate baseline
        baseline = np.arange(minb,maxb+1,1)
        baseq = np.where((years >= minb) & (years <= maxb))[0]
        
        clim = datamask[:,baseq,:,:,:]
        climdist = clim.reshape(ENS,len(baseline)*dayslength,lat.shape[0],lon.shape[0])
    else:
        climdist = baselineanom
    
    ### Calculate heat extremes
    tx90 = np.nanpercentile(climdist,90,axis=1)
    tx95 = np.nanpercentile(climdist,95,axis=1)
    tx99 = np.nanpercentile(climdist,99,axis=1)
    
    ### Count of heat extremes for 90th percentile
    count90v = np.empty((ENS,len(years),lat.shape[0],lon.shape[0]))
    for ens in range(ENS):
        for i in range(lat.shape[0]):
            for j in range(lon.shape[0]):
                for yr in range(years.shape[0]):
                    summerens = datamask[ens,yr,:,i,j]
                    
                    if np.isfinite(np.nanmax(summerens)):
                        count90v[ens,yr,i,j] = (summerens > tx90[ens,i,j]).sum() 
                    else:
                        count90v[ens,yr,i,j] = np.nan
                        
    ### Count of heat extremes for 95th percentile
    count95v = np.empty((ENS,len(years),lat.shape[0],lon.shape[0]))
    for ens in range(ENS):
        for i in range(lat.shape[0]):
            for j in range(lon.shape[0]):
                for yr in range(years.shape[0]):
                    summerens = datamask[ens,yr,:,i,j]
                    
                    if np.isfinite(np.nanmax(summerens)):
                        count95v[ens,yr,i,j] = (summerens > tx95[ens,i,j]).sum() 
                    else:
                        count95v[ens,yr,i,j] = np.nan
                        
    ### Count of heat extremes for 99th percentile
    count99v = np.empty((ENS,len(years),lat.shape[0],lon.shape[0]))
    for ens in range(ENS):
        for i in range(lat.shape[0]):
            for j in range(lon.shape[0]):
                for yr in range(years.shape[0]):
                    summerens = datamask[ens,yr,:,i,j]

                    if np.isfinite(np.nanmax(summerens)):
                        count99v[ens,yr,i,j] = (summerens > tx99[ens,i,j]).sum() 
                    else:
                        count99v[ens,yr,i,j] = np.nan
    
    ### Frequency of heat extremes
    freq90v = count90v/dayslength
    freq95v = count95v/dayslength
    freq99v = count99v/dayslength
    
    return climdist,count90v,count95v,count99v,freq90v,freq95v,freq99v

##############################################################################
##############################################################################
##############################################################################
def netcdfHEAT(lats,lons,count90v,count95v,count99v,freq90v,freq95v,freq99v,directory,model,reg_name,vari):
    print('\n>>> Using netcdfHEAT function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + vari + '_' + model + '.nc'
    filename = directory + name
    ncfile = Dataset(filename,'w',format='NETCDF4')
    ncfile.description = '90th, 95th, and 99th percentiles for JJA heat' 
    
    ### Dimensions
    ncfile.createDimension('ensembles',count90v.shape[0])
    ncfile.createDimension('years',count90v.shape[1])
    ncfile.createDimension('lat',count90v.shape[2])
    ncfile.createDimension('lon',count90v.shape[3])
    
    ### Variables
    ensembles = ncfile.createVariable('ensembles','f4',('ensembles'))
    years = ncfile.createVariable('years','f4',('years'))
    latitude = ncfile.createVariable('lat','f4',('lat'))
    longitude = ncfile.createVariable('lon','f4',('lon'))
    count90n = ncfile.createVariable('count90','f4',('ensembles','years','lat','lon'))
    count95n = ncfile.createVariable('count95','f4',('ensembles','years','lat','lon'))
    count99n = ncfile.createVariable('count99','f4',('ensembles','years','lat','lon'))
    freq90n = ncfile.createVariable('freq90','f4',('ensembles','years','lat','lon'))
    freq95n = ncfile.createVariable('freq95','f4',('ensembles','years','lat','lon'))
    freq99n = ncfile.createVariable('freq99','f4',('ensembles','years','lat','lon'))
    
    ### Units
    count90n.units = 'count'
    count95n.units = 'count'
    freq90n.units = 'frequency'
    freq95n.units = 'frequency'
    ncfile.title = 'heat statistics'
    ncfile.instituion = 'NOAA GFDL SPEAR_MED'
    ncfile.references = 'Delworth et al. 2020'
    
    ### Data
    ensembles[:] = np.arange(count90v.shape[0])
    years[:] = np.arange(count90v.shape[1])
    latitude[:] = lats
    longitude[:] = lons
    count90n[:] = count90v
    count95n[:] = count95v
    count99n[:] = count99v
    freq90n[:] = freq90v
    freq95n[:] = freq95v
    freq99n[:] = freq99v
    
    ncfile.close()
    print('*Completed: Created netCDF4 File!')

summer_LM42p2_test,lat,lon,years_LM42p2_test = readData('SPEAR_MED_LM42p2_test',reg_name)
# summer_osSSP245,lat,lon,years_osSSP245 = readData('SPEAR_MED_SSP245',reg_name)
# summer_osAMOC2,lat,lon,years_osAMOC2 = readData('SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv',reg_name)
# summer_osAMOC,lat,lon,years_osAMOC = readData('SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv',reg_name)
summer_os10ye,lat,lon,years_os10ye = readData('SPEAR_MED_SSP534OS_10ye',reg_name)
summer_os,lat,lon,years_os = readData('SPEAR_MED_SSP534OS',reg_name)
summer,lat,lon,years = readData('SPEAR_MED',reg_name)

climdist,count90sp,count95sp,count99sp,freq90sp,freq95sp,freq99sp = calc_heatExtremes(summer,'SPEAR_MED',lat,lon,np.nan)
climdist_os10ye,count90_os10ye,count95_os10ye,count99_os10ye,freq90_os10ye,freq95_os10ye,freq99_os10ye = calc_heatExtremes(summer_os10ye,'SPEAR_MED_SSP534OS_10ye',lat,lon,climdist)
climdist_os,count90_os,count95_os,count99_os,freq90_os,freq95_os,freq99_os = calc_heatExtremes(summer_os,'SPEAR_MED_SSP534OS',lat,lon,climdist)
# climdist_osAMOC,count90_osAMOC,count95_osAMOC,count99_osAMOC,freq90_osAMOC,freq95_osAMOC,freq99_osAMOC = calc_heatExtremes(summer_osAMOC,'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv',lat,lon,climdist)
# climdist_osAMOC2,count90_osAMOC2,count95_osAMOC2,count99_osAMOC2,freq90_osAMOC2,freq95_osAMOC2,freq99_osAMOC2 = calc_heatExtremes(summer_osAMOC2,'SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv',lat,lon,climdist)
# climdist_osSSP245,count90_osSSP245,count95_osSSP245,count99_osSSP245,freq90_osSSP245,freq95_osSSP245,freq99_osSSP245 = calc_heatExtremes(summer_osSSP245,'SPEAR_MED_SSP245',lat,lon,climdist)
climdist_LM42p2_test,count90_LM42p2_test,count95_LM42p2_test,count99_LM42p2_test,freq90_LM42p2_test,freq95_LM42p2_test,freq99_LM42p2_test = calc_heatExtremes(summer_LM42p2_test,'SPEAR_MED_LM42p2_test',lat,lon,np.nan)

### Save data
netcdfHEAT(lat,lon,count90sp,count95sp,count99sp,freq90sp,freq95sp,freq99sp,directorydata,'SPEAR_MED',reg_name,vari)
netcdfHEAT(lat,lon,count90_os,count95_os,count95_os,freq90_os,freq95_os,freq99_os,directorydata,'SPEAR_MED_SSP534OS',reg_name,vari)
netcdfHEAT(lat,lon,count90_os10ye,count95_os10ye,count99_os10ye,freq90_os10ye,freq95_os10ye,freq99_os10ye,directorydata,'SPEAR_MED_SSP534OS_10ye',reg_name,vari)
# netcdfHEAT(lat,lon,count90_osAMOC,count95_osAMOC,count99_osAMOC,freq90_osAMOC,freq95_osAMOC,freq99_osAMOC,directorydata,'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv',reg_name,vari)
# netcdfHEAT(lat,lon,count90_osAMOC2,count95_osAMOC2,count99_osAMOC2,freq90_osAMOC2,freq95_osAMOC2,freq99_osAMOC2,directorydata,'SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv',reg_name,vari)
# netcdfHEAT(lat,lon,count90_osSSP245,count95_osSSP245,count99_osSSP245,freq90_osSSP245,freq95_osSSP245,freq99_osSSP245,directorydata,'SPEAR_MED_SSP245',reg_name,vari)
netcdfHEAT(lat,lon,count90_LM42p2_test,count95_LM42p2_test,count99_LM42p2_test,freq90_LM42p2_test,freq95_LM42p2_test,freq99_LM42p2_test,directorydata,'SPEAR_MED_LM42p2_test',reg_name,vari)
