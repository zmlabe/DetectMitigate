"""
Explore dry extremes in the OS runs
 
Author    : Zachary M. Labe
Date      : 31 August 2023
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
vari = 'q'
variq = 'q_ref'
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

def calc_DryExtremes(datamask,model,lat,lon,baselineanom):
    
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
    
    ### Calculate Dry extremes
    tx10 = np.nanpercentile(climdist,10,axis=1)
    tx05 = np.nanpercentile(climdist,5,axis=1)
    tx01 = np.nanpercentile(climdist,1,axis=1)
    
    ### Count of Dry extremes for 10th percentile
    count10v = np.empty((ENS,len(years),lat.shape[0],lon.shape[0]))
    for ens in range(ENS):
        for i in range(lat.shape[0]):
            for j in range(lon.shape[0]):
                for yr in range(years.shape[0]):
                    summerens = datamask[ens,yr,:,i,j]
                    
                    if np.isfinite(np.nanmax(summerens)):
                        count10v[ens,yr,i,j] = (summerens < tx10[ens,i,j]).sum() 
                    else:
                        count10v[ens,yr,i,j] = np.nan
                        
    ### Count of Dry extremes for 5th percentile
    count05v = np.empty((ENS,len(years),lat.shape[0],lon.shape[0]))
    for ens in range(ENS):
        for i in range(lat.shape[0]):
            for j in range(lon.shape[0]):
                for yr in range(years.shape[0]):
                    summerens = datamask[ens,yr,:,i,j]
                    
                    if np.isfinite(np.nanmax(summerens)):
                        count05v[ens,yr,i,j] = (summerens < tx05[ens,i,j]).sum() 
                    else:
                        count05v[ens,yr,i,j] = np.nan
                        
    ### Count of Dry extremes for 1st percentile
    count01v = np.empty((ENS,len(years),lat.shape[0],lon.shape[0]))
    for ens in range(ENS):
        for i in range(lat.shape[0]):
            for j in range(lon.shape[0]):
                for yr in range(years.shape[0]):
                    summerens = datamask[ens,yr,:,i,j]

                    if np.isfinite(np.nanmax(summerens)):
                        count01v[ens,yr,i,j] = (summerens < tx01[ens,i,j]).sum() 
                    else:
                        count01v[ens,yr,i,j] = np.nan
    
    ### Frequency of Dry extremes
    freq10v = count10v/dayslength
    freq05v = count05v/dayslength
    freq01v = count01v/dayslength
    
    return climdist,count10v,count05v,count01v,freq10v,freq05v,freq01v

##############################################################################
##############################################################################
##############################################################################
def netcdfDry(lats,lons,count10v,count05v,count01v,freq10v,freq05v,freq01v,directory,model,reg_name,vari):
    print('\n>>> Using netcdfDry function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'DryStats/DryStats' + '_JJA_' + reg_name + '_' + vari + '_' + model + '.nc'
    filename = directory + name
    ncfile = Dataset(filename,'w',format='NETCDF4')
    ncfile.description = '10th, 05th, and 01th percentiles for JJA Dry' 
    
    ### Dimensions
    ncfile.createDimension('ensembles',count10v.shape[0])
    ncfile.createDimension('years',count10v.shape[1])
    ncfile.createDimension('lat',count10v.shape[2])
    ncfile.createDimension('lon',count10v.shape[3])
    
    ### Variables
    ensembles = ncfile.createVariable('ensembles','f4',('ensembles'))
    years = ncfile.createVariable('years','f4',('years'))
    latitude = ncfile.createVariable('lat','f4',('lat'))
    longitude = ncfile.createVariable('lon','f4',('lon'))
    count10n = ncfile.createVariable('count10','f4',('ensembles','years','lat','lon'))
    count05n = ncfile.createVariable('count05','f4',('ensembles','years','lat','lon'))
    count01n = ncfile.createVariable('count01','f4',('ensembles','years','lat','lon'))
    freq10n = ncfile.createVariable('freq10','f4',('ensembles','years','lat','lon'))
    freq05n = ncfile.createVariable('freq05','f4',('ensembles','years','lat','lon'))
    freq01n = ncfile.createVariable('freq01','f4',('ensembles','years','lat','lon'))
    
    ### Units
    count10n.units = 'count'
    count05n.units = 'count'
    freq10n.units = 'frequency'
    freq05n.units = 'frequency'
    ncfile.title = 'Dry statistics'
    ncfile.instituion = 'NOAA GFDL SPEAR_MED'
    ncfile.references = 'Delworth et al. 2020'
    
    ### Data
    ensembles[:] = np.arange(count10v.shape[0])
    years[:] = np.arange(count10v.shape[1])
    latitude[:] = lats
    longitude[:] = lons
    count10n[:] = count10v
    count05n[:] = count05v
    count01n[:] = count01v
    freq10n[:] = freq10v
    freq05n[:] = freq05v
    freq01n[:] = freq01v
    
    ncfile.close()
    print('*Completed: Created netCDF4 File!')

# summer_LM42p2_test,lat,lon,years_LM42p2_test = readData('SPEAR_MED_LM42p2_test',reg_name)
# summer_osSSP245,lat,lon,years_osSSP245 = readData('SPEAR_MED_SSP245',reg_name)
# summer_osAMOC2,lat,lon,years_osAMOC2 = readData('SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv',reg_name)
# summer_osAMOC,lat,lon,years_osAMOC = readData('SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv',reg_name)
summer_os10ye,lat,lon,years_os10ye = readData('SPEAR_MED_SSP534OS_10ye',reg_name)
# summer_os,lat,lon,years_os = readData('SPEAR_MED_SSP534OS',reg_name)
summer,lat,lon,years = readData('SPEAR_MED',reg_name)

climdist,count10sp,count05sp,count01sp,freq10sp,freq05sp,freq01sp = calc_DryExtremes(summer,'SPEAR_MED',lat,lon,np.nan)
climdist_os10ye,count10_os10ye,count05_os10ye,count01_os10ye,freq10_os10ye,freq05_os10ye,freq01_os10ye = calc_DryExtremes(summer_os10ye,'SPEAR_MED_SSP534OS_10ye',lat,lon,climdist)
# climdist_os,count10_os,count05_os,count01_os,freq10_os,freq05_os,freq01_os = calc_DryExtremes(summer_os,'SPEAR_MED_SSP534OS',lat,lon,climdist)
# climdist_osAMOC,count10_osAMOC,count05_osAMOC,count01_osAMOC,freq10_osAMOC,freq05_osAMOC,freq01_osAMOC = calc_DryExtremes(summer_osAMOC,'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv',lat,lon,climdist)
# climdist_osAMOC2,count10_osAMOC2,count05_osAMOC2,count01_osAMOC2,freq10_osAMOC2,freq05_osAMOC2,freq01_osAMOC2 = calc_DryExtremes(summer_osAMOC2,'SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv',lat,lon,climdist)
# climdist_osSSP245,count10_osSSP245,count05_osSSP245,count01_osSSP245,freq10_osSSP245,freq05_osSSP245,freq01_osSSP245 = calc_DryExtremes(summer_osSSP245,'SPEAR_MED_SSP245',lat,lon,climdist)
# climdist_LM42p2_test,count10_LM42p2_test,count05_LM42p2_test,count01_LM42p2_test,freq10_LM42p2_test,freq05_LM42p2_test,freq01_LM42p2_test = calc_DryExtremes(summer_LM42p2_test,'SPEAR_MED_LM42p2_test',lat,lon,np.nan)

### Save data
# netcdfDry(lat,lon,count10sp,count05sp,count01sp,freq10sp,freq05sp,freq01sp,directorydata,'SPEAR_MED',reg_name,vari)
# netcdfDry(lat,lon,count10_os,count05_os,count05_os,freq10_os,freq05_os,freq01_os,directorydata,'SPEAR_MED_SSP534OS',reg_name,vari)
netcdfDry(lat,lon,count10_os10ye,count05_os10ye,count01_os10ye,freq10_os10ye,freq05_os10ye,freq01_os10ye,directorydata,'SPEAR_MED_SSP534OS_10ye',reg_name,vari)
# netcdfDry(lat,lon,count10_osAMOC,count05_osAMOC,count01_osAMOC,freq10_osAMOC,freq05_osAMOC,freq01_osAMOC,directorydata,'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv',reg_name,vari)
# netcdfDry(lat,lon,count10_osAMOC2,count05_osAMOC2,count01_osAMOC2,freq10_osAMOC2,freq05_osAMOC2,freq01_osAMOC2,directorydata,'SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv',reg_name,vari)
# netcdfDry(lat,lon,count10_osSSP245,count05_osSSP245,count01_osSSP245,freq10_osSSP245,freq05_osSSP245,freq01_osSSP245,directorydata,'SPEAR_MED_SSP245',reg_name,vari)
# netcdfDry(lat,lon,count10_LM42p2_test,count05_LM42p2_test,count01_LM42p2_test,freq10_LM42p2_test,freq05_LM42p2_test,freq01_LM42p2_test,directorydata,'SPEAR_MED_LM42p2_test',reg_name,vari)
