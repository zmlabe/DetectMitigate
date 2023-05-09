"""
Function(s) reads in monthly data from SPEAR_LOW_Hindcast-33lev for different 
variables using # of ensemble members for 12 month chunks that are 
initialized in January, February, or March

Notes
-----
    Author : Zachary Labe
    Date   : 29 August 2022

Usage
-----
    [1] read_SPEAR_LOW_Hindcast33lev(directory,vari,month,slicenan,numOfEns)
"""

def read_SPEAR_LOW_Hindcast33lev(directory,vari,month,slicenan,numOfEns):
    """
    Function reads monthly data from SPEAR_MED

    Parameters
    ----------
    directory : string
        path for data
    vari : string
        variable for analysis
    month : string
        which month the simulation is initialized
    slicenan : string or float
        Set missing values
    numOfEns : number of ensembles
        integer

    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : numpy array
        processed variable

    Usage
    -----
    read_SPEAR_LOW_Hindcast33lev(directory,vari,month,slicenan,numOfEns)
    """
    print('\n>>>>>>>>>> STARTING read_SPEAR_LOW_Hindcast33lev function!')

    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_Utilities as UT
    import sys

    ###########################################################################
    ### Parameters
    time = np.arange(1995,2019+1,1)
    mon = 12
    ens = np.arange(1,numOfEns+1,1)
    ###########################################################################
    ### Read in data
    allyearmember = []
    for yr in range(time.shape[0]):
        year1 = time[yr]
        year2 = time[yr] + 1
        if any([month == 'Jan']):
            timeperiod = '%s-%s' % (year1,year1)
            monstr = '01'
        elif any([month=='Feb']):
            timeperiod = '%s-%s' % (year1,year2)
            monstr = '02'
        elif any([month=='Mar']):
            timeperiod = '%s-%s' % (year1,year2)
            monstr = '03'
        else:
            print(ValueError('Wrong month selected!'))
            sys.exit()
            
        directorytotal = directory + 'SPEAR-33lev/raw/%s/Y%s/' % (vari,time[yr])
            
        membersvar = []
        for i,ensmember in enumerate(ens):
            if monstr == '02' and year1 == 2009 and ensmember >= 14:
                var1 = np.empty((mon,180,288))
                var1[:] = np.nan
            elif monstr == '03' and year1 == 2007 and ensmember >= 11:
                var1 = np.empty((mon,180,288))
                var1[:] = np.nan
            elif monstr == '03' and year1 == 2008 and ensmember >= 8:
                var1 = np.empty((mon,180,288))
                var1[:] = np.nan
            else:
                filename1 = directorytotal + '%s_mon-%s_ens-%02d_%s.nc' % (vari,monstr,ensmember,timeperiod)
                data1 = Dataset(filename1,'r')
                lat1 = data1.variables['lat'][:]
                lon1 = data1.variables['lon'][:]
                var1 = data1.variables['%s' % vari][:,:,:]
                data1.close()
    
            print('Completed: read *SPEAR_LOW_Hindcast33lev* Ensemble Member --%s-%s-%s--' % (time[yr],monstr,ensmember))
            membersvar.append(var1)
        allyearmember.append(membersvar)
    allyearmember = np.asarray(allyearmember)
    
    ### Reshape array [years,ensemble members, months, latitude, longitude]
    ensshape = np.asarray(allyearmember)
    print('Completed: read all SPEAR_LOW_Hindcast33lev Members!\n')

    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        ensshape[np.where(np.isnan(ensshape))] = np.nan
        ensshape[np.where(ensshape < -999)] = np.nan 
        ensshape[np.where(ensshape >= 1e20)] = np.nan 
        print('Completed: missing values are =',slicenan)
    else:
        ensshape[np.where(np.isnan(ensshape))] = slicenan
        ensshape[np.where(ensshape < -999)] = slicenan
        ensshape[np.where(ensshape >= 1e20)] = np.nan 

    ###########################################################################
    ### Change units
    if any([vari=='T2M',vari=='SST',vari=='TMAX',vari=='TMIN']):
        ensshape = ensshape - 273.15 # K to C
        print('Completed: Changed units (K to C)!')
    elif any([vari=='PRECL',vari=='PRECC',vari=='PRECT',vari=='WA',vari=='EVAP',vari=='SNOWRATE']):
        ensshape = ensshape * 86400 # kg/m2/s to mm/day
        ### "Average Monthly Rate of Precipitation"
        print('*** CURRENT UNITS ---> [[ mm/day ]]! ***')
    elif any([vari=='SNOW']):
        ensshape = ensshape / 1000 # kg/m^2 to m
        ### "Average Monthly column-integrated snow water"
        print('*** CURRENT UNITS ---> [[ m ]]! ***')
        
    print('Shape of output preproccesed = ', ensshape.shape,[[ensshape.ndim]])
    
    ### Preprocessing stage (under construction for testing)
    output = ensshape - np.nanmean(ensshape[:,:,:,:,:],axis=(0,1))
            
    print('Shape of output FINAL = ', output.shape,[[output.ndim]])
    print('>>>>>>>>>> ENDING read_SPEAR_LOW_Hindcast33lev function!')    
    return lat1,lon1,output

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# import cmocean
# directory = '/work/Zachary.Labe/Data/SPEAR/Hindcasts/SPEAR_LOW/SPEAR-33lev/'
# vari = 'T2M'
# month = 'Mar'
# slicenan = 'nan'
# numOfEns = 15
# lat,lon,var= read_SPEAR_LOW_Hindcast33lev(directory,vari,month,slicenan,numOfEns)

# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
# mean = np.nanmean(ave,axis=1)

# ### Test functions - do not use - compare with observations
# import read_ERA5_monthlyLOWS_hindcastSPEAR_65lev as ER
# late,lone,leve,vare = ER.read_ERA5LOWS_monthlyHindcast(vari,'/work/Zachary.Labe/Data/',month,slicenan,'surface')
# clime = vare - np.nanmean(vare,axis=0)
# lon2e,lat2e = np.meshgrid(lone,late)
# avee = UT.calc_weightedAve(clime,lat2e)


