"""
Function(s) reads in monthly data from FLOR_LOWS large ensemble for different 
variables using 30 ensemble members for all time periods

Notes
-----
    Author : Zachary Labe
    Date   : 12 October 2022

Usage
-----
    [1] read_FLOR_LOWS(directory,vari,sliceperiod,sliceshape,slicenan,numOfEns,timeper)
"""

def read_FLOR_LOWS(directory,vari,sliceperiod,sliceshape,slicenan,numOfEns,timeper):
    """
    Function reads monthly data from FLOR_LOWS

    Parameters
    ----------
    directory : string
        path for data
    vari : string
        variable for analysis
    sliceperiod : string
        how to average time component of data
    sliceshape : string
        shape of output array
    slicenan : string or float
        Set missing values
    numOfEns : number of ensembles
        integer
    timeper : time period of analysis
        string

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
    read_FLOR_LOWS(directory,vari,sliceperiod,sliceshape,
                             slicenan,numOfEns,timeper)
    """
    print('\n>>>>>>>>>> STARTING read_FLOR_LOWS function!')
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_Utilities as UT
    import sys

    ###########################################################################
    ### Parameters
    time = np.arange(1921,2100+1,1)
    mon = 12
    ens = np.arange(1,numOfEns+1,1)

    ###########################################################################
    ### Read in data
    membersvar = []
    for i,ensmember in enumerate(ens):
        filename1 = directory + 'FLOR/FLOR_LOWS/monthly/%s/%s_%03d_1921-2100.nc' % (vari,vari,
                                                          ensmember)
        data1 = Dataset(filename1,'r')
        lat1 = data1.variables['lat'][:]
        lon1 = data1.variables['lon'][:]
        var = data1.variables['%s' % vari][:,:,:]
        data1.close()

        print('Completed: read *FLOR_LOWS* Ensemble Member --%s-- for %s' % (ensmember,vari))
        membersvar.append(var)
        del var
        
    ensvalue = np.reshape(np.asarray(membersvar),(len(ens),time.shape[0],mon,
                                    lat1.shape[0],lon1.shape[0]))
    del membersvar
    print('Completed: read all FLOR_LOWS Members!\n')

    ###########################################################################
    ### Slice over months (currently = [ens,yr,mn,lat,l10on])
    ### Shape of output array
    if sliceperiod == 'annual':
        ensvalue = np.nanmean(ensvalue,axis=2)
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = ensvalue
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: ANNUAL MEAN!')
    elif sliceperiod == 'DJF':
        ensshape = np.empty((ensvalue.shape[0],ensvalue.shape[1]-1,
                             lat1.shape[0],lon1.shape[0]))
        for i in range(ensvalue.shape[0]):
            ensshape[i,:,:,:] = UT.calcDecJanFeb(ensvalue[i,:,:,:,:],
                                                 lat1,lon1,'surface',1)
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: DJF MEAN!')
    elif sliceperiod == 'MAM':
        enstime = np.nanmean(ensvalue[:,:,2:5,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: MAM MEAN!')
    elif sliceperiod == 'JJA':
        enstime = np.nanmean(ensvalue[:,:,5:8,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JJA MEAN!')
    elif sliceperiod == 'SON':
        enstime = np.nanmean(ensvalue[:,:,8:11,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: SON MEAN!')
    elif sliceperiod == 'JFM':
        enstime = np.nanmean(ensvalue[:,:,0:3,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JFM MEAN!')
    elif sliceperiod == 'FMA':
        enstime = np.nanmean(ensvalue[:,:,1:4,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: FMA MEAN!')        
    elif sliceperiod == 'FM':
        enstime = np.nanmean(ensvalue[:,:,1:3,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: FM MEAN!')
    elif sliceperiod == 'AMJ':
        enstime = np.nanmean(ensvalue[:,:,3:6,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: AMJ MEAN!')
    elif sliceperiod == 'JAS':
        enstime = np.nanmean(ensvalue[:,:,6:9,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JAS MEAN!')
    elif sliceperiod == 'OND':
        enstime = np.nanmean(ensvalue[:,:,9:,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: OND MEAN!')
    elif sliceperiod == 'January':
        enstime = np.nanmean(ensvalue[:,:,0:1,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: January MEAN!')
    elif sliceperiod == 'February':
        enstime = np.nanmean(ensvalue[:,:,1:2,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: February MEAN!')
    elif sliceperiod == 'March':
        enstime = np.nanmean(ensvalue[:,:,2:3,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: March MEAN!')
    elif sliceperiod == 'April':
        enstime = np.nanmean(ensvalue[:,:,3:4,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: April MEAN!')
    elif sliceperiod == 'May':
        enstime = np.nanmean(ensvalue[:,:,4:5,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: May MEAN!')
    elif sliceperiod == 'none':
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif sliceshape == 4:
            ensshape = np.reshape(ensvalue,(ensvalue.shape[0],ensvalue.shape[1]*ensvalue.shape[2],
                                             ensvalue.shape[3],ensvalue.shape[4]))
        elif any([sliceshape == 5,sliceshape == 6]):
            ensshape = ensvalue
        print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
        print('Completed: ALL RAVELED MONTHS!')
    else:
        print(ValueError('WRONG MONTHS SELECTED FOR WHAT IS AVAILABLE!!!'))
        sys.exit()

    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        ensshape[np.where(np.isnan(ensshape))] = np.nan
        ensshape[np.where(ensshape < -999)] = np.nan 
        print('Completed: missing values are =',slicenan)
    else:
        ensshape[np.where(np.isnan(ensshape))] = slicenan
        ensshape[np.where(ensshape < -999)] =slicenan

    ###########################################################################
    ### Change units
    if any([vari=='T2M',vari=='SST',vari=='TMAX',vari=='TMIN']):
        ensshape = ensshape - 273.15 # K to C
        print('Completed: Changed units (K to C)!')
    elif any([vari=='PRECL',vari=='PRECC',vari=='PRECT']):
        ensshape = ensshape * 86400 # kg/m2/s to mm/day
        ### "Average Monthly Rate of Precipitation"
        print('*** CURRENT UNITS ---> [[ mm/day ]]! ***')
    elif any([vari=='SNOW']):
        ensshape = ensshape / 1000 # kg/m^2 to m
        ### "Average Monthly column-integrated snow water"
        print('*** CURRENT UNITS ---> [[ m ]]! ***')
    
    ###########################################################################
    ### Select years of analysis (1921-2100)
    if timeper == 'all':
        print('ALL SIMULATION YEARS')
        print(time)
        histmodel = ensshape
    elif timeper == 'satellite':
        yearhistq = np.where((time >= 1979) & (time <= 2020))[0]
        print('HISTORICAL YEARS')
        print(time[yearhistq])
        histmodel = ensshape[:,yearhistq,:,:]
    elif timeper == 'historical':
        yearhistq = np.where((time >= 1921) & (time <= 2021))[0]
        print('HISTORICAL YEARS')
        print(time[yearhistq])
        histmodel = ensshape[:,yearhistq,:,:]
    elif timeper == 'future':
        yearhistq = np.where((time >= 2023) & (time <= 2100))[0]
        print('FUTURE YEARS')
        print(time[yearhistq])
        histmodel = ensshape[:,yearhistq,:,:]
    else:
        print(ValueError('SOMETHING IS WRONG WITH THE YEARS OF ANALYSIS!!!'))
        sys.exit()

    print('Shape of output FINAL = ', histmodel.shape,[[histmodel.ndim]])
    print('>>>>>>>>>> ENDING read_FLOR_LOWS function!')    
    return lat1,lon1,histmodel 

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directory = '/work/Zachary.Labe/Data/'
# vari = 'T2M'
# sliceperiod = 'annual'
# sliceshape = 4
# slicenan = 'nan'
# numOfEns = 30
# timeper = 'all'
# lat,lon,var = read_FLOR_LOWS(directory,vari,sliceperiod,sliceshape,slicenan,numOfEns,timeper)

# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
