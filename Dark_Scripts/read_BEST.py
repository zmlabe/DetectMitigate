"""
Function reads in monthly data from BEST
 
Notes
-----
    Author : Zachary Labe
    Date   : 6 July 2020
    
Usage
-----
    [1] read_BEST(directory,sliceperiod,sliceyear,
                  sliceshape,addclimo,slicenan)
"""

def read_BEST(directory,sliceperiod,sliceyear,sliceshape,addclimo,slicenan):
    """
    Function reads monthly data from BEST
    
    Parameters
    ----------
    directory : string
        path for data
    sliceperiod : string
        how to average time component of data
    sliceyear : string
        how to slice number of years for data
    sliceshape : string
        shape of output array
    addclimo : binary
        True or false to add climatology
    slicenan : string or float
        Set missing values
        
    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : 3d numpy array or 4d numpy array 
        [time,lat,lon] or [year,month,lat,lon]
        
    Usage
    -----
    lat,lon,var = read_BEST(directory,sliceperiod,sliceyear,
                            sliceshape,addclimo,slicenan)
    """
    print('\n>>>>>>>>>> STARTING read_BEST function!')
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import warnings
    import calc_Utilities as UT
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ###########################################################################
    ### Parameters
    time = np.arange(1850,2022+1,1)
    monthslice = sliceyear.shape[0]*12
    mon = 12
    
    ###########################################################################
    ### Read in data
    filename = 'T2M_BEST_1850-2022.nc'
    data = Dataset(directory + filename,'r')
    lat1 = data.variables['latitude'][:]
    lon1 = data.variables['longitude'][:]
    anom = data.variables['T2M'][:,:,:]
    data.close()
    
    print('Years of output =',sliceyear.min(),'to',sliceyear.max())
    ###########################################################################
    ### Reshape data into [year,month,lat,lon]
    datamon = np.reshape(anom,(anom.shape[0]//mon,mon,
                               lat1.shape[0],lon1.shape[0]))
    
    ###########################################################################
    ### Return absolute temperature (1951-1980 baseline)
    if addclimo == False:
        filename = 'CLIM_BEST_months.nc'
        datac = Dataset(directory + filename,'r')
        clim = datac['CLIM'][:,:,:]
        datac.close()
        
        ### Add [anomaly+climatology]
        varmon = datamon + clim
        print('Completed: calculated absolute temperature!')
    else:
        varmon = datamon
        print('Completed: calculated anomalies!')
    
    ###########################################################################
    ### Slice over months (currently = [yr,mn,lat,lon])
    ### Shape of output array
    if sliceperiod == 'annual':
        vartime = np.nanmean(varmon,axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: ANNUAL MEAN!')
    elif sliceperiod == 'DJF':
        varshape = UT.calcDecJanFeb(varmon,lat1,lon1,'surface',1)
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: DJF MEAN!')
    elif sliceperiod == 'JJA':
        vartime = np.nanmean(varmon[:,5:8,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: JJA MEAN!')
    elif sliceperiod == 'JFM':
        vartime = np.nanmean(varmon[:,0:3,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: JFM MEAN!')
    elif sliceperiod == 'FMA':
        vartime = np.nanmean(varmon[:,1:4,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: FMA MEAN!')
    elif sliceperiod == 'MAM':
        vartime = np.nanmean(varmon[:,2:5,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: MAM MEAN!')
    elif sliceperiod == 'FM':
        vartime = np.nanmean(varmon[:,1:3,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: FM MEAN!')
    elif sliceperiod == 'AMJ':
        vartime = np.nanmean(varmon[:,3:6,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: AMJ MEAN!')
    elif sliceperiod == 'JAS':
        vartime = np.nanmean(varmon[:,6:9,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: JAS MEAN!')
    elif sliceperiod == 'OND':
        vartime = np.nanmean(varmon[:,9:,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif sliceshape == 3:
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: OND MEAN!')
    elif sliceperiod == 'January':
        vartime = np.nanmean(varmon[:,0:1,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: January MEAN!')
    elif sliceperiod == 'February':
        vartime = np.nanmean(varmon[:,1:2,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: February MEAN!')
    elif sliceperiod == 'March':
        vartime = np.nanmean(varmon[:,2:3,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: March MEAN!')
    elif sliceperiod == 'April':
        vartime = np.nanmean(varmon[:,3:4,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: April MEAN!')
    elif sliceperiod == 'May':
        vartime = np.nanmean(varmon[:,4:5,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: May MEAN!')
    elif sliceperiod == 'none':
        vartime = varmon
        if sliceshape == 1:
            varshape = varshape.ravel()
        elif sliceshape == 3:
            varshape = np.reshape(vartime,(vartime.shape[0]*vartime.shape[1],
                                             vartime.shape[2],vartime.shape[3]))
        elif any([sliceshape == 4,sliceshape == 5]):
            varshape = varmon
        print('Shape of output =', varshape.shape, [[varshape.ndim]])
        print('Completed: ALL MONTHS!')
        
    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        varshape[np.where(np.isnan(varshape))] = np.nan
        print('Completed: missing values are =',slicenan)
    else:
        varshape[np.where(np.isnan(varshape))] = slicenan
        
    ###########################################################################
    ### Change years
    yearhistq = np.where((time >= 1929) & (time <= 2022))[0]
    print(time[yearhistq])
    histmodel = varshape[yearhistq,:,:]
        
    print('>>>>>>>>>> ENDING read_BEST function!')
    return lat1,lon1,histmodel

### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# directory = '/Users/zlabe/Data/BEST/'
# sliceperiod = 'DJF'
# sliceyear = np.arange(1956,2019+1,1)
# sliceshape = 3
# slicenan = 'nan'
# addclimo = True
# lat,lon,var = read_BEST(directory,sliceperiod,sliceyear,sliceshape,addclimo,slicenan)
