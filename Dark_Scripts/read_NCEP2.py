"""
Function reads in monthly data from NCEP2
 
Notes
-----
    Author : Zachary Labe
    Date   : 10 January 2022
    
Usage
-----
    [1] read_NCEP2(directory,sliceperiod,sliceyear,
                  sliceshape,addclimo,slicenan,newon)
"""

def read_NCEP2(directory,sliceperiod,sliceyear,sliceshape,addclimo,slicenan,newon):
    """
    Function reads monthly data from NCEP2
    
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
    newon : binary
        True or false for reading in new data from NCEP2 not processed
        
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
    lat,lon,var = read_NCEP2(directory,sliceperiod,sliceyear,
                            sliceshape,addclimo,slicenan)
    """
    print('\n>>>>>>>>>> STARTING read_NCEP2 function!')
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import warnings
    import calc_Utilities as UT
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ###########################################################################
    ### Parameters
    mon = 12
    variq = 'T2M'
    
    ###########################################################################
    ### Read in data
    if newon == False:
        time = np.arange(1979,2021+1,1)
        mon = 12
        monthslice = sliceyear.shape[0]*mon
        filename = 'monthly/%s_NCEP2_1979-2021.nc' % variq
        data = Dataset(directory + filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        var = data.variables['%s' % variq][:,:,:]
        data.close()
    elif newon == True:
        time = np.arange(1979,2022+1,1)
        mon = 12
        monthslice = sliceyear.shape[0]*mon
        filename = 'monthly/%s_NCEP2_1979-2022.nc' % variq
        data = Dataset(directory + filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        var = data.variables['%s' % variq][:,:,:]
        data.close()
        
        empty = np.empty((time.shape[0]*12-len(var),lat1.shape[0],lon1.shape[0]))
        empty[:] = np.nan
        yr2022 = np.append(var[-12+len(empty):,:,:],empty,axis=0)
        temp = np.reshape(var[:-12+len(empty)],(var.shape[0]//12,12,lat1.shape[0],lon1.shape[0]))
        tempqq = np.append(temp,yr2022[np.newaxis,:,:,:],axis=0)
        var = tempqq.reshape(tempqq.shape[0]*12,lat1.shape[0],lon1.shape[0])
    
    print('Years of output =',sliceyear.min(),'to',sliceyear.max())
    ###########################################################################
    ### Reshape data into [year,month,lat,lon]
    datamon = np.reshape(var,(var.shape[0]//mon,mon,
                               lat1.shape[0],lon1.shape[0]))
    
    ###########################################################################
    ### Return absolute temperature (1981-2010 baseline)
    if addclimo == True:
        varmon = datamon
        print('Completed: calculated absolute variable!')
    else:
        yearbasemin = 1981
        yearbasemax = 2010
        yearq = np.where((time >= yearbasemin) & (time<=yearbasemax))[0]
        varmon = datamon - np.nanmean(datamon[yearq,:,:,:],axis=0)
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
        varshape[np.where(varshape < -9999)] = np.nan
        print('Completed: missing values are =',slicenan)
    else:
        varshape[np.where(np.isnan(varshape < -9999))] = slicenan
        
    ###########################################################################
    ### Change units
    if variq == 'T2M':
        varshape = varshape - 273.15 # K to C
        print('Completed: Changed units (K to C)!')
       
    print('>>>>>>>>>> ENDING read_NCEP2 function!')
    return lat1,lon1,varshape

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# directory = '/Users/zlabe/Data/NCEP2/'
# sliceperiod = 'FMA'
# sliceyear = np.arange(1979,2022+1,1)
# sliceshape = 3
# slicenan = 'nan'
# addclimo = False
# newon = True
# lat,lon,var = read_NCEP2(directory,sliceperiod,sliceyear,sliceshape,addclimo,slicenan,newon)
