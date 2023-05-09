"""
Function reads in monthly data from CMIP6 on native grid
 
Notes
-----
    Author : Zachary Labe
    Date   : 11 August 2022
    
Usage
-----
    [1] read_CMIP6_monthly(variq,GCM,months,sliceshape,periodyrs,slicenan,level)
"""

def read_CMIP6_monthly(variq,GCM,months,sliceshape,periodyrs,slicenan,level):
    """
    Function reads monthly data from ERA5
    
    Parameters
    ----------
    variq : string
        variable to retrieve
    GCM : string
        model name
    months : string
        how to average time component of data
    sliceshape : string
        shape of output array
    periodyrs : string
        select which set of years to use for each variable
    slicenan : string or float
        Set missing values
    level : string
        surface or vertical
        
    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    lev : string or 1d array
        surface or vertical levels in an array
    var : 3d numpy array or 4d numpy array or 5d array 
        [time,lat,lon] or [year,month,lat,lon] or [year,month,level,lat,lon]
        
    Usage
    -----
    lat,lon,lev,var = read_CMIP6_monthly(variq,GCM,months,sliceshape,periodyrs,slicenan,level)
    """
    print('\n>>>>>>>>>> STARTING read_CMIP6_monthly function! for -- %s -- %s!' % (variq,GCM))
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import warnings
    import calc_Utilities as UT
    import sys
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ### Parameters
    mon = 12
    directory = '/work/Zachary.Labe/Data/CMIP6/'
    directoryGCM = directory + '%s/monthly/' % GCM
    filename = directoryGCM + '%s_%s_%s.nc' % (variq,GCM,periodyrs)
    
    ###########################################################################
    ### Read in data
    if level == 'surface':
        data = Dataset(filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        lev1 = level
        var = data.variables['%s' % variq][:,:,:]
        data.close()
            
        #######################################################################
        ### Reshape data into [year,month,lat,lon]
        varmon = np.reshape(var,(var.shape[0]//mon,mon,
                                   lat1.shape[0],lon1.shape[0]))
    ###########################################################################
    ###########################################################################
    ###########################################################################
    elif level == 'vertical':
        data = Dataset(filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        lev1 = data.variables['plev'][:]
        var = data.variables['%s' % variq][:,:,:,:]
        data.close()
            
        #######################################################################
        ### Reshape data into [year,month,level,lat,lon]
        varmon = np.reshape(var,(var.shape[0]//mon,mon,lev1.shape[0],
                                   lat1.shape[0],lon1.shape[0]))
    ###########################################################################
    ###########################################################################
    ###########################################################################        
    else:
        print(ValueError('SOMETHING IS WRONG WITH THE LEVEL SELECTED!'))
        sys.exit()
    
    ###########################################################################
    ### Slice over months (currently = [yr,mn,lat,lon])
    ### Shape of output array
    if months == 'annual':
        vartime = np.nanmean(varmon,axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: ANNUAL MEAN!')
    elif months == 'DJF':
        varshape = UT.calcDecJanFeb(varmon,lat1,lon1,'surface',1)
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: DJF MEAN!')
    elif months == 'JJA':
        vartime = np.nanmean(varmon[:,5:8,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: JJA MEAN!')
    elif months == 'JFM':
        vartime = np.nanmean(varmon[:,0:3,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: JFM MEAN!')
    elif months == 'FMA':
        vartime = np.nanmean(varmon[:,1:4,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: FMA MEAN!')
    elif months == 'MAM':
        vartime = np.nanmean(varmon[:,2:5,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: MAM MEAN!')
    elif months == 'FM':
        vartime = np.nanmean(varmon[:,1:3,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: FM MEAN!')
    elif months == 'AMJ':
        vartime = np.nanmean(varmon[:,3:6,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: AMJ MEAN!')
    elif months == 'JAS':
        vartime = np.nanmean(varmon[:,6:9,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: JAS MEAN!')
    elif months == 'OND':
        vartime = np.nanmean(varmon[:,9:,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif sliceshape == 3:
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: OND MEAN!')
    elif months == 'January':
        vartime = np.nanmean(varmon[:,0:1,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: January MEAN!')
    elif months == 'February':
        vartime = np.nanmean(varmon[:,1:2,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: February MEAN!')
    elif months == 'March':
        vartime = np.nanmean(varmon[:,2:3,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: March MEAN!')
    elif months == 'April':
        vartime = np.nanmean(varmon[:,3:4,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: April MEAN!')
    elif months == 'May':
        vartime = np.nanmean(varmon[:,4:5,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: May MEAN!')
    elif months == 'AMJJ':
        vartime = np.nanmean(varmon[:,3:7,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: AMJJ MEAN!')
    elif months == 'none':
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
    ### Change units
    if variq == 'SLP':
        varshape = varshape/100 # Pa to hPa
        print('Completed: Changed units (Pa to hPa)!')
    elif any([variq=='T2M',variq=='SST']):
        varshape = varshape - 273.15 # K to C
        print('Completed: Changed units (K to C)!')
        
    print('>>>>>>>>>> ENDING read_CMIP6_monthly function for -- %s -- %s!' % (variq,GCM))
    return lat1,lon1,lev1,varshape

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# variq = 'GEOP'
# GCM = 'CESM2_WACCM6'
# months = 'annual'
# periodyrs = '1979-2022'
# sliceshape = 4
# slicenan = 'nan'
# level = 'vertical'
# lat,lon,lev,var = read_CMIP6_monthly(variq,GCM,months,sliceshape,periodyrs,slicenan,level)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
