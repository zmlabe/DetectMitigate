"""
Function reads in monthly data from ERA5 for LOWS data
 
Notes
-----
    Author : Zachary Labe
    Date   : 29 August 2022
    
Usage
-----
    [1] read_ERA5_monthlyLOWS(variq,directory,sliceperiod,sliceyear,
                  sliceshape,addclimo,slicenan,level)
"""

def read_ERA5_monthlyLOWS(variq,directory,sliceperiod,sliceyear,sliceshape,addclimo,slicenan,level):
    """
    Function reads monthly data from ERA5
    
    Parameters
    ----------
    variq : string
        variable to retrieve
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
    lat,lon,lev,var = read_ERA5_monthlyLOWS(variq,directory,sliceperiod,sliceyear,
                            sliceshape,addclimo,slicenan,level)
    """
    print('\n>>>>>>>>>> STARTING read_ERA5_monthly function!')
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import warnings
    import calc_Utilities as UT
    import sys
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ###########################################################################
    ### Change variable names if needed
    if variq == 'SNOW':
        variq = 'SND'
    
    ###########################################################################
    ### Read in data
    if level == 'surface':
        time = np.arange(1979,2021+1,1)
        mon = 12
        monthslice = sliceyear.shape[0]*mon
        filename = 'ERA5_LOWS/%s_1979-2021.nc' % variq
        data = Dataset(directory + filename,'r')
        lat1 = data.variables['lat'][:]
        lon1 = data.variables['lon'][:]
        lev1 = level
        var = data.variables['%s' % variq][:,:,:]
        data.close()
            
        print('Years of output =',sliceyear.min(),'to',sliceyear.max())
        #######################################################################
        ### Reshape data into [year,month,lat,lon]
        datamon = np.reshape(var,(var.shape[0]//mon,mon,
                                   lat1.shape[0],lon1.shape[0]))
    ###########################################################################
    ###########################################################################
    ###########################################################################
    elif level == 'vertical':
        time = np.arange(1979,2021+1,1)
        mon = 12
        monthslice = sliceyear.shape[0]*mon
        filename = 'ERA5_LOWS/%s_1979-2021.nc' % variq
        data = Dataset(directory + filename,'r')
        lat1 = data.variables['lat'][:]
        lon1 = data.variables['lon'][:]
        lev1 = data.variables['level'][:]
        var = data.variables['%s' % variq][:,:,:,:]
        data.close()
            
        print('Years of output =',sliceyear.min(),'to',sliceyear.max())
        #######################################################################
        ### Reshape data into [year,month,level,lat,lon]
        datamon = np.reshape(var,(var.shape[0]//mon,mon,lev1.shape[0],
                                   lat1.shape[0],lon1.shape[0]))
    ###########################################################################
    ###########################################################################
    ###########################################################################        
    else:
        print(ValueError('SOMETHING IS WRONG WITH THE LEVEL SELECTED!'))
        sys.exit()
    
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
    elif sliceperiod == 'MA':
        vartime = np.nanmean(varmon[:,2:4,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: MA MEAN!')
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
    elif sliceperiod == 'AMJJ':
        vartime = np.nanmean(varmon[:,3:7,:,:],axis=1)
        if sliceshape == 1:
            varshape = vartime.ravel()
        elif any([sliceshape == 3,sliceshape == 4]):
            varshape = vartime
        print('Shape of output = ', varshape.shape,[[varshape.ndim]])
        print('Completed: AMJJ MEAN!')
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
    ### Change units
    if variq == 'SLP':
        varshape = varshape/100 # Pa to hPa
        print('Completed: Changed units (Pa to hPa)!')
    elif any([variq=='T2M',variq=='SST',variq=='TMAX',variq=='TMIN']):
        varshape = varshape - 273.15 # K to C
        print('Completed: Changed units (K to C)!')
    elif variq == 'P':
        varshape = varshape*1000 # m/day to mm/day
        ### "Average Monthly Rate of Precipitation"
        print('*** CURRENT UNITS ---> [[ mm/day ]]! ***')
        
    print('>>>>>>>>>> ENDING read_ERA5_monthlyLOWS function for -- %s!' % variq)
    return lat1,lon1,lev1,varshape

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# variq = 'T925'
# directory = '/work/Zachary.Labe/Data/'
# sliceperiod = 'annual'
# sliceyear = np.arange(1979,2021+1,1)
# sliceshape = 4
# slicenan = 'nan'
# addclimo = True
# level = 'surface'
# lat,lon,lev,var = read_ERA5_monthlyLOWS(variq,directory,sliceperiod,
#                                 sliceyear,sliceshape,addclimo,
#                                 slicenan,level)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
