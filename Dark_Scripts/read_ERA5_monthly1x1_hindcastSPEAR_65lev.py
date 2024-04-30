"""
Function reads in monthly data from ERA5 for 1x1 data to better evaluate
with the SPEAR_LOW hindcast experiment from SPEAR-65lev
 
Notes
-----
    Author : Zachary Labe
    Date   : 19 August 2022
    
Usage
-----
    [1] read_ERA5_monthlyHindcast(variq,directory,month,slicenan,level)
"""

def read_ERA5_monthlyHindcast(variq,directory,month,slicenan,level):
    """
    Function reads monthly data from ERA5
    
    Parameters
    ----------
    variq : string
        variable to retrieve
    directory : string
        path for data
    month : string
        which month to begin the hindcast period (12 month chunks)
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
    var : 4d numpy array or 5d array 
        [year,month,lat,lon] or [year,month,level,lat,lon]
        
    Usage
    -----
    lat,lon,lev,var = read_ERA5_monthlyHindcast(variq,directory,month,slicenan,level)
    """
    print('\n>>>>>>>>>> STARTING read_ERA5_monthlyHindcast (1x1) function!')
    
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
        yearsh = np.arange(1995,2019+2) # need some months in 2020
        mon = 12
        filename = 'ERA5_1x1/%s_1979-2021.nc' % variq
        data = Dataset(directory + filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        lev1 = level
        var = data.variables['%s' % variq][:,:,:]
        data.close()
            
        #######################################################################
        ### Reshape data into [year,month,lat,lon]
        datamon = np.reshape(var,(var.shape[0]//mon,mon,
                                   lat1.shape[0],lon1.shape[0]))
    ###########################################################################
    ###########################################################################
    ###########################################################################
    elif level == 'profile':
        time = np.arange(1979,2021+1,1)
        yearsh = np.arange(1995,2019+2) # need some months in 2020
        mon = 12
        filename = 'ERA5_1x1/%s_1979-2021.nc' % variq
        data = Dataset(directory + filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        lev1 = data.variables['level'][:]
        var = data.variables['%s' % variq][:,:,:,:]
        data.close()
            
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
    ### Reshape into 12 month chunks depending on the hindcast
    if level == 'surface':
        yearqh = np.where((time >= yearsh.min()) & (time <= yearsh.max()))[0] 
        datamon_h = datamon[yearqh]
        
        if month == 'Jan':
            varshape = datamon_h[:-1] # do not include 2020
        elif month == 'Feb':
            datamon_hravel = np.reshape(datamon_h,(datamon_h.shape[0]*mon,
                                       lat1.shape[0],lon1.shape[0]))
            varshape = []
            for i in range(0,datamon_hravel.shape[0]-12,12):
                tempchunk = datamon_hravel[i+1:i+1+12] # start in February
                varshape.append(tempchunk)
            varshape = np.asarray(varshape).reshape(yearsh.shape[0]-1,mon,lat1.shape[0],lon1.shape[0])
        elif month == 'Mar':
            datamon_hravel = np.reshape(datamon_h,(datamon_h.shape[0]*mon,
                                       lat1.shape[0],lon1.shape[0]))
            varshape = []
            for i in range(0,datamon_hravel.shape[0]-12,12):
                tempchunk = datamon_hravel[i+2:i+2+12] # start in March
                varshape.append(tempchunk)
            varshape = np.asarray(varshape).reshape(yearsh.shape[0]-1,mon,lat1.shape[0],lon1.shape[0])
    ###########################################################################
    ###########################################################################
    ########################################################################### 
    elif level == 'profile':
        yearqh = np.where((time >= yearsh.min()) & (time <= yearsh.max()))[0] 
        datamon_h = datamon[yearqh]
        
        if month == 'Jan':
            varshape = datamon_h[:-1] # do not include 2020
        elif month == 'Feb':
            datamon_hravel = np.reshape(datamon_h,(datamon_h.shape[0]*mon,
                                       lev1.shape[0],lat1.shape[0],lon1.shape[0]))
            varshape = []
            for i in range(0,datamon_hravel.shape[0]-12,12):
                tempchunk = datamon_hravel[i+1:i+1+12] # start in February
                varshape.append(tempchunk)
            varshape = np.asarray(varshape).reshape(yearsh.shape[0]-1,mon,lev1.shape[0],lat1.shape[0],lon1.shape[0])
        elif month == 'Mar':
            datamon_hravel = np.reshape(datamon_h,(datamon_h.shape[0]*mon,
                                       lev1.shape[0],lat1.shape[0],lon1.shape[0]))
            varshape = []
            for i in range(0,datamon_hravel.shape[0]-12,12):
                tempchunk = datamon_hravel[i+2:i+2+12] # start in March
                varshape.append(tempchunk)
            varshape = np.asarray(varshape).reshape(yearsh.shape[0]-1,mon,lev1.shape[0],lat1.shape[0],lon1.shape[0])
    ###########################################################################
    ###########################################################################
    ###########################################################################        
    else:
        print(ValueError('SOMETHING IS WRONG WITH THE LEVEL SELECTED!'))
        sys.exit()

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
    elif variq == 'P':
        varshape = varshape*1000 # m/day to mm/day
        ### "Average Monthly Rate of Precipitation"
        print('*** CURRENT UNITS ---> [[ mm/day ]]! ***')
        
    print('Shape of the final ERA5 array is %s!' % [varshape.shape])
    print('>>>>>>>>>> ENDING read_ERA5_monthlyHindcast (1x1) function for -- %s!' % variq)
    return lat1,lon1,lev1,varshape

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# variq = 'T2M'
# directory = '/work/Zachary.Labe/Data/'
# month = 'Jan'
# years = np.arange(1979,2021+1,1)
# slicenan = 'nan'
# level = 'surface'
# lat,lon,lev,var1 = read_ERA5_monthlyHindcast(variq,directory,month,slicenan,level)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave1 = UT.calc_weightedAve(var1,lat2)

# lat,lon,lev,var2 = read_ERA5_monthlyHindcast(variq,directory,'Feb',slicenan,level)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave2 = UT.calc_weightedAve(var2,lat2)

# lat,lon,lev,var3 = read_ERA5_monthlyHindcast(variq,directory,'Mar',slicenan,level)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave3 = UT.calc_weightedAve(var3,lat2)
