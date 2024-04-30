"""
Function(s) reads in monthly data from different large ensembles using the
SPEAR_LOW horizontal resolution

Notes
-----
    Author : Zachary Labe
    Date   : 10 October 2022

Usage
-----
    [1] read_MMLEA(directory,vari,sliceperiod,sliceshape,slicenan,timeper)
"""

def read_MMLEA(directory,vari,sliceperiod,sliceshape,slicenan,timeper):
    """
    Function reads monthly data from MMLEA

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
    read_MMLEA(directory,vari,sliceperiod,sliceshape,slicenan,timeper)
    """
    print('\n>>>>>>>>>> STARTING read_MMLEA function!')

    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_Utilities as UT
    import sys
    import calc_dataFunctions as df

    ###########################################################################
    ### Parameters
    years = np.arange(1921,2100+1,1)
    mon = 12
    resolution = 'LOWS'

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Function to read the various datasets   
    def read_primary_dataset(variq,dataset,monthlychoice,scenario):
        data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
        print('\nOur dataset: ',dataset,' is shaped',data.shape)
        return data,lats,lons  
    
    ### Models (can edit)
    spear,lats,lons = read_primary_dataset(vari,'SPEAR_LOW',sliceperiod,'SSP585')
    lens2,lats,lons = read_primary_dataset(vari,'LENS2_LOWS',sliceperiod,'SSP370')
    mpi,lats,lons = read_primary_dataset(vari,'MPI_ESM12_HR_LOWS',sliceperiod,'SSP370')
    miroc,lats,lons = read_primary_dataset(vari,'MIROC6_LE_LOWS',sliceperiod,'SSP585')
    lens1,lats,lons = read_primary_dataset(vari,'LENS1_LOWS',sliceperiod,'RCP85')
    flor,lats,lons = read_primary_dataset(vari,'FLOR_LOWS',sliceperiod,'RCP85')
    
    ### Create super ensemble
    dataall = np.concatenate([spear,lens2,mpi,miroc,lens1,flor],axis=0)

    print('Shape of output FINAL = ', dataall.shape,[[dataall.ndim]])
    print('>>>>>>>>>> ENDING read_MMLEA function!')    
    return lats,lons,dataall  

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directory = '/work/Zachary.Labe/Data/'
# vari = 'T2M'
# sliceperiod = 'annual'
# sliceshape = 4
# slicenan = 'nan'
# timeper = 'all'
# lat,lon,var = read_MMLEA(directory,vari,sliceperiod,sliceshape,slicenan,timeper)

# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
