"""
Function(s) reads in monthly data from SPEAR_MED_Scenarios for different variables 
using # of ensemble members for all time periods

Notes
-----
    Author : Zachary Labe
    Date   : 2 June 2022

Usage
-----
    [1] read_SPEAR_MED_Scenario(directory,scenario,vari,sliceperiod,sliceshape,slicenan,numOfEns,timeper)
"""

def read_SPEAR_MED_Scenario(directory,scenario,vari,sliceperiod,sliceshape,slicenan,numOfEns,timeper):
    """
    Function reads monthly data from SPEAR_MED_Scenario

    Parameters
    ----------
    directory : string
        path for data
    scenario : string
         SSP119 or SSP245 or SSP534OS or SSP370
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
    read_SPEAR_MED_Scenario(directory,scenario,vari,sliceperiod,sliceshape,
                             slicenan,numOfEns,timeper)
    """
    print('\n>>>>>>>>>> STARTING read_SPEAR_MED_Scenario function!')

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
    
    timesat1 = np.arange(1921,2010+1,1)
    timesat2 = np.arange(2011,2100+1,1)
 
    ###########################################################################
    ### Climate change emissions scenario
    if scenario == 'SSP119':
        directoryScenario = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_%s/monthly/' % scenario
    elif scenario == 'SSP245':
        directoryScenario = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_%s/monthly/' % scenario
    elif scenario == 'SSP370':
        directoryScenario = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_%s/monthly/' % scenario
    elif scenario == 'SSP534OS':
        directoryScenario = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_%s/monthly/' % scenario
    elif scenario == 'SSP585':
        '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/'
    else:
        print(ValueError('WRONG scenario picked for SPEAR!!!'))
        sys.exit()
    print('Scenario for this simulation of SPEAR_MED is ---> %s!' % scenario)

    ###########################################################################
    ### Read in data
    membersvar1 = []
    for i,ensmember in enumerate(ens):
        filename1 = directory + '%s/%s_%02d_1921-2010.nc' % (vari,vari,
                                                          ensmember)
        data1 = Dataset(filename1,'r')
        lat1 = data1.variables['lat'][:]
        lon1 = data1.variables['lon'][:]
        var1 = data1.variables['%s' % vari][-timesat1.shape[0]*mon:,:,:]
        data1.close()

        print('Completed: read *SPEAR_MED historical* Ensemble Member --%s--' % ensmember)
        membersvar1.append(var1)
        del var1
    
    membersvar2 = []
    for i,ensmember in enumerate(ens):
        filename2 = directoryScenario + '%s/%s_%02d_2011-2100.nc' % (vari,vari,
                                                          ensmember)
        data2 = Dataset(filename2,'r')
        lat1 = data2.variables['lat'][:]
        lon1 = data2.variables['lon'][:]
        var2 = data2.variables['%s' % vari][:timesat2.shape[0]*mon,:,:]
        data2.close()

        print('Completed: read *SPEAR_MED %s Ensemble Member --%s--' % (scenario,ensmember))
        membersvar2.append(var2)
        del var2
        
    ### Append both time periods together
    membersvar1 = np.asarray(membersvar1)
    membersvar2 = np.asarray(membersvar2)
    membersvar = np.append(membersvar1,membersvar2,axis=1)
    ensvalue = np.reshape(membersvar,(len(ens),time.shape[0],mon,
                                    lat1.shape[0],lon1.shape[0]))
    del membersvar
    del membersvar1
    del membersvar2
    print('Completed: appended all SPEAR_MED Members!\n')

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
    elif sliceperiod == 'June':
        enstime = np.nanmean(ensvalue[:,:,5:6,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: June MEAN!')
    elif sliceperiod == 'July':
        enstime = np.nanmean(ensvalue[:,:,6:7,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: July MEAN!')
    elif sliceperiod == 'August':
        enstime = np.nanmean(ensvalue[:,:,7:8,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: August MEAN!')
    elif sliceperiod == 'September':
        enstime = np.nanmean(ensvalue[:,:,8:9,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: September MEAN!')
    elif sliceperiod == 'October':
        enstime = np.nanmean(ensvalue[:,:,9:10,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: October MEAN!')
    elif sliceperiod == 'November':
        enstime = np.nanmean(ensvalue[:,:,10:11,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: November MEAN!')
    elif sliceperiod == 'December':
        enstime = np.nanmean(ensvalue[:,:,11:12,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: December MEAN!')
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
        ensshape[np.where(ensshape <= -999)] = np.nan 
        ensshape[np.where(ensshape >= 1e20)] = np.nan 
        print('Completed: missing values are =',slicenan)
    else:
        ensshape[np.where(np.isnan(ensshape))] = slicenan
        ensshape[np.where(ensshape <= -999)] = slicenan
        ensshape[np.where(ensshape >= 1e20)] = slicenan

    ###########################################################################
    ### Change units
    if any([vari=='T2M',vari=='SST',vari=='TMAX',vari=='TMIN',vari=='TMAXabs',vari=='TMINabs']):
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
    elif any([vari == 'tau_x',vari == 'tau_y']):
        ensshape = ensshape * -1 # (-1 x forcing on the atmosphere; show downward stress)
        print('*** CURRENT UNITS (multiplied by -1) ---> [[ N/m2 ]]! ***')
    
    ###########################################################################
    ### Select years of analysis (1921-2100)
    if timeper == 'all':
        print('ALL SIMULATION YEARS')
        print(time)
        histmodel = ensshape
    elif timeper == 'futureforcing':
        yearhistq = np.where((time >= 2015) & (time <= 2100))[0]
        print('FUTURE YEARS')
        print(time[yearhistq])
        histmodel = ensshape[:,yearhistq,:,:]
    elif timeper == '10ye':
        yearhistq = np.where((time >= 1921) & (time <= 2030))[0]
        print('10ye')
        print(time[yearhistq])
        histmodel = ensshape[:,yearhistq,:,:]
    elif timeper == 'OS40a':
        yearhistq = np.where((time >= 1921) & (time <= 2040))[0]
        print('OS40a')
        print(time[yearhistq])
        histmodel = ensshape[:,yearhistq,:,:]
    elif timeper == 'STRONGAMOC':
        yearhistq = np.where((time >= 1921) & (time <= 2040))[0]
        print('STRONGAMOC')
        print(time[yearhistq])
        histmodel = ensshape[:,yearhistq,:,:]
    else:
        print(ValueError('SOMETHING IS WRONG WITH THE YEARS OF ANALYSIS!!!'))
        sys.exit()

    print('Shape of output FINAL = ', histmodel.shape,[[histmodel.ndim]])
    print('>>>>>>>>>> ENDING read_SPEAR_MED_Scenario function!')    
    return lat1,lon1,histmodel 

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directory = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/'
# vari = 'T2M'
# sliceperiod = 'annual'
# sliceshape = 4
# slicenan = 'nan'
# numOfEns = 30
# timeper = 'all'
# scenario = 'SSP534OS'
# lat,lon,var = read_SPEAR_MED_Scenario(directory,scenario,vari,sliceperiod,sliceshape,slicenan,numOfEns,timeper)

# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
