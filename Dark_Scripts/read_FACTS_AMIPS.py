"""
Function(s) reads in monthly data from the FACTS repository using different
experiments and climate models for all available ensemble members. The start
of the script is the metadata needed for each experiment.

Notes
-----
    Author : Zachary Labe
    Date   : 2 June 2022

Usage
-----
    [1] read_FACTS_Experi(scenario,model,vari,slicemonth,sliceshape,slicenan,level)
"""

def read_FACTS_Experi(scenario,model,vari,slicemonth,sliceshape,slicenan,level):
    """
    Function reads monthly data from SPEAR_MED_Scenario

    Parameters
    ----------
    scenario : string
         amip_1880s_rf or amip_obs_rf or amip_clim_polar
    model : string
        ECHAM5 or CAM4 or ESRL-GFS or ESRL-CAM5
    vari : string
        variable for analysis
    slicemonth : string
        how to average time component of data
    sliceshape : string
        shape of output array
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
    var : numpy array
        processed variable

    Usage
    -----
    read_FACTS_Experi(scenario,model,vari,slicemonth,sliceshape,slicenan,level)
    """
    print('\n>>>>>>>>>> STARTING read_FACTS_Experi function!')

    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_Utilities as UT
    import sys

    ###########################################################################
    ### Parameters
    mon = 12
    
    ###########################################################################
    ### Setting up FACTS-AMIP simulation information
    if scenario == 'amip_1880s_rf':
        directory = '/work/Zachary.Labe/Data/FACTS/%s/' % scenario
        if model == 'CAM4':
            yrmin = 1979
            yrmax = 2012
            numOfEns = 20
        elif model == 'ECHAM5':
            yrmin = 1979
            yrmax = 2021
            numOfEns = 50
        elif model == 'ESRL-CAM5':
            yrmin = 1979
            yrmax = 2020
            numOfEns = 40       
            if vari == 'SNOW':
                numOfEns = 30
        else:
            print(ValueError('WRONG MODEL FOR AMIP EXPERIMENT!!!'))
            sys.exit()
    elif scenario == 'amip_obs_rf':
        directory = '/work/Zachary.Labe/Data/FACTS/%s/' % scenario
        if model == 'CAM4':
            yrmin = 1979
            yrmax = 2015
            numOfEns = 20
        elif model == 'ECHAM5':
            yrmin = 1979
            yrmax = 2021
            numOfEns = 50
        elif model == 'ESRL-CAM5':
            yrmin = 1900
            yrmax = 2020
            numOfEns = 40  
        else:
            print(ValueError('WRONG MODEL FOR AMIP EXPERIMENT!!!'))
            sys.exit()
    elif scenario == 'amip_clim_polar':
        directory = '/work/Zachary.Labe/Data/FACTS/%s/' % scenario
        if model == 'CAM4':
            yrmin = 1979
            yrmax = 2015
            numOfEns = 20
        elif model == 'ECHAM5':
            yrmin = 1979
            yrmax = 2018
            numOfEns = 30
        elif model == 'ESRL-CAM5':
            yrmin = 1979
            yrmax = 2017
            numOfEns = 20       
        elif model == 'ESRL-GFS':
            yrmin = 1979
            yrmax = 2015
            numOfEns = 30  
        else:
            print(ValueError('WRONG MODEL FOR AMIP EXPERIMENT!!!'))
            sys.exit()
    elif scenario == 'spear':
        directory = '/work/Zachary.Labe/Data/SPEAR/SPEAR_c192_pres_HadOISST_HIST_AllForc_Q50/monthly/'
        if model == 'spear':
            yrmin = 1921
            yrmax = 2020
            numOfEns = 3
        else:
            print(ValueError('WRONG MODEL FOR AMIP EXPERIMENT!!!'))
            sys.exit()            
    else:
        print(ValueError('WRONG AMIP SCENARIO!!!'))
        sys.exit()
    
    time = np.arange(yrmin,yrmax+1,1)
    ens = np.arange(1,numOfEns+1,1)
    print('AMIP Scenario ----> %s for Model ----> %s' % (scenario,model))

    ###########################################################################
    ### Read in data
    if level == 'surface':
        membersvar = []
        for i,ensmember in enumerate(ens):
            if scenario == 'spear':
                filename = directory + '%s_%02d_%s-%s.nc' % (vari,ensmember,yrmin,yrmax)        
            else:
                filename = directory + '%s/%s_%02d_%s-%s.nc' % (model,vari,ensmember,yrmin,yrmax)
            data = Dataset(filename,'r')
            lat1 = data.variables['lat'][:]
            lon1 = data.variables['lon'][:]
            lev1 = 'surface'
            var = data.variables['%s' % vari][:,:,:]
            data.close()
            print('Completed: read *%s-%s* Ensemble Member --%s--' % (scenario,
                                                                      model,
                                                                      ensmember))
            
            ### Not all ensemble members go through the end of the year
            if len(var) != time.shape[0]*12:
                empty = np.empty((time.shape[0]*12-len(var),lat1.shape[0],lon1.shape[0]))
                empty[:] = np.nan
                var = np.append(var,empty,axis=0)
            
            membersvar.append(var)
            del var
    
        ensvalue = np.reshape(membersvar,(len(ens),time.shape[0],mon,
                                        lat1.shape[0],lon1.shape[0]))
        del membersvar
    ###########################################################################
    ###########################################################################
    ###########################################################################
    elif level == 'vertical':
        if scenario == 'spear':
            numOfEns = 3
            ens = np.arange(1,numOfEns+1,1)
        else:
            numOfEns = 10
            ens = np.arange(1,numOfEns+1,1)
            print('\nCHANGED NUMBER OF ENSEMBLES DUE TO VERTICAL LEVELS!!!\n')
        membersvar = []
        for i,ensmember in enumerate(ens):
            if scenario == 'spear':
                filename = directory + '%s_%02d_%s-%s.nc' % (vari,ensmember,yrmin,yrmax)  
                nameOfLevelVar = 'level'
            else:
                filename = directory + '%s/%s_%02d_%s-%s.nc' % (model,vari,ensmember,yrmin,yrmax)
                nameOfLevelVar = 'plev'
            data = Dataset(filename,'r')
            lat1 = data.variables['lat'][:]
            lon1 = data.variables['lon'][:]
            lev1 = data.variables[nameOfLevelVar][:]
            var = data.variables['%s' % vari][:,:,:,:]
            data.close()
            print('Completed: read *%s-%s* Ensemble Member --%s--' % (scenario,
                                                                      model,
                                                                      ensmember))
            
            ### Not all ensemble members go through the end of the year
            if len(var) != time.shape[0]*12:
                empty = np.empty((time.shape[0]*12-len(var),lev1.shape[0],lat1.shape[0],lon1.shape[0]))
                empty[:] = np.nan
                var = np.append(var,empty,axis=0)
            
            membersvar.append(var)
            del var
    
        ensvalue = np.reshape(membersvar,(len(ens),time.shape[0],mon,
                                          lev1.shape[0],
                                          lat1.shape[0],lon1.shape[0]))
        del membersvar

    ###########################################################################
    ### Slice over months (currently = [ens,yr,mn,lat,lon])
    ### Shape of output array
    if slicemonth == 'annual':
        ensvalue = np.nanmean(ensvalue,axis=2)
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = ensvalue
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: ANNUAL MEAN!')
    elif slicemonth == 'DJF':
        ensshape = np.empty((ensvalue.shape[0],ensvalue.shape[1]-1,
                             lat1.shape[0],lon1.shape[0]))
        for i in range(ensvalue.shape[0]):
            ensshape[i,:,:,:] = UT.calcDecJanFeb(ensvalue[i,:,:,:,:],
                                                 lat1,lon1,'surface',1)
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: DJF MEAN!')
    elif slicemonth == 'MAM':
        enstime = np.nanmean(ensvalue[:,:,2:5,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: MAM MEAN!')
    elif slicemonth == 'JJA':
        enstime = np.nanmean(ensvalue[:,:,5:8,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JJA MEAN!')
    elif slicemonth == 'SON':
        enstime = np.nanmean(ensvalue[:,:,8:11,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: SON MEAN!')
    elif slicemonth == 'JFM':
        enstime = np.nanmean(ensvalue[:,:,0:3,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JFM MEAN!')
    elif slicemonth == 'FMA':
        enstime = np.nanmean(ensvalue[:,:,1:4,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: FMA MEAN!')   
    elif slicemonth == 'MA':
        enstime = np.nanmean(ensvalue[:,:,2:4,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: MA MEAN!')  
    elif slicemonth == 'FM':
        enstime = np.nanmean(ensvalue[:,:,1:3,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: FM MEAN!')
    elif slicemonth == 'AMJ':
        enstime = np.nanmean(ensvalue[:,:,3:6,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: AMJ MEAN!')
    elif slicemonth == 'JAS':
        enstime = np.nanmean(ensvalue[:,:,6:9,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JAS MEAN!')
    elif slicemonth == 'OND':
        enstime = np.nanmean(ensvalue[:,:,9:,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: OND MEAN!')
    elif slicemonth == 'January':
        enstime = np.nanmean(ensvalue[:,:,0:1,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: January MEAN!')
    elif slicemonth == 'February':
        enstime = np.nanmean(ensvalue[:,:,1:2,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: February MEAN!')
    elif slicemonth == 'March':
        enstime = np.nanmean(ensvalue[:,:,2:3,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: March MEAN!')
    elif slicemonth == 'April':
        enstime = np.nanmean(ensvalue[:,:,3:4,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: April MEAN!')
    elif slicemonth == 'May':
        enstime = np.nanmean(ensvalue[:,:,4:5,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif any([sliceshape == 4,sliceshape == 5]):
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: May MEAN!')
    elif slicemonth == 'none':
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif sliceshape == 4:
            ensshape = np.reshape(ensvalue,(ensvalue.shape[0],ensvalue.shape[1]*ensvalue.shape[2],
                                             ensvalue.shape[3],ensvalue.shape[4]))
        elif any([sliceshape == 5,sliceshape == 6]):
            ensshape = ensvalue
        print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
        print('Completed: ALL RAVELED MONTHS!')

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
    if any([vari=='T2M',vari=='SST']):
        ensshape = ensshape - 273.15 # K to C
        print('Completed: Changed units (K to C)!')
    elif any([vari=='PRECL',vari=='PRECC',vari=='PRECT']):
        ensshape = ensshape * 8.64e7 # m/s to mm/day
        ### "Average Monthly Rate of Precipitation"
        print('*** CURRENT UNITS ---> [[ mm/day ]]! ***')
    elif any([vari=='SLP',vari=='PS']):
        if np.nanmean(ensshape) > 10000.:
            if np.nanmean(ensshape) < 0.:
                print(ValueError('SOMETHING IS WRONG WITH SLP!'))
                sys.exit()
            ensshape = ensshape/100. # Pa to hPa
            print('Completed: Changed units (Pa to hPa)!')
        else:
            ensshape = ensshape
    if scenario == 'spear':
        if any([vari=='SNOW']):
            ensshape = ensshape / 1000 # kg/m^2 to m
            ### "Average Monthly column-integrated snow water"
            print('*** CURRENT UNITS ---> [[ m ]]! ***')
        
    
    ###########################################################################
    ### Remove last year, because it is not complete in most AMIPS
    if scenario != 'spear':
        if ensshape.ndim == 4:
            ensshapeyr = ensshape[:,:-1,:,:]
            timeyr = time[:-1]
            print('Remove last year for AMIPs')
        elif ensshape.ndim == 5:
            ensshapeyr = ensshape[:,:-1,:,:,:]
            timeyr = time[:-1]
            print('Remove last year for AMIPs')
        elif ensshape.ndim == 6:
            ensshapeyr = ensshape[:,:-1,:,:,:,:]
            timeyr = time[:-1]
            print('Remove last year for AMIPs')
        else:
            print(ValueError('SOMETHING IS WRONG WITH THE DIMENSIONS!'))
            sys.exit()
    else:
        ensshapeyr = ensshape
        timeyr = time
    
    print('Shape of output FINAL = ', ensshapeyr.shape,[[ensshapeyr.ndim]])
    print('>>>>>>>>>> ENDING read_FACTS_Experi function!')    
    return lat1,lon1,lev1,ensshapeyr,timeyr

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# vari = 'U'
# slicemonth = 'none'
# sliceshape = 6
# slicenan = 'nan'
# timeper = 'all'
# scenario = 'amip_obs_rf'
# model = 'ESRL-CAM5'
# level = 'vertical'
# lat,lon,lev,var,years = read_FACTS_Experi(scenario,model,vari,slicemonth,sliceshape,slicenan,level)
# latss,lonss,levss,varss,yearsss = read_FACTS_Experi('spear','spear',vari,slicemonth,sliceshape,slicenan,level)

# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
# lon2ss,lat2ss = np.meshgrid(lonss,latss)
# avess = UT.calc_weightedAve(varss,lat2ss)
