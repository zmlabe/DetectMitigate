"""
Evaluate all of the transitions using integrated gradients

Author     : Zachary M. Labe
Date       : 27 October 2023
Version    : 4
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
from sklearn.metrics import accuracy_score
import cmasher as cmr
import cmocean
from scipy.ndimage import gaussian_filter
import calc_Utilities as UT
import scipy.stats as sts
import scipy.stats.mstats as mstats

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Segment data for training and testing data
fac = 0.80 # 0.70 training, 0.13 validation, 0.07 for testing
random_segment_seed = 71541
random_network_seed = 87750

### Figure parameters
allvariables = ['Temperature','Temperature','Temperature','Precipitation','Precipitation','Precipitation']

fig = plt.figure(figsize=(10,4))
transitionsLength = 6
for pp in range(transitionsLength):
    if pp == 0:
        variablesall = ['T2M']
        numOfEns = 30
        numOfEns_10ye = 30
        pickSMILEall = [[]] 
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'
        directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/'
        directorydata2 = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/LRP/'
        letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
        ###############################################################################
        ###############################################################################
        modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
        experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
        scenarioall = ['SSP245','SSP585']
        dataset_obs = 'ERA5_MEDS'
        seasons = ['annual']
        monthlychoice = seasons[0]
        variq = variablesall[0]
        reg_name = 'Globe'
        ###############################################################################
        ###############################################################################
        lenOfPicks = len(scenarioall)
        ###############################################################################
        ###############################################################################
        land_only = False
        ocean_only = False
        ###############################################################################
        ###############################################################################
        rm_merid_mean = False
        rm_annual_mean = True
        ###############################################################################
        ###############################################################################
        rm_ensemble_mean = False
        rm_observational_mean = False
        ###############################################################################
        ###############################################################################
        calculate_anomalies = False
        if calculate_anomalies == True:
            baseline = np.arange(1951,1980+1,1)
        ###############################################################################
        ###############################################################################
        window = 0
        ensTypeExperi = 'ENS'
        ###############################################################################
        ###############################################################################
        if ensTypeExperi == 'ENS':
            if window == 0:
                rm_standard_dev = False
                ravel_modelens = False
                ravelmodeltime = False
            else:
                rm_standard_dev = True
                ravelmodeltime = False
                ravel_modelens = True
        yearsall = [np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1)]
        yearsobs = np.arange(1979+window,2021+1,1)
        lentime = len(yearsall)
        ###############################################################################
        ###############################################################################
        ravelyearsbinary = False
        ravelbinary = False
        num_of_class = lenOfPicks
        ###############################################################################
        ###############################################################################
        lrpRule = 'integratedgradient'
        normLRP = True
        
        ### Model paramaters
        if variq == 'T2M':
            hidden = [20]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.2
            actFun = 'relu'
        elif variq == 'PRECT':
            hidden = [100,100,100]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.05
            actFun = 'relu'
        else:
            print(ValueError('WRONG VARIABLE NOT TUNED YET FOR ANN!'))
            sys.exit()
            
        ### Check that transition is consistent
        print('\n',pp,'-',scenarioall,'--------')    
            
        ### Select how to save files
        savename = 'ANNv4_EmissionScenarioBinary_ssp_245-585_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
        print('*Filename == < %s >' % savename) 
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################       
        ### Read in data
        def readData(directory,typemodel,savename):
            """
            Read in LRP maps
            """
            
            name = 'LRPMap_IG' + typemodel + '_' + savename + '.nc'
            print(name)
            filename = directory + name
            data = Dataset(filename)
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            lrp = data.variables['LRP'][:,:,:,:]
            data.close()
            
            return lrp,lat,lon
        
        ### Read in training and testing predictions and labels
        predos = np.int_(np.genfromtxt(directorydata + 'overshootPredictedLabels_' + savename + '.txt'))[:,:]
        
        ### Read in LRP maps
        lrp,lat1,lon1 = readData(directorydata2,'OS',savename)
        lon2,lat2 = np.meshgrid(lon1,lat1)
        print(lrp.shape)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    elif pp == 1:
        variablesall = ['T2M']
        numOfEns = 30
        numOfEns_10ye = 30
        pickSMILEall = [[]] 
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'
        directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/'
        directorydata2 = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/LRP/'
        letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
        ###############################################################################
        ###############################################################################
        modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
        experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
        scenarioall = ['SSP245','SSP585']
        dataset_obs = 'ERA5_MEDS'
        seasons = ['annual']
        monthlychoice = seasons[0]
        variq = variablesall[0]
        reg_name = 'Globe'
        ###############################################################################
        ###############################################################################
        lenOfPicks = len(scenarioall)
        ###############################################################################
        ###############################################################################
        land_only = False
        ocean_only = False
        ###############################################################################
        ###############################################################################
        rm_merid_mean = False
        rm_annual_mean = True
        ###############################################################################
        ###############################################################################
        rm_ensemble_mean = False
        rm_observational_mean = False
        ###############################################################################
        ###############################################################################
        calculate_anomalies = False
        if calculate_anomalies == True:
            baseline = np.arange(1951,1980+1,1)
        ###############################################################################
        ###############################################################################
        window = 0
        ensTypeExperi = 'ENS'
        ###############################################################################
        ###############################################################################
        if ensTypeExperi == 'ENS':
            if window == 0:
                rm_standard_dev = False
                ravel_modelens = False
                ravelmodeltime = False
            else:
                rm_standard_dev = True
                ravelmodeltime = False
                ravel_modelens = True
        yearsall = [np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1)]
        yearsobs = np.arange(1979+window,2021+1,1)
        lentime = len(yearsall)
        ###############################################################################
        ###############################################################################
        ravelyearsbinary = False
        ravelbinary = False
        num_of_class = lenOfPicks
        ###############################################################################
        ###############################################################################
        lrpRule = 'integratedgradient'
        normLRP = True
        
        ### Model paramaters
        if variq == 'T2M':
            hidden = [20]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.2
            actFun = 'relu'
        elif variq == 'PRECT':
            hidden = [100,100,100]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.05
            actFun = 'relu'
        else:
            print(ValueError('WRONG VARIABLE NOT TUNED YET FOR ANN!'))
            sys.exit()
            
        ### Check that transition is consistent
        print('\n',pp,'-',scenarioall,'--------')    
            
        ### Select how to save files
        savename = 'ANNv4_EmissionScenarioBinary_ssp_245-585_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
        print('*Filename == < %s >' % savename) 
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################       
        ### Read in data
        def readData(directory,typemodel,savename):
            """
            Read in LRP maps
            """
            
            name = 'LRPMap_IG' + typemodel + '_' + savename + '.nc'
            print(name)
            filename = directory + name
            data = Dataset(filename)
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            lrp = data.variables['LRP'][:,:,:,:]
            data.close()
            
            return lrp,lat,lon
        
        ### Read in training and testing predictions and labels
        predos_10ye = np.int_(np.genfromtxt(directorydata + 'overshoot_10yePredictedLabels_' + savename + '.txt'))[:,:]
        
        ### Read in LRP maps
        lrp,lat1,lon1 = readData(directorydata2,'OS_10ye',savename)
        lon2,lat2 = np.meshgrid(lon1,lat1)
        print(lrp.shape)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    elif pp == 2:
        variablesall = ['T2M']
        numOfEns = 30
        numOfEns_10ye = 30
        pickSMILEall = [[]] 
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'
        directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/'
        directorydata2 = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/LRP/'
        letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
        ###############################################################################
        ###############################################################################
        modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
        experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
        scenarioall = ['SSP119','SSP245']
        dataset_obs = 'ERA5_MEDS'
        seasons = ['annual']
        monthlychoice = seasons[0]
        variq = variablesall[0]
        reg_name = 'Globe'
        ###############################################################################
        ###############################################################################
        lenOfPicks = len(scenarioall)
        ###############################################################################
        ###############################################################################
        land_only = False
        ocean_only = False
        ###############################################################################
        ###############################################################################
        rm_merid_mean = False
        rm_annual_mean = True
        ###############################################################################
        ###############################################################################
        rm_ensemble_mean = False
        rm_observational_mean = False
        ###############################################################################
        ###############################################################################
        calculate_anomalies = False
        if calculate_anomalies == True:
            baseline = np.arange(1951,1980+1,1)
        ###############################################################################
        ###############################################################################
        window = 0
        ensTypeExperi = 'ENS'
        ###############################################################################
        ###############################################################################
        if ensTypeExperi == 'ENS':
            if window == 0:
                rm_standard_dev = False
                ravel_modelens = False
                ravelmodeltime = False
            else:
                rm_standard_dev = True
                ravelmodeltime = False
                ravel_modelens = True
        yearsall = [np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1)]
        yearsobs = np.arange(1979+window,2021+1,1)
        lentime = len(yearsall)
        ###############################################################################
        ###############################################################################
        ravelyearsbinary = False
        ravelbinary = False
        num_of_class = lenOfPicks
        ###############################################################################
        ###############################################################################
        lrpRule = 'integratedgradient'
        normLRP = True
        
        ### Model paramaters
        if variq == 'T2M':
            hidden = [20,20]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.05
            actFun = 'relu'
        elif variq == 'PRECT':
            hidden = [100,100,100]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.05
            actFun = 'relu'
        else:
            print(ValueError('WRONG VARIABLE NOT TUNED YET FOR ANN!'))
            sys.exit()
            
        ### Check that transition is consistent
        print('\n',pp,'-',scenarioall,'--------')    
    
        ### Select how to save files
        savename = 'ANNv4_EmissionScenarioBinary_ssp_119-245_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
        print('*Filename == < %s >' % savename) 

        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################       
        ### Read in data
        def readData(directory,typemodel,savename):
            """
            Read in LRP maps
            """
            
            name = 'LRPMap_IG' + typemodel + '_' + savename + '.nc'
            print(name)
            filename = directory + name
            data = Dataset(filename)
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            lrp = data.variables['LRP'][:,:,:,:]
            data.close()
            
            return lrp,lat,lon
        
        ### Read in training and testing predictions and labels
        predos_10ye = np.int_(np.genfromtxt(directorydata + 'overshoot_10yePredictedLabels_' + savename + '.txt'))[:,:]
        
        ### Read in LRP maps
        lrp,lat1,lon1 = readData(directorydata2,'OS_10ye',savename)
        lon2,lat2 = np.meshgrid(lon1,lat1)
        print(lrp.shape)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    elif pp == 3:
        variablesall = ['PRECT']
        numOfEns = 30
        numOfEns_10ye = 30
        pickSMILEall = [[]] 
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'
        directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/'
        directorydata2 = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/LRP/'
        letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
        ###############################################################################
        ###############################################################################
        modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
        experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
        scenarioall = ['SSP245','SSP585']
        dataset_obs = 'ERA5_MEDS'
        seasons = ['annual']
        monthlychoice = seasons[0]
        variq = variablesall[0]
        reg_name = 'Globe'
        ###############################################################################
        ###############################################################################
        lenOfPicks = len(scenarioall)
        ###############################################################################
        ###############################################################################
        land_only = False
        ocean_only = False
        ###############################################################################
        ###############################################################################
        rm_merid_mean = False
        rm_annual_mean = True
        ###############################################################################
        ###############################################################################
        rm_ensemble_mean = False
        rm_observational_mean = False
        ###############################################################################
        ###############################################################################
        calculate_anomalies = False
        if calculate_anomalies == True:
            baseline = np.arange(1951,1980+1,1)
        ###############################################################################
        ###############################################################################
        window = 0
        ensTypeExperi = 'ENS'
        ###############################################################################
        ###############################################################################
        if ensTypeExperi == 'ENS':
            if window == 0:
                rm_standard_dev = False
                ravel_modelens = False
                ravelmodeltime = False
            else:
                rm_standard_dev = True
                ravelmodeltime = False
                ravel_modelens = True
        yearsall = [np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1)]
        yearsobs = np.arange(1979+window,2021+1,1)
        lentime = len(yearsall)
        ###############################################################################
        ###############################################################################
        ravelyearsbinary = False
        ravelbinary = False
        num_of_class = lenOfPicks
        ###############################################################################
        ###############################################################################
        lrpRule = 'integratedgradient'
        normLRP = True
        
        ### Model paramaters
        if variq == 'T2M':
            hidden = [20]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.2
            actFun = 'relu'
        elif variq == 'PRECT':
            hidden = [100,100,100]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.05
            actFun = 'relu'
        else:
            print(ValueError('WRONG VARIABLE NOT TUNED YET FOR ANN!'))
            sys.exit()
            
        ### Check that transition is consistent
        print('\n',pp,'-',scenarioall,'--------')    
        
        ### Select how to save files
        savename = 'ANNv4_EmissionScenarioBinary_ssp_245-585_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
        print('*Filename == < %s >' % savename) 
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################       
        ### Read in data
        def readData(directory,typemodel,savename):
            """
            Read in LRP maps
            """
            
            name = 'LRPMap_IG' + typemodel + '_' + savename + '.nc'
            print(name)
            filename = directory + name
            data = Dataset(filename)
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            lrp = data.variables['LRP'][:,:,:,:]
            data.close()
            
            return lrp,lat,lon
        
        ### Read in training and testing predictions and labels
        predos = np.int_(np.genfromtxt(directorydata + 'overshootPredictedLabels_' + savename + '.txt'))[:,:]
        
        ### Read in LRP maps
        lrp,lat1,lon1 = readData(directorydata2,'OS',savename)
        lon2,lat2 = np.meshgrid(lon1,lat1)
        print(lrp.shape)
     
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    elif pp == 4:
        variablesall = ['PRECT']
        numOfEns = 30
        numOfEns_10ye = 30
        pickSMILEall = [[]] 
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'
        directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/'
        directorydata2 = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/LRP/'
        letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
        ###############################################################################
        ###############################################################################
        modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
        experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
        scenarioall = ['SSP245','SSP585']
        dataset_obs = 'ERA5_MEDS'
        seasons = ['annual']
        monthlychoice = seasons[0]
        variq = variablesall[0]
        reg_name = 'Globe'
        ###############################################################################
        ###############################################################################
        lenOfPicks = len(scenarioall)
        ###############################################################################
        ###############################################################################
        land_only = False
        ocean_only = False
        ###############################################################################
        ###############################################################################
        rm_merid_mean = False
        rm_annual_mean = True
        ###############################################################################
        ###############################################################################
        rm_ensemble_mean = False
        rm_observational_mean = False
        ###############################################################################
        ###############################################################################
        calculate_anomalies = False
        if calculate_anomalies == True:
            baseline = np.arange(1951,1980+1,1)
        ###############################################################################
        ###############################################################################
        window = 0
        ensTypeExperi = 'ENS'
        ###############################################################################
        ###############################################################################
        if ensTypeExperi == 'ENS':
            if window == 0:
                rm_standard_dev = False
                ravel_modelens = False
                ravelmodeltime = False
            else:
                rm_standard_dev = True
                ravelmodeltime = False
                ravel_modelens = True
        yearsall = [np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1)]
        yearsobs = np.arange(1979+window,2021+1,1)
        lentime = len(yearsall)
        ###############################################################################
        ###############################################################################
        ravelyearsbinary = False
        ravelbinary = False
        num_of_class = lenOfPicks
        ###############################################################################
        ###############################################################################
        lrpRule = 'integratedgradient'
        normLRP = True
        
        ### Model paramaters
        if variq == 'T2M':
            hidden = [20]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.2
            actFun = 'relu'
        elif variq == 'PRECT':
            hidden = [100,100,100]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.05
            actFun = 'relu'
        else:
            print(ValueError('WRONG VARIABLE NOT TUNED YET FOR ANN!'))
            sys.exit()
            
        ### Check that transition is consistent
        print('\n',pp,'-',scenarioall,'--------')
            
        ### Select how to save files
        savename = 'ANNv4_EmissionScenarioBinary_ssp_245-585_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
        print('*Filename == < %s >' % savename) 
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################       
        ### Read in data
        def readData(directory,typemodel,savename):
            """
            Read in LRP maps
            """
            
            name = 'LRPMap_IG' + typemodel + '_' + savename + '.nc'
            print(name)
            filename = directory + name
            data = Dataset(filename)
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            lrp = data.variables['LRP'][:,:,:,:]
            data.close()
            
            return lrp,lat,lon
        
        ### Read in training and testing predictions and labels
        predos_10ye = np.int_(np.genfromtxt(directorydata + 'overshoot_10yePredictedLabels_' + savename + '.txt'))[:,:]
        
        ### Read in LRP maps
        lrp,lat1,lon1 = readData(directorydata2,'OS_10ye',savename)
        lon2,lat2 = np.meshgrid(lon1,lat1)
        print(lrp.shape)
        
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    elif pp == 5:
        variablesall = ['PRECT']
        numOfEns = 30
        numOfEns_10ye = 30
        pickSMILEall = [[]] 
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'
        directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/'
        directorydata2 = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/LRP/'
        letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
        ###############################################################################
        ###############################################################################
        modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
        experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
        scenarioall = ['SSP119','SSP245']
        dataset_obs = 'ERA5_MEDS'
        seasons = ['annual']
        monthlychoice = seasons[0]
        variq = variablesall[0]
        reg_name = 'Globe'
        ###############################################################################
        ###############################################################################
        lenOfPicks = len(scenarioall)
        ###############################################################################
        ###############################################################################
        land_only = False
        ocean_only = False
        ###############################################################################
        ###############################################################################
        rm_merid_mean = False
        rm_annual_mean = True
        ###############################################################################
        ###############################################################################
        rm_ensemble_mean = False
        rm_observational_mean = False
        ###############################################################################
        ###############################################################################
        calculate_anomalies = False
        if calculate_anomalies == True:
            baseline = np.arange(1951,1980+1,1)
        ###############################################################################
        ###############################################################################
        window = 0
        ensTypeExperi = 'ENS'
        ###############################################################################
        ###############################################################################
        if ensTypeExperi == 'ENS':
            if window == 0:
                rm_standard_dev = False
                ravel_modelens = False
                ravelmodeltime = False
            else:
                rm_standard_dev = True
                ravelmodeltime = False
                ravel_modelens = True
        yearsall = [np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1)]
        yearsobs = np.arange(1979+window,2021+1,1)
        lentime = len(yearsall)
        ###############################################################################
        ###############################################################################
        ravelyearsbinary = False
        ravelbinary = False
        num_of_class = lenOfPicks
        ###############################################################################
        ###############################################################################
        lrpRule = 'integratedgradient'
        normLRP = True
        
        ### Model paramaters
        if variq == 'T2M':
            hidden = [20,20]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.05
            actFun = 'relu'
        elif variq == 'PRECT':
            hidden = [100,100,100]
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = 0.05
            actFun = 'relu'
        else:
            print(ValueError('WRONG VARIABLE NOT TUNED YET FOR ANN!'))
            sys.exit()
        
        ### Check that transition is consistent
        print('\n',pp,'-',scenarioall,'--------')
            
        ### Select how to save files
        savename = 'ANNv4_EmissionScenarioBinary_ssp_119-245_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
        print('*Filename == < %s >' % savename) 

        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################       
        ### Read in data
        def readData(directory,typemodel,savename):
            """
            Read in LRP maps
            """
            
            name = 'LRPMap_IG' + typemodel + '_' + savename + '.nc'
            print(name)
            filename = directory + name
            data = Dataset(filename)
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            lrp = data.variables['LRP'][:,:,:,:]
            data.close()
            
            return lrp,lat,lon
        
        ### Read in training and testing predictions and labels
        predos_10ye = np.int_(np.genfromtxt(directorydata + 'overshoot_10yePredictedLabels_' + savename + '.txt'))[:,:]
        
        ### Read in LRP maps
        lrp,lat1,lon1 = readData(directorydata2,'OS_10ye',savename)
        lon2,lat2 = np.meshgrid(lon1,lat1)
        print(lrp.shape)
        
    ### Take ensemble mean of LRP fields
    lrpmean = np.nanmean(lrp[:,:,:,:],axis=0)
    
    ### Slice periods and take difference (+- 5 years)
    years = yearsall[0]
    os_years = yearsall[0]
    os_10ye_years = yearsall[0]
    sliceperiod = 5
    
    if pp == 0:
        yearq_1 = np.where((years == 2052))[0][0]
        yearq_2 = np.where((years == 2063))[0][0]
        
        lrp_1 = lrpmean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = lrpmean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################
    elif pp == 1:
        yearq_1 = np.where((years == 2048))[0][0]
        yearq_2 = np.where((years == 2056))[0][0]
        
        lrp_1 = lrpmean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = lrpmean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################        
    elif pp == 2:
        yearq_1 = np.where((years == 2073))[0][0]
        yearq_2 = np.where((years == 2084))[0][0]
        
        lrp_1 = lrpmean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = lrpmean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################       
    elif pp == 3:
        yearq_1 = np.where((years == 2049))[0][0]
        yearq_2 = np.where((years == 2064))[0][0]
        
        lrp_1 = lrpmean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = lrpmean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################        
    elif pp == 4:
        yearq_1 = np.where((years == 2045))[0][0]
        yearq_2 = np.where((years == 2051))[0][0]
        
        lrp_1 = lrpmean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = lrpmean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################        
    elif pp == 5:
        yearq_1 = np.where((years == 2072))[0][0]
        yearq_2 = np.where((years == 2086))[0][0]
        
        lrp_1 = lrpmean[yearq_1-(sliceperiod-1):yearq_1+1,:,:]
        lrp_2 = lrpmean[yearq_2:yearq_2+sliceperiod,:,:]
        
        lrp_diff = lrp_2 - lrp_1
        lrp_diffmean = np.nanmean(lrp_diff,axis=0)
###############################################################################        

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plot subplot of LRP means OS
    limit = np.arange(-0.1,0.101,0.001)
    barlim = np.round(np.arange(-0.1,0.11,0.1),2)
    cmap = cmocean.cm.balance
    label = r'\textbf{Relevance}'
        
    ### Retrieve variable
    var = lrp_diffmean
    
    ax1 = plt.subplot(2,3,pp+1)
    m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
    m.drawcoastlines(color='dimgrey',linewidth=0.35)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    x,y = m(lon2,lat2)
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    cs1.set_cmap(cmap) 
    
    if any([pp==0,pp==1]):
        plt.title(r'\textbf{SSP5-8.5 to SSP2-4.5}',fontsize=12,color='k')
    elif any([pp==2]):
        plt.title(r'\textbf{SSP2-4.5 to SSP1-1.9}',fontsize=12,color='k')

    if any([pp==0,pp==3]):
        ax1.annotate(r'\textbf{%s}' % (allvariables[pp]),xy=(0,0),xytext=(-0.05,0.50),
                  textcoords='axes fraction',color='k',fontsize=15,
                  rotation=90,ha='center',va='center')
            
    ax1.annotate(r'\textbf{[%s]}' % letters[pp],xy=(0,0),xytext=(0.86,0.98),
                  textcoords='axes fraction',color='k',fontsize=10,
                  rotation=330,ha='center',va='center')
   
ax1.annotate(r'\textbf{SPEAR_MED_SSP534OS}',xy=(0,0),xytext=(-1.55,2.5),
              textcoords='axes fraction',color='dimgrey',fontsize=15,
              rotation=0,ha='center',va='center')
ax1.annotate(r'\textbf{SPEAR_MED_SSP534OS_10ye}',xy=(0,0),xytext=(-0.07,2.5),
              textcoords='axes fraction',color='dimgrey',fontsize=15,
              rotation=0,ha='center',va='center')
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.09,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='k',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

# plt.tight_layout()
plt.subplots_adjust(top=0.89,wspace=0.02,hspace=0.02)

plt.savefig(directoryfigure + 'MSFigure_ANN_LRP_Transitions.png',dpi=600)
