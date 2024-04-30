"""
Third version of classification problem for emission scenarios

Author     : Zachary M. Labe
Date       : 9 March 2023
Version    : 2 - added XAI capability, 3 - added capability for all XAI of OS ensemble members
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

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

### Model paramaters
fac = 0.80 # 0.70 training, 0.13 validation, 0.07 for testing
random_segment_seed = 71541
random_network_seed = 87750
hidden = [30,30,30]
n_epochs = 1500
batch_size = 128
lr_here = 0.0001
ridgePenalty = 0.1
actFun = 'relu'

variablesall = ['WA']
numOfEns = 30
pickSMILEall = [[]] 
for va in range(len(variablesall)):
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Data preliminaries 
    directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Dark_Figures/'
    directorydata = '/home/Zachary.Labe/Research/DetectMitigate/Data/'
    directorydata2 = '/home/Zachary.Labe/Research/DetectMitigate/Data/LRP/'
    letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
    ###############################################################################
    ###############################################################################
    modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_NATURAL','SPEAR_MED','SPEAR_MED_Scenario','SPEAR_MED_Scenario']
    scenarioall = ['historical','natural','SSP585','SSP119','SSP245']
    scenarioalln = ['HISTORICAL','NATURAL','SSP5-8.5','SSP1-1.9','SSP2-4.5']
    dataset_obs = 'ERA5_MEDS'
    seasons = ['annual']
    monthlychoice = seasons[0]
    variq = variablesall[va]
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
    yearsall = [np.arange(1929+window,2014+1,1),np.arange(2015+window,2100+1,1),
                np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1),
                np.arange(2015+window,2100+1,1)]
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
        
    ### Select how to save files
    savename = 'ANNv3_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
    print('*Filename == < %s >' % savename) 
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Create sample class labels for each model for my own testing
    ### Appends a twin set of classes for the random noise class 
    classesl = np.empty((lenOfPicks,numOfEns,len(yearsall[0])))
    for i in range(lenOfPicks):
        classesl[i,:,:] = np.full((numOfEns,len(yearsall[0])),i)  
        
    if ensTypeExperi == 'ENS':
        classeslnew = np.swapaxes(classesl,0,1)
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
        filename = directory + name
        data = Dataset(filename)
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
        lrp = data.variables['LRP'][:]
        data.close()
        
        return lrp,lat,lon
    
    ### Read in training and testing predictions and labels
    classesltrain = np.int_(np.genfromtxt(directorydata + 'trainingTrueLabels_' + savename + '.txt'))
    classesltest = np.int_(np.genfromtxt(directorydata + 'testingTrueLabels_' + savename + '.txt'))
    
    predtrain = np.int_(np.genfromtxt(directorydata + 'trainingPredictedLabels_' + savename + '.txt'))
    predtest = np.int_(np.genfromtxt(directorydata + 'testingPredictedLabels_' + savename + '.txt'))
    
    ### Count testing data
    uniquetest,counttest = np.unique(predtest,return_counts=True)
    
    ### Read in LRP maps
    lrptestdata,lat1,lon1 = readData(directorydata2,'Testing',savename)
    lon2,lat2 = np.meshgrid(lon1,lat1)

    ###############################################################################
    ###############################################################################
    ############################################################################### 
    ### Find which model
    print('\nPrinting *length* of predicted labels for training and testing!')
    model_test = []
    for i in range(lenOfPicks):
        modelloc = np.where((predtest == int(i)))[0]
        print(len(modelloc))
        model_test.append(modelloc)
    
    lrptest = []
    for i in range(lenOfPicks):
        lrpmodel = lrptestdata[model_test[i]]
        lrpmodelmean = np.nanmean(lrpmodel,axis=0)
        lrptest.append(lrpmodelmean)
    lrptest = np.asarray(lrptest,dtype=object)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Calculate Accuracy 
    def accuracyTotalTime(data_pred,data_true):
        """
        Compute accuracy for the entire time series
        """
        
        data_truer = data_true
        data_predr = data_pred
        accdata_pred = accuracy_score(data_truer,data_predr)*100 # 0-100%
            
        return accdata_pred
    
    acctrain = accuracyTotalTime(predtrain,classesltrain)
    acctest = accuracyTotalTime(predtest,classesltest)
    print('\n\nAccuracy Training == %s%%' % acctrain)
    print('Accuracy Testing == %s%%' % acctest)

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plot subplot of LRP means training
    limit = np.arange(-0.4,0.41,0.005)
    barlim = np.round(np.arange(-0.4,0.41,0.2),2)
    if variq == 'T2M':
        cmap = cmr.fusion_r
    elif any([variq == 'PRECT',variq=='WA']):
        cmap = cmr.waterlily
    label = r'\textbf{Relevance}'
    
    fig = plt.figure(figsize=(10,2))
    for r in range(lenOfPicks):
        var = lrptest[r]/np.max(abs(lrptest[r]))
        
        ax1 = plt.subplot(1,lenOfPicks,r+1)
        m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
        m.drawcoastlines(color='dimgrey',linewidth=0.27)
            
        # var, lons_cyclic = addcyclic(var, lon1)
        # var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        # lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
        # x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='dimgrey',color='darkgrey',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
        cs1.set_cmap(cmap) 
                
        ax1.annotate(r'\textbf{%s}' % (scenarioalln[r]),xy=(0,0),xytext=(0.5,1.10),
                      textcoords='axes fraction',color='w',fontsize=19,
                      rotation=0,ha='center',va='center')
        
    ###############################################################################
    cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
    cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                        extend='both',extendfrac=0.07,drawedges=False)
    cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
    cbar1.set_ticks(barlim)
    cbar1.set_ticklabels(list(map(str,barlim)))
    cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
    cbar1.outline.set_edgecolor('darkgrey')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
     
    # plt.text(-11,0.00,r'\textbf{TRAINING ACCURACY = %s \%%}' % np.round(acctrain,1),color='k',
    #       fontsize=7)
    # plt.text(-11,-1,r'\textbf{TESTING ACCURACY = %s \%%}' % np.round(acctest,1),color='k',
    #           fontsize=7)
    
    plt.savefig(directoryfigure + 'PRESENTATION_LRPComposites_%s.png' % (savename),dpi=300)
