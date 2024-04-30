"""
Fourth version of classification problem for emission scenarios for LRP

Author     : Zachary M. Labe
Date       : 26 October 2023
Version    : 2 - added XAI capability, 3 - added capability for all XAI of OS ensemble members, 4 - selected architecture based on hyperparameter sweep
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support,plot_confusion_matrix,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
import cmasher as cmr
import cmocean
from scipy.ndimage import gaussian_filter

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'
directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/'
directorydata2 = '/work/Zachary.Labe/Research/DetectMitigate/Data/LRP/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
scenarioallnames = ['Historical','Natural','SSP5-8.5','SSP1-1.9','SSP2-4.5',
                    'Historical','Natural','SSP5-8.5','SSP1-1.9','SSP2-4.5']
allvariables = ['Temperature','Temperature','Temperature','Temperature','Temperature','Precipitation','Precipitation','Precipitation','Precipitation','Precipitation']

### Model paramaters
fac = 0.80 # 0.70 training, 0.13 validation, 0.07 for testing
random_segment_seed = 71541
random_network_seed = 87750
variablesall = ['T2M']
variq = variablesall[0]

### Smooth LRP fields
sigmafactor = 1.5

### Model paramaters
if variq == 'T2M':
    hidden = [100]
    n_epochs = 1500
    batch_size = 128
    lr_here = 0.0001
    ridgePenalty = 0.1
    actFun = 'relu'
elif variq == 'PRECT':
    hidden = [100]
    n_epochs = 1500
    batch_size = 128
    lr_here = 0.0001
    ridgePenalty = 0.1
    actFun = 'relu'
else:
    print(ValueError('WRONG VARIABLE NOT TUNED YET FOR ANN!'))
    sys.exit()
    
def readLRPVariables(variq):
    ###############################################################################
    ###############################################################################
    modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_NATURAL','SPEAR_MED','SPEAR_MED_Scenario','SPEAR_MED_Scenario']
    scenarioall = ['historical','natural','SSP585','SSP119','SSP245']
    dataset_obs = 'ERA5_MEDS'
    seasons = ['annual']
    reg_name = 'Globe'
    numOfEns = 30
    monthlychoice = seasons[0]
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
    savename = 'ANNv4_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
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
    ### Calculate accuracy statistics
    def accuracyTotalTime(data_pred,data_true):
        """
        Compute accuracy for the entire time series
        """  
        data_truer = data_true
        data_predr = data_pred
        accdata_pred = accuracy_score(data_truer,data_predr)
            
        return accdata_pred
    def precisionTotalTime(data_pred,data_true):
        """
        Compute precision for the entire time series
        """
        data_truer = data_true
        data_predr = data_pred
        precdata_pred = precision_score(data_truer,data_predr,average=None)
        
        return precdata_pred
    def recallTotalTime(data_pred,data_true):
        """
        Compute recall for the entire time series
        """
        data_truer = data_true
        data_predr = data_pred
        recalldata_pred = recall_score(data_truer,data_predr,average=None)
        
        return recalldata_pred
    def f1TotalTime(data_pred,data_true):
        """
        Compute f1 for the entire time series
        """
        data_truer = data_true
        data_predr = data_pred
        f1data_pred = f1_score(data_truer,data_predr,average=None)
        
        return f1data_pred
    
    acctrain = accuracyTotalTime(predtrain,classesltrain)
    acctest = accuracyTotalTime(predtest,classesltest)
    
    prectrain = precisionTotalTime(predtrain,classesltrain)     
    prectest = precisionTotalTime(predtest,classesltest)
    
    recalltrain = recallTotalTime(predtrain,classesltrain)     
    recalltest = recallTotalTime(predtest,classesltest)
    
    f1_train = f1TotalTime(predtrain,classesltrain)     
    f1_test = f1TotalTime(predtest,classesltest)
    print('\n\nAccuracy Training == %s%%' % acctrain)
    print('Accuracy Testing == %s%%' % acctest)

    return acctest,prectest,recalltest,f1_test,lrptest,lat1,lon1

### Read in data for LRP and scores
acctest_t2m,prectest_t2m,recalltest_t2m,f1_test_t2m,lrptest_t2m,lat1,lon1 = readLRPVariables('T2M')
acctest_precip,prectest_precip,recalltest_precip,f1_test_precip,lrptest_precip,lat1,lon1 = readLRPVariables('PRECT')

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of LRP means training
limit = np.arange(-0.3,0.31,0.005)
barlim = np.round(np.arange(-0.3,0.31,0.3),2)
    
### Other parameters
lenOfPicks = len(lrptest_t2m)
lon2,lat2 = np.meshgrid(lon1,lat1)

fig = plt.figure(figsize=(10,3))
for r in range(lenOfPicks*2):
    
    if r < 5 :
        lrptest = lrptest_t2m[r]
        acc = acctest_t2m
        rec = recalltest_t2m[r]
        pre = prectest_t2m[r]
        f1 = f1_test_t2m[r]
        cmap = cmr.prinsenvlag_r
        variqname = 'Temperature'
        label = r'\textbf{Relevance}'
    else:
        lrptest = lrptest_precip[r-5]
        acc = acctest_precip
        rec = recalltest_precip[r-5]
        pre = prectest_precip[r-5]
        f1 = f1_test_precip[r-5]
        # cmap = cmr.waterlily
        cmap = cmocean.cm.tarn
        variqname = 'Precipitation'
        label = r'\textbf{Relevance}'
        
    ### Scale maps
    var = lrptest/np.max(abs(lrptest))
    # var= gaussian_filter(lrptest.astype('float32'),sigma=sigmafactor,order=0)
    
    ax1 = plt.subplot(2,lenOfPicks,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    cs1.set_cmap(cmap) 
            
    # ax1.annotate(r'\textbf{%s, [%s/%s]}' % (scenarioall[r],len(model_test[r]),predtest.shape[0]//num_of_class),xy=(0,0),xytext=(0.5,1.10),
    #               textcoords='axes fraction',color='dimgrey',fontsize=8,
    #               rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='dimgrey',fontsize=8,
                  rotation=330,ha='center',va='center')
    
    ax1.annotate(r'\textbf{Precision=%s, Recall=%s, F1=%s}' % (np.round(pre,2),np.round(rec,2),np.round(f1,2)),xy=(0,0),xytext=(0.5,-0.1),
                  textcoords='axes fraction',color='k',fontsize=6.5,
                  rotation=0,ha='center',va='center')
    
    if any([r==0,r==5]):
        ax1.annotate(r'\textbf{%s [%s]}' % (allvariables[r],np.round(acc,2)),xy=(0,0),xytext=(-0.08,0.50),
                  textcoords='axes fraction',color='k',fontsize=8,
                  rotation=90,ha='center',va='center')
        
    if r < 5:
        plt.title(r'\textbf{%s}' % scenarioallnames[r],fontsize=20,color='dimgrey',y=1.2)
        
    if r == 4:
        cbar_ax1 = fig.add_axes([0.94,0.557,0.01,0.25])                 
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='vertical',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=10,color='k',labelpad=4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='y', size=.01,labelsize=7)
        cbar1.outline.set_edgecolor('dimgrey')
    elif r == 9:
        cbar_ax1 = fig.add_axes([0.94,0.163,0.01,0.25])                 
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='vertical',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=10,color='k',labelpad=5.5)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='y', size=.01,labelsize=7)
        cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.08,bottom=0.09,right=0.93)

plt.savefig(directoryfigure + 'MSFigure_ANN_LRP_SSPs.png',dpi=600)
