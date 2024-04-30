"""
Plots of LRP for each OS classification

Author     : Zachary M. Labe
Date       : 23 October 2023
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
from sklearn.metrics import accuracy_score
import cmasher as cmr
import cmocean

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

### Segment data for training and testing data
fac = 0.80 # 0.70 training, 0.13 validation, 0.07 for testing
random_segment_seed = 71541
random_network_seed = 87750
reg_name = 'Globe'
numOfEns = 30
numOfEns_10ye = 30
years = np.arange(2015,2100+1,1)
def readDataOSLRP(variq,typeOfModel):
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Data preliminaries 
    directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'
    directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/'
    directorydata2 = '/work/Zachary.Labe/Research/DetectMitigate/Data/LRP/'
    letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
    ###############################################################################
    ###############################################################################
    modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_NATURAL','SPEAR_MED','SPEAR_MED_Scenario','SPEAR_MED_Scenario']
    scenarioall = ['historical','natural','SSP585','SSP119','SSP245']
    dataset_obs = 'ERA5_MEDS'
    monthlychoice = 'annual'

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
    predosALL = np.int_(np.genfromtxt(directorydata + 'overshootPredictedLabels_' + savename + '.txt'))[:,:]
    predos_10yeALL = np.int_(np.genfromtxt(directorydata + 'overshoot_10yePredictedLabels_' + savename + '.txt'))[:,:]
    
    ### Read in LRP maps
    lrp_osALL,lat1,lon1 = readData(directorydata2,'OS',savename)
    lrp_os_10yeALL,lat1,lon1 = readData(directorydata2,'OS_10ye',savename)
    lon2,lat2 = np.meshgrid(lon1,lat1)

    ###############################################################################
    ###############################################################################
    ############################################################################### 
    ### Find which model
    lrppredos = []
    predos_test_all = []
    for ens30 in range(numOfEns):
        predos = predosALL[ens30]
        lrp_os = lrp_osALL[ens30]
    
        predos_test = []
        for i in range(lenOfPicks):
            predosloc = np.where((predos == int(i)))[0]
            predos_test.append(predosloc)
        
        ### Composite lrp maps of the models
        predostest = []
        for i in range(lenOfPicks):
            lrppredosq = lrp_os[predos_test[i]]
            lrppredosmean = np.nanmean(lrppredosq,axis=0)
            if type(lrppredosmean) == float:
                lrppredosmean = np.full((lat1.shape[0],lon1.shape[0]),np.nan)
            predostest.append(lrppredosmean)
        
        lrppredosq = np.asarray(predostest ,dtype=object)
        lrppredos.append(lrppredosq)
        predos_test_all.append(predos_test)
    lrppredos_ensembles = np.asarray(lrppredos)
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    lrppredos_10ye = []
    predos_10ye_test_all = []
    for ens30 in range(numOfEns_10ye):
        predos_10ye = predos_10yeALL[ens30]
        lrp_os_10ye = lrp_os_10yeALL[ens30]
        
        predos_10ye_test = []
        for i in range(lenOfPicks):
            predos_10yeloc = np.where((predos_10ye == int(i)))[0]
            predos_10ye_test.append(predos_10yeloc)
        
        ### Composite lrp maps of the models
        predos_10yetest = []
        for i in range(lenOfPicks):
            lrppredos_10yeq = lrp_os_10ye[predos_10ye_test[i]]
            lrppredos_10yemean = np.nanmean(lrppredos_10yeq,axis=0)
            if type(lrppredos_10yemean) == float:
                lrppredos_10yemean = np.full((lat1.shape[0],lon1.shape[0]),np.nan)
            predos_10yetest.append(lrppredos_10yemean)

        lrppredosq_10ye = np.asarray(predos_10yetest,dtype=object)
        lrppredos_10ye.append(lrppredosq_10ye)
        predos_10ye_test_all.append(predos_10ye_test)
    lrppredos_10ye_ensembles = np.asarray(lrppredos_10ye)

    ########################################################################### 
    ### Count classes for os
    count_os = []
    for e in range(len(predos_test_all)):
        count_os_q = []
        for c in range(num_of_class):
            count_os_n = len(predos_test_all[e][c])
            count_os_q.append(count_os_n)
        count_os.append(count_os_q)

    count_os = np.asarray(count_os)
    count_os_sum = np.sum(count_os,axis=0)
    
    ########################################################################### 
    ### Count classes for os_10ye
    count_os_10ye = []
    for e in range(len(predos_10ye_test_all)):
        count_os_10yeq = []
        for c in range(num_of_class):
            count_os_10yen = len(predos_10ye_test_all[e][c])
            count_os_10yeq.append(count_os_10yen)
        count_os_10ye.append(count_os_10yeq)

    count_os_10ye = np.asarray(count_os_10ye)
    count_os_10yesum = np.sum(count_os_10ye,axis=0)
    
    ########################################################################### 
    ### Fix zeros when no classes are predicted
    if variq == 'T2M':
        wherezero_os = np.argwhere(count_os_sum  == 0)[0]
        wherezero_os_10ye = np.argwhere(count_os_10yesum  == 0)[0]
        if len(wherezero_os) > 0:
            lrppredos_ensembles[:,wherezero_os,:,:] = 9999
        if len(wherezero_os_10ye) > 0:
            lrppredos_10ye_ensembles[:,wherezero_os_10ye,:,:] = 9999
    
    ###########################################################################
    ###########################################################################
    ###########################################################################    
    ### Calculate ensemble means
    lrppredos = np.nanmean(lrppredos_ensembles,axis=0)
    lrppredos_max = np.nanmax(lrppredos_ensembles,axis=0)
    lrppredos_min = np.nanmin(lrppredos_ensembles,axis=0)
    lrppredos_range = lrppredos_max - lrppredos_min
    
    lrppredos_10ye = np.nanmean(lrppredos_10ye_ensembles,axis=0)
    lrppredos_10ye_max = np.nanmax(lrppredos_10ye_ensembles,axis=0)
    lrppredos_10ye_min = np.nanmin(lrppredos_10ye_ensembles,axis=0)
    lrppredos_10ye_range = lrppredos_10ye_max - lrppredos_10ye_min
    
    ###########################################################################
    ###########################################################################
    ########################################################################### 
    ### Replace missing class with nans
    lrppredos[np.where(lrppredos == 9999)] = np.nan
    lrppredos_10ye[np.where(lrppredos_10ye == 9999)] = np.nan
    
    ### Select which OS
    if typeOfModel == 'OS':
        lrpreturn = lrppredos
        count_sumpick = count_os_sum
    elif typeOfModel == 'OS10ye':
        lrpreturn = lrppredos_10ye
        count_sumpick = count_os_10yesum
    
    return lrpreturn,count_sumpick,lat1,lon1
    
### Read in data for LRP and scores
variq = 'PRECT'
lrp_os,count_os_sum,lat1,lon1 = readDataOSLRP(variq,'OS')
lrp_os10ye,count_os10ye_sum,lat1,lon1 = readDataOSLRP(variq,'OS10ye')

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of LRP means training
limit = np.arange(-0.3,0.31,0.005)
barlim = np.round(np.arange(-0.3,0.31,0.3),2)
    
### Other parameters
lenOfPicks = len(lrp_os)
lon2,lat2 = np.meshgrid(lon1,lat1)

fig = plt.figure(figsize=(10,3))
for r in range(lenOfPicks*2):
    
    if r < 5 :
        lrptest = lrp_os
        count_sum = count_os_sum[r]
        var = lrptest[r]/np.max(abs(lrptest[r]))
    else:
        lrptest = lrp_os10ye
        count_sum = count_os10ye_sum[r-5]
        var = lrptest[r-5]/np.max(abs(lrptest[r-5]))
        
    if variq == 'PRECT':
        variqname = 'Precipitation'
        label = r'\textbf{Relevance [N=%s Maps] -- %s}' % ((numOfEns*(years.shape[0])),variqname)
        cmap = cmocean.cm.tarn
        # cmap = cmr.waterlily
    elif variq == 'T2M':
        variqname = 'Temperature'
        label = r'\textbf{Relevance [N=%s Maps] -- %s}' % ((numOfEns*(years.shape[0])),variqname)
        cmap = cmr.prinsenvlag_r
        
    ax1 = plt.subplot(2,lenOfPicks,r+1)
    m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
 
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    x,y = m(lon2,lat2)
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    cs1.set_cmap(cmap) 
    
    if r < 5:
        plt.title(r'\textbf{%s}' % (scenarioallnames[r]),fontsize=20,color='dimgrey',y=1.1)

    if any([r==0]):
        ax1.annotate(r'\textbf{SSP534OS}',xy=(0,0),xytext=(-0.08,0.50),
                  textcoords='axes fraction',color='k',fontsize=8,
                  rotation=90,ha='center',va='center')
    elif any([r==5]):
        ax1.annotate(r'\textbf{SSP534OS_10ye}',xy=(0,0),xytext=(-0.08,0.50),
                  textcoords='axes fraction',color='k',fontsize=8,
                  rotation=90,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,1),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    ax1.annotate(r'\textbf{n=%s}' % count_sum,xy=(0,0),xytext=(0.04,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
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
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
    
    plt.savefig(directoryfigure + 'MSFigure_ANN_LRP_OS_%s.png' % variq ,dpi=600)
    
