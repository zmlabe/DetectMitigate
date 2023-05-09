"""
Third version of classification problem for emission scenario using the 
overshoot scenarios

Author     : Zachary M. Labe
Date       : 6 March 2023
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
numOfEns_10ye = 9
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
    lrpRule = 'Z'
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
        
        name = 'LRPMap_Z' + typemodel + '_' + savename + '.nc'
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
            lrppredos_10yeq = lrp_os[predos_10ye_test[i]]
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
    
    if len(np.where(count_os_sum  == 0)[0]) > 0:
        wherezero_os = np.argwhere(count_os_sum  == 0)[0]
    else:
        wherezero_os = []
    
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
    
    if len(np.where(count_os_10yesum  == 0)[0]) > 0:
        wherezero_os_10ye = np.argwhere(count_os_10yesum  == 0)[0]
    else:
        wherezero_os_10ye = []
    
    ########################################################################### 
    ### Fix zeros when no classes are predicted
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

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plot subplot of LRP ensemble means OS
    limit = np.arange(-0.4,0.41,0.005)
    barlim = np.round(np.arange(-0.4,0.41,0.2),2)
    if variq == 'T2M':
        cmap = cmr.fusion_r
    elif any([variq == 'PRECT',variq=='WA']):
        cmap = cmr.waterlily
    label = r'\textbf{Relevance}'
    
    fig = plt.figure(figsize=(10,2))
    for r in range(lenOfPicks):
        var = lrppredos[r]/np.max(abs(lrppredos[r]))
        
        ax1 = plt.subplot(1,lenOfPicks,r+1)
        m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
        m.drawcoastlines(color='darkgrey',linewidth=0.27)
            
        # var, lons_cyclic = addcyclic(var, lon1)
        # var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        # lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
        # x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='k',color='k',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        x,y = m(lon2,lat2)
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
    
    plt.savefig(directoryfigure + 'PRESENTATION_LRPComposites_%s_OS_ensemblemean.png' % (savename),dpi=300)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plot subplot of LRP means OS_10ye
    limit = np.arange(-0.4,0.41,0.005)
    barlim = np.round(np.arange(-0.4,0.41,0.2),2)
    if variq == 'T2M':
        cmap = cmr.fusion_r
    elif variq == 'PRECT':
        cmap = cmr.waterlily
    label = r'\textbf{Relevance}'
    
    fig = plt.figure(figsize=(10,2))
    for r in range(lenOfPicks):
        var = lrppredos_10ye[r]/np.max(abs(lrppredos_10ye[r]))
        
        ax1 = plt.subplot(1,lenOfPicks,r+1)
        m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
        m.drawcoastlines(color='darkgrey',linewidth=0.27)
            
        # var, lons_cyclic = addcyclic(var, lon1)
        # var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        # lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
        # x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='k',color='k',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        x,y = m(lon2,lat2)
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
    
    plt.savefig(directoryfigure + 'PRESENTATION_LRPComposites_%s_OS_10ye_ensemblemean.png' % (savename),dpi=300)

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
    ###############################################################################
    ###############################################################################
    ### Plot subplot of LRP ensemble range OS
    limit = np.arange(0,0.41,0.005)
    barlim = np.round(np.arange(0,0.41,0.2),2)
    cmap = cmr.dusk
    label = r'\textbf{Relevance}'
    
    fig = plt.figure(figsize=(10,2))
    for r in range(lenOfPicks):
        if count_os_sum[r] > 2:
            var = lrppredos_range[r]
        else:
            var = np.full(lrppredos_range[r].shape,np.nan)
        
        ax1 = plt.subplot(1,lenOfPicks,r+1)
        m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
        m.drawcoastlines(color='darkgrey',linewidth=0.27)
            
        # var, lons_cyclic = addcyclic(var, lon1)
        # var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        # lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
        # x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='k',color='k',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        x,y = m(lon2,lat2)
        cs1 = m.contourf(lon2,lat2,var,limit,extend='max',latlon=True)
        cs1.set_cmap(cmap) 
                
        ax1.annotate(r'\textbf{%s}' % (scenarioalln[r]),xy=(0,0),xytext=(0.5,1.10),
                      textcoords='axes fraction',color='w',fontsize=19,
                      rotation=0,ha='center',va='center')
        
    ###############################################################################
    cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
    cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
    cbar1.set_ticks(barlim)
    cbar1.set_ticklabels(list(map(str,barlim)))
    cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
    cbar1.outline.set_edgecolor('darkgrey')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
    
    plt.savefig(directoryfigure + 'PRESENTATION_LRPComposites_%s_OS_ensemblerange.png' % (savename),dpi=300)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plot subplot of LRP ensemble range OS_10ye
    limit = np.arange(0,0.41,0.005)
    barlim = np.round(np.arange(0,0.41,0.2),2)
    cmap = cmr.dusk
    label = r'\textbf{Relevance}'
    
    fig = plt.figure(figsize=(10,2))
    for r in range(lenOfPicks):
        
        if count_os_10yesum[r] > 2:
            var = lrppredos_10ye_range[r]
        else:
            var = np.full(lrppredos_10ye_range[r].shape,np.nan)
        
        ax1 = plt.subplot(1,lenOfPicks,r+1)
        m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
        m.drawcoastlines(color='darkgrey',linewidth=0.27)
            
        # var, lons_cyclic = addcyclic(var, lon1)
        # var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        # lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
        # x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='k',color='k',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        x,y = m(lon2,lat2)
        cs1 = m.contourf(lon2,lat2,var,limit,extend='max',latlon=True)
        cs1.set_cmap(cmap) 
                
        ax1.annotate(r'\textbf{%s}' % (scenarioalln[r]),xy=(0,0),xytext=(0.5,1.10),
                      textcoords='axes fraction',color='w',fontsize=19,
                      rotation=0,ha='center',va='center')
        
    ###############################################################################
    cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
    cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
    cbar1.set_ticks(barlim)
    cbar1.set_ticklabels(list(map(str,barlim)))
    cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
    cbar1.outline.set_edgecolor('dARKgrey')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
    
    plt.savefig(directoryfigure + 'PRESENTATION_LRPComposites_%s_OS_10ye_ensemblerange.png' % (savename),dpi=300)
