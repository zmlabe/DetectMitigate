"""
Second version of classification problem for emission scenario, but now for 
looping hyperparameters

Author     : Zachary M. Labe
Date       : 2 June 2023
Version    : 1 binary for ssp585 or ssp245, 2 - added capability for all XAI of OS ensemble members
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_SegmentData as FRAC
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Activation
import calc_Stats as dSS
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support,plot_confusion_matrix,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
import calc_LRPclass_Detect_PosNeg as LRPf

###############################################################################
###############################################################################
###############################################################################
### For XAI
import innvestigate
tf.compat.v1.disable_eager_execution() # bug fix with innvestigate v2.0.1 - tf2

###############################################################################
###############################################################################
###############################################################################
### To remove some warnings I don't care about
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Remove GPU error message
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)

###############################################################################
###############################################################################
###############################################################################
### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Directories
directoryoutputo = '/work/Zachary.Labe/Research/DetectMitigate/Data/LoopHyperparametersBinary_ssp_245-585/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED']
dataset_obs = 'ERA5_MEDS'
lenOfPicks = len(modelGCMs)
allDataLabels = modelGCMs
monthlychoice = 'annual'
variq = 'T2M'
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
timeper = ['futureforcing','futureforcing']
scenarioall = ['SSP245','SSP585']
num_of_class = len(scenarioall)
shuffletype = 'GAUSS'
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
baseline = np.arange(1951,1980+1,1)
###############################################################################
###############################################################################
window = 0
ensTypeExperi = 'ENS'
shuffletype = 'RANDGAUSS'
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

if variq == 'T2M':
    yearsobs = np.arange(1950+window,2021+1,1)
else:
    yearsobs = np.arange(1979+window,2021+1,1)
lentime = len(yearsall)
###############################################################################
###############################################################################
numOfEns = 30
numOfEns_10ye = 9
dataset_inference = True
###############################################################################
###############################################################################
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model and observational/reanalysis data
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons 
##############################################################################
##############################################################################
##############################################################################
def netcdfLRPz(lats,lons,var,directory,typemodel,savename):
    print('\n>>> Using netcdfLRP function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'LRPMap_Z' + typemodel + '_' + savename + '.nc'
    filename = directory + name
    ncfile = Dataset(filename,'w',format='NETCDF4')
    ncfile.description = 'LRP maps for using selected seed' 
    
    ### Dimensions
    ncfile.createDimension('years',var.shape[0])
    ncfile.createDimension('lat',var.shape[1])
    ncfile.createDimension('lon',var.shape[2])
    
    ### Variables
    years = ncfile.createVariable('years','f4',('years'))
    latitude = ncfile.createVariable('lat','f4',('lat'))
    longitude = ncfile.createVariable('lon','f4',('lon'))
    varns = ncfile.createVariable('LRP','f4',('years','lat','lon'))
    
    ### Units
    varns.units = 'unitless relevance'
    ncfile.title = 'LRP relevance'
    ncfile.instituion = 'Colorado State University'
    ncfile.references = 'Barnes et al. [2020]'
    
    ### Data
    years[:] = np.arange(var.shape[0])
    latitude[:] = lats
    longitude[:] = lons
    varns[:] = var
    
    ncfile.close()
    print('*Completed: Created netCDF4 File!')
##############################################################################
##############################################################################
##############################################################################
### Overshoot files
def netcdfLRPz_OS(lats,lons,var,directory,typemodel,savename):
    print('\n>>> Using netcdfLRP function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'LRPMap_Z' + typemodel + '_' + savename + '.nc'
    filename = directory + name
    ncfile = Dataset(filename,'w',format='NETCDF4')
    ncfile.description = 'LRP maps for using selected seed of overshoot runs' 
    
    ### Dimensions
    ncfile.createDimension('years',var.shape[0])
    ncfile.createDimension('lat',var.shape[1])
    ncfile.createDimension('lon',var.shape[2])
    
    ### Variables
    years = ncfile.createVariable('years','f4',('years'))
    latitude = ncfile.createVariable('lat','f4',('lat'))
    longitude = ncfile.createVariable('lon','f4',('lon'))
    varns = ncfile.createVariable('LRP','f4',('years','lat','lon'))
    
    ### Units
    varns.units = 'unitless relevance'
    ncfile.title = 'LRP relevance'
    ncfile.instituion = 'Colorado State University'
    ncfile.references = 'Barnes et al. [2020]'
    
    ### Data
    years[:] = np.arange(var.shape[0])
    latitude[:] = lats
    longitude[:] = lons
    varns[:] = var
    
    ncfile.close()
    print('*Completed: Created netCDF4 File!')
    
###############################################################################
###############################################################################
###############################################################################    
COUNTER = 5

###############################################################################
### Create hyperparameter list
# hiddenalltry = [[5],[20],[30],[100],[5,5],[20,20],[30,30],[100,100],[5,5,5],[20,20,20],[30,30,30],[100,100,100],[5,5,5,5],[20,20,20,20],[30,30,30,30],[100,100,100,100]]
hiddenalltry = [[30,30],[100,100],[5,5,5],[20,20,20],[30,30,30],[100,100,100],[5,5,5,5],[20,20,20,20],[30,30,30,30],[100,100,100,100]]
#######
#######
# hhtt = 12
#######
#######
for hh in range(len(hiddenalltry)):
    version = hh
    hiddenall = hiddenalltry[version]
    hiddenallstr = str(len(hiddenalltry[version])) + 'x' + str(hiddenalltry[version][0])
    ridgePenaltyall = [0.001,0.01,0.05,0.1,0.2,0.5,1,5]
    random_network_seedall = []
    random_segment_seedall = []
    savenameall = []
    
    ### Create folder for hiddens
    hiddenfolder = 'hidden_' + hiddenallstr + '/'
    directoryoutput = directoryoutputo + hiddenfolder
    isExist = os.path.exists(directoryoutput)
    if not isExist:
       os.makedirs(directoryoutput)
       
    ### Create folder for hiddens and scores
    hiddenfolder = 'hidden_' + hiddenallstr + '/'
    directoryoutputs = directoryoutputo + hiddenfolder + 'Scores/'
    isExist = os.path.exists(directoryoutputs)
    if not isExist:
       os.makedirs(directoryoutputs)
       
    ### Create folder for hiddens and overshoot predictions
    hiddenfolder = 'hidden_' + hiddenallstr + '/'
    directoryoutputover = directoryoutputo + hiddenfolder + 'OvershootPred/'
    isExist = os.path.exists(directoryoutputover)
    if not isExist:
       os.makedirs(directoryoutputover)
       
    ### Create folder for hiddens and LRP
    hiddenfolder = 'hidden_' + hiddenallstr + '/'
    directoryoutputl = directoryoutputo + hiddenfolder + 'LRP/'
    isExist = os.path.exists(directoryoutputl)
    if not isExist:
       os.makedirs(directoryoutputl)
    
    ### Begin loop
    for rp in range(len(ridgePenaltyall)):
        for seee in range(COUNTER):    
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
            data_all = []
            for no in range(len(modelGCMs)):
                dataset = modelGCMs[no]
                scenario = scenarioall[no]
                data_allq,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
                data_all.append(data_allq)
            data = np.asarray(data_all)
            
            ### Fix missing values for snow depth
            if any([variq=='SNOW']):
                data[np.where(data < 0.)] = 0.
                print('\n\n\n\n\n\n--THE VARIABLE IS SNOW NEGATIVE SNOW DEPTH!--\n\n\n\n\n\n')
            
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Segment data for training and testing data
            fac = 0.80 
            random_segment_seed = None
            random_network_seed = None
            Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtrain_shape,Xtest_shape,Xval_shape,testIndices,trainIndices,valIndices,class_weight = FRAC.segment_data(data,classesl,ensTypeExperi,fac,random_segment_seed)
            
            ### Model paramaters
            hidden = hiddenall
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = ridgePenaltyall[rp]
            actFun = 'relu'
            input_shape=np.shape(Xtrain)[1]
            output_shape=np.shape(Ytrain)[1]
            
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### XAI methods to try
            lrpRule1 = 'z'
            lrpRule2 = 'epsilon'
            lrpRule3 = 'integratedgradient'
            normLRP = True
            numDim = 3
            biasBool = False
            annType = 'class'
            
            ###############################################################################
            ###############################################################################
            ###############################################################################
            def loadmodel(Xtrain,Xval,Ytrain,Yval,hidden,random_network_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,class_weight,input_shape,output_shape):
                print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')
                keras.backend.clear_session()
                model = keras.models.Sequential()
                tf.random.set_seed(0)
            
                ### Input layer
                model.add(Dense(hidden[0],input_shape=(input_shape,),
                                activation=actFun,use_bias=True,
                                kernel_regularizer=keras.regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                                bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                
                ### Whether to add dropout after first layer
                model.add(layers.Dropout(rate=0.4,seed=random_network_seed)) 
            
                ### Initialize other layers
                for layer in hidden[1:]:
                    model.add(Dense(layer,activation=actFun,
                                    use_bias=True,
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.00,l2=0.00),
                                    bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                     
                    # model.add(layers.Dropout(rate=0.125,seed=random_network_seed)) 
                    print('\nTHIS IS AN ANN!\n')
            
                #### Initialize output layer
                model.add(Dense(output_shape,activation=None,use_bias=True,
                                kernel_regularizer=keras.regularizers.l1_l2(l1=0.00, l2=0.00),
                                bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
            
                ### Add softmax layer at the end
                model.add(Activation('softmax'))
                
                ### Compile the model
                model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_here,
                              momentum=0.9,nesterov=True),  
                              loss = 'categorical_crossentropy',
                              metrics=[keras.metrics.categorical_accuracy]) 
                
                ### Callbacks
                early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                patience=10,
                                                                verbose=1,
                                                                mode='auto',
                                                                restore_best_weights=True)
                
                ### Model fit
                history = model.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=n_epochs,
                                    shuffle=True,verbose=1,class_weight=class_weight,
                                    callbacks=[early_stopping],
                                    validation_data=(Xval,Yval))
                
                model.summary() 
                return model,history
            
            ###############################################################################
            ### Standardize data
            ### If the variable is snow
            if any([variq=='SNOW',variq=='SNOWRATE']):
                Xtrain[np.where(Xtrain == 0.)] = np.nan
                Xtest[np.where(Xtest == 0.)] = np.nan
                Xval[np.where(Xval == 0.)] = np.nan
                print('\n\n\n\n\n\n--THE VARIABLE IS SNOW WITH ZEROS OVER WARMER AREAS!--\n\n\n\n\n\n')
            XtrainS,XtestS,XvalS,stdVals = dSS.standardize_dataVal(Xtrain,Xtest,Xval)
            Xmean, Xstd = stdVals  
            
            ###############################################################################
            ### Compile neural network
            model,history = loadmodel(XtrainS,XvalS,Ytrain,Yval,hidden,random_network_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,class_weight,input_shape,output_shape)
            
            ###############################################################################
            ### Actual hiatus
            actual_classtrain = classeslnew[trainIndices,:,:].ravel()
            actual_classtest = classeslnew[testIndices,:,:].ravel()
            actual_classval = classeslnew[valIndices,:,:].ravel()
            actual_classtrain = np.asarray(actual_classtrain,dtype=int)
            actual_classtest = np.asarray(actual_classtest,dtype=int)
            actual_classval = np.asarray(actual_classval,dtype=int)
            
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Prediction for training
            ypred_train = model.predict(XtrainS,verbose=1)
            ypred_picktrain = np.argmax(ypred_train,axis=1)
            
            ### Prediction for testing
            ypred_test = model.predict(XtestS,verbose=1)
            ypred_picktest = np.argmax(ypred_test,axis=1)
            
            ### Prediction for validation
            ypred_val = model.predict(XvalS,verbose=1)
            ypred_pickval = np.argmax(ypred_val,axis=1)
            
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Start saving everything
            savename = 'ANNv2_EmissionScenarioBinary_ssp_245-585_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)  + '_COUNTER' + str(seee+1) 
    
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Training/testing for saving output
            np.savetxt(directoryoutput + 'trainingEnsIndices_' + savename + '.txt',trainIndices)
            np.savetxt(directoryoutput + 'testingEnsIndices_' + savename + '.txt',testIndices)
            np.savetxt(directoryoutput + 'validationEnsIndices_' + savename + '.txt',valIndices)
            
            np.savetxt(directoryoutput + 'trainingTrueLabels_' + savename + '.txt',actual_classtrain)
            np.savetxt(directoryoutput + 'testingTrueLabels_' + savename + '.txt',actual_classtest)
            np.savetxt(directoryoutput + 'validationTrueLabels_' + savename + '.txt',actual_classval)
            
            np.savetxt(directoryoutput + 'trainingPredictedLabels_' + savename + '.txt',ypred_picktrain)
            np.savetxt(directoryoutput + 'trainingPredictedConfidence_' + savename+ '.txt',ypred_train)
            np.savetxt(directoryoutput + 'testingPredictedLabels_' + savename+ '.txt',ypred_picktest)
            np.savetxt(directoryoutput + 'testingPredictedConfidence_' + savename+ '.txt',ypred_test)
            np.savetxt(directoryoutput + 'validationPredictedLabels_' + savename+ '.txt',ypred_pickval)
            np.savetxt(directoryoutput + 'validationPredictedConfidence_' + savename+ '.txt',ypred_val)
            
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
            
            acctrain = accuracyTotalTime(ypred_picktrain,actual_classtrain)     
            acctest = accuracyTotalTime(ypred_picktest,actual_classtest)
            accval = accuracyTotalTime(ypred_pickval,actual_classval)
            
            prectrain = precisionTotalTime(ypred_picktrain,actual_classtrain)     
            prectest = precisionTotalTime(ypred_picktest,actual_classtest)
            precval = precisionTotalTime(ypred_pickval,actual_classval)
            
            recalltrain = recallTotalTime(ypred_picktrain,actual_classtrain)     
            recalltest = recallTotalTime(ypred_picktest,actual_classtest)
            recallval = recallTotalTime(ypred_pickval,actual_classval)
            
            f1_train = f1TotalTime(ypred_picktrain,actual_classtrain)     
            f1_test = f1TotalTime(ypred_picktest,actual_classtest)
            f1_val = f1TotalTime(ypred_pickval,actual_classval)
            
            ### Savedata
            np.savez(directoryoutputs + savename + '_SCORES_LoopHyper.npz',
                     acctrain=acctrain,acctest=acctest,accval=accval,
                     prectrain=prectrain,prectest=prectest,
                     precval=precval,recalltrain=recalltrain,
                     recalltest=recalltest,recallval=recallval,
                     f1_train=f1_train,f1_test=f1_test,f1_val=f1_val)
            
            ##############################################################################
            ##############################################################################
            ##############################################################################
            ## Visualizing through LRP
            numLats = lats.shape[0]
            numLons = lons.shape[0]  
            numDim = 3
            
            ##############################################################################
            ##############################################################################
            ##############################################################################
            ### For testing data only (z-rule)
            # lrptestz = LRPf.calc_LRPModel(model,XtestS,Ytest,biasBool,
            #                                         annType,num_of_class,
            #                                         yearsall,lrpRule1,normLRP,
            #                                         numLats,numLons,numDim)
            
            # netcdfLRPz(lats,lons,lrptestz,directoryoutputl,'Testing',savename)
                
            ##############################################################################
            ##############################################################################
            ##############################################################################
            ### Overshoot stuff
            ### Try one ensemble member for SPEAR SSP534OS
            ensembleMember = 0
            ensembleMember_10ye = 0
            testingEnsembleMemberSq = dSS.read_InferenceLargeEnsemble(variq,'SPEAR_MED_Scenario',dataset_obs,monthlychoice,
                                                                    'SSP534OS','MED',lat_bounds,lon_bounds,
                                                                    land_only,ocean_only,Xmean,Xstd,ensembleMember)
            ypred_overshootq = model.predict(testingEnsembleMemberSq,verbose=1)
            ypred_pickovershootq = np.argmax(ypred_overshootq,axis=1)
                
            ###############################################################
            ###############################################################
            ###############################################################
            ### Try one ensemble members for SPEAR SSP534OS_10ye
            testingEnsembleMemberSq_10ye = dSS.read_InferenceLargeEnsemble(variq,'SPEAR_MED_SSP534OS_10ye',dataset_obs,monthlychoice,
                                                                    'SSP534OS_10ye','MED',lat_bounds,lon_bounds,
                                                                    land_only,ocean_only,Xmean,Xstd,ensembleMember_10ye)
            ypred_overshootq_10ye = model.predict(testingEnsembleMemberSq_10ye,verbose=1)
            ypred_pickovershootq_10ye = np.argmax(ypred_overshootq_10ye,axis=1)
            
            ### For OS data only to calculate XAI
            lrp_os = LRPf.calc_LRPObs(model,testingEnsembleMemberSq,biasBool,annType,
                                                num_of_class,yearsall,lrpRule3,
                                                normLRP,numLats,numLons,numDim)
            lrp_os_10ye = LRPf.calc_LRPObs(model,testingEnsembleMemberSq_10ye,biasBool,annType,
                                                num_of_class,yearsall,lrpRule3,
                                                normLRP,numLats,numLons,numDim)
            
            netcdfLRPz_OS(lats,lons,lrp_os,directoryoutputl,'OS',savename)
            netcdfLRPz_OS(lats,lons,lrp_os_10ye,directoryoutputl,'OS_10ye',savename)
            
            ##############################################################################
            ##############################################################################
            ##############################################################################
            ### Save overshoot predictions for all ensemble members
            np.savetxt(directoryoutputover + 'overshootPredictedLabels_' + savename+ '.txt',ypred_pickovershootq)
            np.savez(directoryoutputover + 'overshootPredictedConfidence_' + savename+ '.npz',overshootconf = ypred_overshootq)
            np.savetxt(directoryoutputover + 'overshoot_10yePredictedLabels_' + savename+ '.txt',ypred_pickovershootq_10ye)
            np.savez(directoryoutputover + 'overshoot_10yePredictedConfidence_' + savename+ '.npz',overshootconf = ypred_overshootq)
    
            ### Save metadata
            random_network_seedall.append(random_network_seed)
            random_segment_seedall.append(random_segment_seed)
            savenameall.append(savename)
            
            print('\n\n\n<<<<<<<<< [%s] ITERATION for %s with l2=%s  >>>>>>>>>>\n\n\n' % (seee+1,hidden,ridgePenalty))
            
### Save seeds
np.savez(directoryoutputo + 'Metadata_LoopResultsfor_ANNv2_EmissionScenarioBinary_ssp_245-585__LoopHyper.npz',
         random_network_seedall=np.asarray(random_network_seedall),
         random_segment_seedall=np.asarray(random_segment_seedall),
         savenamesall=np.asarray(savenameall))
