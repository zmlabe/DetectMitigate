"""
Third version of classification problem for emission scenario

Author     : Zachary M. Labe
Date       : 27 February 2023
Version    : 2 - added XAI capability, 3 - added capability for all XAI of OS ensemble members
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import cmocean
import cmasher as cmr
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
from sklearn.metrics import ConfusionMatrixDisplay
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
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
directoryoutput = '/work/Zachary.Labe/Research/DetectMitigate/Data/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_NATURAL','SPEAR_MED','SPEAR_MED_Scenario','SPEAR_MED_Scenario']
dataset_obs = 'ERA5_MEDS'
lenOfPicks = len(modelGCMs)
allDataLabels = modelGCMs
monthlychoice = 'annual'
variq = 'PRECT'
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
timeper = ['historicalforcing','futureforcing','futureforcing','futureforcing','futureforcing']
scenarioall = ['historical','natural','SSP585','SSP119','SSP245']
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
fac = 0.80 # 0.70 training, 0.13 validation, 0.07 for testing
random_segment_seed = 71541
random_network_seed = 87750
Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtrain_shape,Xtest_shape,Xval_shape,testIndices,trainIndices,valIndices,class_weight = FRAC.segment_data(data,classesl,ensTypeExperi,fac,random_segment_seed)

### Model paramaters
hidden = [30,30,30]
n_epochs = 1500
batch_size = 256
lr_here = 0.0001
ridgePenalty = 0.8
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
    
    ### See epochs
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],label = 'training')
    plt.plot(history.history['val_loss'], label = 'validation')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(directoryfigure + 'loss.png',dpi=300)
    
    plt.subplot(1,2,2)
    plt.plot(history.history['categorical_accuracy'],label = 'training')
    plt.plot(history.history['val_categorical_accuracy'],label = 'validation')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(directoryfigure + 'accuracy.png',dpi=300)
    
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

###############################################################
###############################################################
###############################################################
### Read in observations (standardize by training mean)
data_obs,lats_obs,lons_obs = read_primary_dataset(variq,dataset_obs,monthlychoice,scenario,lat_bounds,lon_bounds)

### Standardize obs
Xobs = data_obs.reshape(data_obs.shape[0],data_obs.shape[1]*data_obs.shape[2])
XobsS = (Xobs - Xmean)/Xstd
XobsS[np.isnan(XobsS)] = 0 

### Predict observations
ypred_obs = model.predict(XobsS)
ypred_pickobs = np.argmax(ypred_obs,axis=1)

###############################################################
###############################################################
###############################################################
### Read in observations (standardize by obs themselves - experimental)
data_obst,lats_obst,lons_obst = read_primary_dataset(variq,dataset_obs,monthlychoice,scenario,lat_bounds,lon_bounds)

### Standardize obs
Xobst = data_obst.reshape(data_obst.shape[0],data_obst.shape[1]*data_obst.shape[2])
Xobstmean = np.nanmean(Xobst,axis=0)
Xobststd = np.nanstd(Xobst,axis=0)
XobstS = (Xobst - Xobstmean)/Xobststd
XobstS[np.isnan(XobstS)] = 0 

### Predict observations
ypred_obst = model.predict(XobstS)
ypred_pickobst = np.argmax(ypred_obst,axis=1)

###############################################################################
###############################################################################
###############################################################################
### Start saving everything, including the ANN
dirname = '/work/Zachary.Labe/Research/DetectMitigate/savedModels'
savename = 'ANNv3_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 

modelwrite = dirname + savename + '.h5'
# model.save_weights(modelwrite)
# np.savez(dirname + savename + '.npz',trainModels=trainIndices,testModels=testIndices,Xtrain=Xtrain,Ytrain=Ytrain,Xtest=Xtest,Ytest=Ytest,Xmean=Xmean,Xstd=Xstd,lats=lats,lons=lons)

# ###############################################################################
# ###############################################################################
# ###############################################################################
# ### Training/testing for saving output
# np.savetxt(directoryoutput + 'trainingEnsIndices_' + savename + '.txt',trainIndices)
# np.savetxt(directoryoutput + 'testingEnsIndices_' + savename + '.txt',testIndices)
# np.savetxt(directoryoutput + 'validationEnsIndices_' + savename + '.txt',valIndices)

# np.savetxt(directoryoutput + 'trainingTrueLabels_' + savename + '.txt',actual_classtrain)
# np.savetxt(directoryoutput + 'testingTrueLabels_' + savename + '.txt',actual_classtest)
# np.savetxt(directoryoutput + 'validationTrueLabels_' + savename + '.txt',actual_classval)

# np.savetxt(directoryoutput + 'trainingPredictedLabels_' + savename + '.txt',ypred_picktrain)
# np.savetxt(directoryoutput + 'trainingPredictedConfidence_' + savename+ '.txt',ypred_train)
# np.savetxt(directoryoutput + 'testingPredictedLabels_' + savename+ '.txt',ypred_picktest)
# np.savetxt(directoryoutput + 'testingPredictedConfidence_' + savename+ '.txt',ypred_test)
# np.savetxt(directoryoutput + 'validationPredictedLabels_' + savename+ '.txt',ypred_pickval)
# np.savetxt(directoryoutput + 'validationPredictedConfidence_' + savename+ '.txt',ypred_val)

# np.savetxt(directoryoutput + 'observationsPredictedLabels_' + savename+ '.txt',ypred_pickobs)
# np.savez(directoryoutput + 'observationsPredictedConfidence_' + savename+ '.npz',obsconf = ypred_obs,yearsobs = yearsobs)

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

acctrain = accuracyTotalTime(ypred_picktrain,actual_classtrain)     
acctest = accuracyTotalTime(ypred_picktest,actual_classtest)
accval = accuracyTotalTime(ypred_pickval,actual_classval)
print(acctrain,accval,acctest)

plt.figure()
cm = confusion_matrix(actual_classtest,ypred_picktest)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=scenarioall)
disp.plot(cmap=cmr.fall)

sys.exit()
### Define variable for analysis
print('\n\n------------------------')
print(variq,'= Variable!')
print(monthlychoice,'= Time!')
print(reg_name,'= Region!')
print(lat_bounds,lon_bounds)
print(dataset,'= Model!')
print(dataset_obs,'= Observations!\n')
print(land_only,'= land_only')
print(ocean_only,'= ocean_only')

### Variables for plotting
lons2,lats2 = np.meshgrid(lons,lats) 
modeldata = data
modeldatamean = np.nanmean(modeldata,axis=1)

spatialmean_mod = UT.calc_weightedAve(modeldata,lats2)
spatialmean_modmean = np.nanmean(spatialmean_mod,axis=1)
plt.figure()
plt.plot(spatialmean_modmean.transpose())

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
lrptestz = LRPf.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                        annType,num_of_class,
                                        yearsall,lrpRule1,normLRP,
                                        numLats,numLons,numDim)
### For testing data only (e-rule)
lrpteste = LRPf.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                        annType,num_of_class,
                                        yearsall,lrpRule2,normLRP,
                                        numLats,numLons,numDim)
### For testing data only (integrated gradients)
lrptestig = LRPf.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                        annType,num_of_class,
                                        yearsall,lrpRule3,normLRP,
                                        numLats,numLons,numDim)

### Test plot for LRP
meanlrp = np.nanmean(lrptestz,axis=0)
fig=plt.figure()
plt.contourf(meanlrp,300,cmap=cmocean.cm.thermal)

##############################################################################
##############################################################################
##############################################################################
### For obs data only (z-rule)
lrpobsz = LRPf.calc_LRPObs(model,XobsS,biasBool,annType,num_of_class,
                            yearsall,lrpRule1,normLRP,numLats,numLons,numDim)

### For testing data only (e-rule)
lrpobse = LRPf.calc_LRPObs(model,XobsS,biasBool,annType,num_of_class,
                            yearsall,lrpRule2,normLRP,numLats,numLons,numDim)

### For testing data only (integrated gradients)
lrpobsig = LRPf.calc_LRPObs(model,XobsS,biasBool,annType,num_of_class,
                            yearsall,lrpRule3,normLRP,numLats,numLons,numDim)

### Test plot for LRP
meanobslrp = np.nanmean(lrpobsz,axis=0)
fig=plt.figure()
plt.contourf(meanobslrp,300,cmap=cmocean.cm.thermal)

##############################################################################
##############################################################################
##############################################################################
def netcdfLRPz(lats,lons,var,directory,typemodel,savename):
    print('\n>>> Using netcdfLRP function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'LRP/LRPMap_Z' + typemodel + '_' + savename + '.nc'
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

netcdfLRPz(lats,lons,lrptestz,directoryoutput,'Testing',savename)
netcdfLRPz(lats,lons,lrptestz,directoryoutput,'Obs',savename)

##############################################################################
##############################################################################
##############################################################################
def netcdfLRPe(lats,lons,var,directory,typemodel,savename):
    print('\n>>> Using netcdfLRP-e-rule function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'LRP/LRPMap_E' + typemodel + '_' + savename + '.nc'
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
    
netcdfLRPe(lats,lons,lrpteste,directoryoutput,'Testing',savename)
netcdfLRPe(lats,lons,lrpteste,directoryoutput,'Obs',savename)

##############################################################################
##############################################################################
##############################################################################
def netcdfLRPig(lats,lons,var,directory,typemodel,savename):
    print('\n>>> Using netcdfLRP-integratedgradients function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'LRP/LRPMap_IG' + typemodel + '_' + savename + '.nc'
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
    
netcdfLRPig(lats,lons,lrptestig,directoryoutput,'Testing',savename)
netcdfLRPig(lats,lons,lrptestig,directoryoutput,'Obs',savename)

##############################################################################
##############################################################################
##############################################################################
### Overshoot files
def netcdfLRPz_OS(lats,lons,var,directory,typemodel,savename):
    print('\n>>> Using netcdfLRP function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'LRP/LRPMap_Z' + typemodel + '_' + savename + '.nc'
    filename = directory + name
    ncfile = Dataset(filename,'w',format='NETCDF4')
    ncfile.description = 'LRP maps for using selected seed of overshoot runs' 
    
    ### Dimensions
    ncfile.createDimension('ensembles',var.shape[0])
    ncfile.createDimension('years',var.shape[1])
    ncfile.createDimension('lat',var.shape[2])
    ncfile.createDimension('lon',var.shape[3])
    
    ### Variables
    ensembles = ncfile.createVariable('ensembles','f4',('ensembles'))
    years = ncfile.createVariable('years','f4',('years'))
    latitude = ncfile.createVariable('lat','f4',('lat'))
    longitude = ncfile.createVariable('lon','f4',('lon'))
    varns = ncfile.createVariable('LRP','f4',('ensembles','years','lat','lon'))
    
    ### Units
    varns.units = 'unitless relevance'
    ncfile.title = 'LRP relevance'
    ncfile.instituion = 'Colorado State University'
    ncfile.references = 'Barnes et al. [2020]'
    
    ### Data
    ensembles[:] = np.arange(var.shape[0])
    years[:] = np.arange(var.shape[1])
    latitude[:] = lats
    longitude[:] = lons
    varns[:] = var
    
    ncfile.close()
    print('*Completed: Created netCDF4 File!')

##############################################################################
##############################################################################
##############################################################################
def netcdfLRPe_OS(lats,lons,var,directory,typemodel,savename):
    print('\n>>> Using netcdfLRP-e-rule function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'LRP/LRPMap_E' + typemodel + '_' + savename + '.nc'
    filename = directory + name
    ncfile = Dataset(filename,'w',format='NETCDF4')
    ncfile.description = 'LRP maps for using selected seed' 
    
    ### Dimensions
    ncfile.createDimension('ensembles',var.shape[0])
    ncfile.createDimension('years',var.shape[1])
    ncfile.createDimension('lat',var.shape[2])
    ncfile.createDimension('lon',var.shape[3])
    
    ### Variables
    ensembles = ncfile.createVariable('ensembles','f4',('ensembles'))
    years = ncfile.createVariable('years','f4',('years'))
    latitude = ncfile.createVariable('lat','f4',('lat'))
    longitude = ncfile.createVariable('lon','f4',('lon'))
    varns = ncfile.createVariable('LRP','f4',('ensembles','years','lat','lon'))
    
    ### Units
    varns.units = 'unitless relevance'
    ncfile.title = 'LRP relevance'
    ncfile.instituion = 'Colorado State University'
    ncfile.references = 'Barnes et al. [2020]'
    
    ### Data
    ensembles[:] = np.arange(var.shape[0])
    years[:] = np.arange(var.shape[1])
    latitude[:] = lats
    longitude[:] = lons
    varns[:] = var
    
    ncfile.close()
    print('*Completed: Created netCDF4 File!')

##############################################################################
##############################################################################
##############################################################################
def netcdfLRPig_OS(lats,lons,var,directory,typemodel,savename):
    print('\n>>> Using netcdfLRP-integratedgradients function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'LRP/LRPMap_IG' + typemodel + '_' + savename + '.nc'
    filename = directory + name
    ncfile = Dataset(filename,'w',format='NETCDF4')
    ncfile.description = 'LRP maps for using selected seed' 
    
    ### Dimensions
    ncfile.createDimension('ensembles',var.shape[0])
    ncfile.createDimension('years',var.shape[1])
    ncfile.createDimension('lat',var.shape[2])
    ncfile.createDimension('lon',var.shape[3])
    
    ### Variables
    ensembles = ncfile.createVariable('ensembles','f4',('ensembles'))
    years = ncfile.createVariable('years','f4',('years'))
    latitude = ncfile.createVariable('lat','f4',('lat'))
    longitude = ncfile.createVariable('lon','f4',('lon'))
    varns = ncfile.createVariable('LRP','f4',('ensembles','years','lat','lon'))
    
    ### Units
    varns.units = 'unitless relevance'
    ncfile.title = 'LRP relevance'
    ncfile.instituion = 'Colorado State University'
    ncfile.references = 'Barnes et al. [2020]'
    
    ### Data
    ensembles[:] = np.arange(var.shape[0])
    years[:] = np.arange(var.shape[1])
    latitude[:] = lats
    longitude[:] = lons
    varns[:] = var
    
    ncfile.close()
    print('*Completed: Created netCDF4 File!')
    
##############################################################################
##############################################################################
##############################################################################
### Overshoot stuff
### Try all 30 ensemble members for SPEAR SSP534OS
ensembleMembers_all = []
testingEnsembleMemberS_all = []
ypred_overshoot_all = []
ypred_pickovershoot_all = []
for ensIn in range(numOfEns):
    if dataset_inference:
        ensembleMember = ensIn
        testingEnsembleMemberSq = dSS.read_InferenceLargeEnsemble(variq,'SPEAR_MED_Scenario',dataset_obs,monthlychoice,
                                                                'SSP534OS','MED',lat_bounds,lon_bounds,
                                                                land_only,ocean_only,Xmean,Xstd,ensembleMember)
        ypred_overshootq = model.predict(testingEnsembleMemberSq,verbose=1)
        ypred_pickovershootq = np.argmax(ypred_overshootq,axis=1)
    else:
        ensembleMember = np.nan
        testingEnsembleMemberSq = np.nan
        ypred_overshootq = np.nan
        ypred_pickovershootq = np.nan
        
    ensembleMembers_all.append(ensembleMember)
    testingEnsembleMemberS_all.append(testingEnsembleMemberSq)
    ypred_overshoot_all.append(ypred_overshootq)
    ypred_pickovershoot_all.append(ypred_pickovershootq)
    
###############################################################
###############################################################
###############################################################
### Try all 9 ensemble members for SPEAR SSP534OS_10ye
ensembleMembers_10ye_all = []
testingEnsembleMemberS_10ye_all = []
ypred_overshoot_10ye_all = []
ypred_pickovershoot_10ye_all = []
for ensIn_10ye in range(numOfEns_10ye):
    if dataset_inference:
        ensembleMember_10ye = ensIn_10ye
        testingEnsembleMemberSq_10ye = dSS.read_InferenceLargeEnsemble(variq,'SPEAR_MED_SSP534OS_10ye',dataset_obs,monthlychoice,
                                                                'SSP534OS_10ye','MED',lat_bounds,lon_bounds,
                                                                land_only,ocean_only,Xmean,Xstd,ensembleMember_10ye)
        ypred_overshootq_10ye = model.predict(testingEnsembleMemberSq_10ye,verbose=1)
        ypred_pickovershootq_10ye = np.argmax(ypred_overshootq_10ye,axis=1)
    else:
        ensembleMember_10ye = np.nan
        testingEnsembleMemberSq_10ye = np.nan
        ypred_overshootq_10ye = np.nan
        ypred_pickovershootq_10ye = np.nan
        
    ensembleMembers_10ye_all.append(ensembleMember_10ye)
    testingEnsembleMemberS_10ye_all.append(testingEnsembleMemberSq_10ye)
    ypred_overshoot_10ye_all.append(ypred_overshootq_10ye)
    ypred_pickovershoot_10ye_all.append(ypred_pickovershootq_10ye)

### For OS data only to calculate XAI
lrp_os_z = np.full((numOfEns,yearsall[-1].shape[0],lats.shape[0],lons.shape[0]),np.nan)
lrp_os_e = np.full((numOfEns,yearsall[-1].shape[0],lats.shape[0],lons.shape[0]),np.nan)
lrp_os_ig = np.full((numOfEns,yearsall[-1].shape[0],lats.shape[0],lons.shape[0]),np.nan)
for osi in range(len(testingEnsembleMemberS_all)):
    testingEnsembleMemberSq = testingEnsembleMemberS_all[osi]
    lrp_os_z[osi,:,:,:] = LRPf.calc_LRPObs(model,testingEnsembleMemberSq,biasBool,annType,
                                        num_of_class,yearsall,lrpRule1,
                                        normLRP,numLats,numLons,numDim)
    lrp_os_e[osi,:,:,:] = LRPf.calc_LRPObs(model,testingEnsembleMemberSq,biasBool,annType,
                                        num_of_class,yearsall,lrpRule2,
                                        normLRP,numLats,numLons,numDim)
    lrp_os_ig[osi,:,:,:] = LRPf.calc_LRPObs(model,testingEnsembleMemberSq,biasBool,annType,
                                        num_of_class,yearsall,lrpRule3,
                                        normLRP,numLats,numLons,numDim)

lrp_os_10ye_z = np.full((numOfEns_10ye,yearsall[-1].shape[0],lats.shape[0],lons.shape[0]),np.nan)
lrp_os_10ye_e = np.full((numOfEns_10ye,yearsall[-1].shape[0],lats.shape[0],lons.shape[0]),np.nan)
lrp_os_10ye_ig = np.full((numOfEns_10ye,yearsall[-1].shape[0],lats.shape[0],lons.shape[0]),np.nan)
for osti in range(len(testingEnsembleMemberS_10ye_all)):
    testingEnsembleMemberSq_10ye = testingEnsembleMemberS_10ye_all[osti]
    lrp_os_10ye_z[osti,:,:,:] = LRPf.calc_LRPObs(model,testingEnsembleMemberSq_10ye,biasBool,annType,
                                        num_of_class,yearsall,lrpRule1,
                                        normLRP,numLats,numLons,numDim)
    lrp_os_10ye_e[osti,:,:,:] = LRPf.calc_LRPObs(model,testingEnsembleMemberSq_10ye,biasBool,annType,
                                        num_of_class,yearsall,lrpRule2,
                                        normLRP,numLats,numLons,numDim)
    lrp_os_10ye_ig[osti,:,:,:] = LRPf.calc_LRPObs(model,testingEnsembleMemberSq_10ye,biasBool,annType,
                                        num_of_class,yearsall,lrpRule3,
                                        normLRP,numLats,numLons,numDim)
    
netcdfLRPz_OS(lats,lons,lrp_os_z,directoryoutput,'OS',savename)
netcdfLRPe_OS(lats,lons,lrp_os_e,directoryoutput,'OS',savename)
netcdfLRPig_OS(lats,lons,lrp_os_ig,directoryoutput,'OS',savename)

netcdfLRPz_OS(lats,lons,lrp_os_10ye_z,directoryoutput,'OS_10ye',savename)
netcdfLRPe_OS(lats,lons,lrp_os_10ye_e,directoryoutput,'OS_10ye',savename)
netcdfLRPig_OS(lats,lons,lrp_os_10ye_ig,directoryoutput,'OS_10ye',savename)

##############################################################################
##############################################################################
##############################################################################
### Save overshoot predictions for all ensemble members
np.savetxt(directoryoutput + 'overshootPredictedLabels_' + savename+ '.txt',ypred_pickovershoot_all)
np.savez(directoryoutput + 'overshootPredictedConfidence_' + savename+ '.npz',overshootconf = ypred_overshoot_all)
np.savetxt(directoryoutput + 'overshoot_10yePredictedLabels_' + savename+ '.txt',ypred_pickovershoot_10ye_all)
np.savez(directoryoutput + 'overshoot_10yePredictedConfidence_' + savename+ '.npz',overshootconf = ypred_overshoot_10ye_all)
