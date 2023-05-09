"""
First version of classification problem for emission scenario

Author     : Zachary M. Labe
Date       : 13 September 2021
Version    : 1 
"""

### Import packages
import sys
import matplotlib.pyplot as plt
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

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Directories
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_NATURAL','SPEAR_MED','SPEAR_MED_Scenario','SPEAR_MED_Scenario']
dataset_obs = 'ERA5_MEDS'
lenOfPicks = len(modelGCMs)
allDataLabels = modelGCMs
monthlychoice = 'annual'
variq = 'SNOW'
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
dataset_inference = True
timeper = ['historicalforcing','futureforcing','futureforcing','futureforcing','futureforcing']
scenarioall = ['historical','natural','SSP585','SSP119','SSP245']
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
yearsobs = np.arange(1979+window,2021+1,1)
###############################################################################
###############################################################################
numOfEns = 30
numOfEns_10ye = 9
lentime = len(yearsall)
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

###############################################################################
###############################################################################
###############################################################################
### Segment data for training and testing data
fac = 0.70 # 0.70 training, 0.2 validation, 0.1 for testing
random_segment_seed = 71541
random_network_seed = 87750
Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtrain_shape,Xtest_shape,Xval_shape,testIndices,trainIndices,valIndices,class_weight = FRAC.segment_data(data,classesl,ensTypeExperi,fac,random_segment_seed)

### Model paramaters
# hidden = [20,20]
# n_epochs = 500
# batch_size = 128
# lr_here = 0.001
# ridgePenalty = 0.05
hidden = [10,10,10]
n_epochs = 500
batch_size = 128
lr_here = 0.001
ridgePenalty = 0.1
actFun = 'relu'
input_shape=np.shape(Xtrain)[1]
output_shape=np.shape(Ytrain)[1]

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

###############################################################################
###############################################################################
###############################################################################
### Start saving everything, including the ANN
dirname = '/home/Zachary.Labe/Research/DetectMitigate/SavedModels/'
savename = 'ANNv2_EmissionScenario_' + variq + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 

modelwrite = dirname + savename + '.h5'
# model.save_weights(modelwrite)
# np.savez(dirname + savename + '.npz',trainModels=trainIndices,testModels=testIndices,Xtrain=Xtrain,Ytrain=Ytrain,Xtest=Xtest,Ytest=Ytest,Xmean=Xmean,Xstd=Xstd,lats=lats,lons=lons)

###############################################################################
###############################################################################
###############################################################################
### Observations saving output
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/'

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

np.savetxt(directoryoutput + 'overshootPredictedLabels_' + savename+ '.txt',ypred_pickovershoot_all)
np.savez(directoryoutput + 'overshootPredictedConfidence_' + savename+ '.npz',overshootconf = ypred_overshoot_all)
np.savetxt(directoryoutput + 'overshoot_10yePredictedLabels_' + savename+ '.txt',ypred_pickovershoot_10ye_all)
np.savez(directoryoutput + 'overshoot_10yePredictedConfidence_' + savename+ '.npz',overshootconf = ypred_overshoot_10ye_all)

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
