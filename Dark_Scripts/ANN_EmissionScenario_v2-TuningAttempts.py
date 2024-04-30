"""
First version of classification problem for emission scenario

Author     : Zachary M. Labe
Date       : 11 January 2013
Version    : 2 - added XAI capability, randomly attempting to tune ANN first
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import math
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_SegmentData as FRAC
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Activation
import keras_tuner
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
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_NATURAL','SPEAR_MED','SPEAR_MED_Scenario','SPEAR_MED_Scenario']
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
yearsobs = np.arange(1979+window,2021+1,1)
lentime = len(yearsall)
###############################################################################
###############################################################################
numOfEns = 30
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
fac = 0.80 # 0.70 training, 0.13 validation, 0.07 for testing
random_segment_seed = 71541
random_network_seed = 87750
Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtrain_shape,Xtest_shape,Xval_shape,testIndices,trainIndices,valIndices,class_weight = FRAC.segment_data(data,classesl,ensTypeExperi,fac,random_segment_seed)

###############################################################################
### Standardize data
XtrainS,XtestS,XvalS,stdVals = dSS.standardize_dataVal(Xtrain,Xtest,Xval)
Xmean, Xstd = stdVals  

### Model paramaters
hidden = [30,30,30]
n_epochs = 1500
batch_size = 128
lr_here = 0.0001
ridgePenalty = 0.1
actFun = 'relu'
annType = 'class'
input_shape=np.shape(Xtrain)[1]
output_shape=np.shape(Ytrain)[1]

###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Learning rate schedulizer
def step_decay(epoch):
    initial_lrate = lr_here # The initial learning rate
    drop_ratio = 0.5 # The fraction by which the learning rate is decreased
    epochs_drop = 10 # Number of epochs before the learning rate is decreased
    lrate = initial_lrate * drop_ratio**(math.floor((1+epoch)/epochs_drop)) # Re-declaring the learning rate
    return lrate  
def step_decayExp(epoch):
    initial_lrate = lr_here # The initial learning rate
    exp_decay = 0.001
    lrate = initial_lrate * math.exp(-exp_decay*epoch)
    return lrate 
def step_decayTime(epoch):
    initial_lrate = lr_here # The initial learning rate
    lrate = initial_lrate/(epoch+1)
    return lrate

###############################################################################
###############################################################################
###############################################################################
def loadmodel(Xtrain,Xval,Ytrain,Yval,hidden,random_network_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,class_weight,input_shape,output_shape):
    print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')
    
    ### Callbacks
    lrate = keras.callbacks.LearningRateScheduler(step_decayTime)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    verbose=1,
                                                    mode='auto',
                                                    restore_best_weights=True)
    
    ### Clear previous models
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
    model.add(layers.Dropout(rate=0.6,seed=random_network_seed)) 

    ### Initialize other layers
    for layer in hidden[1:]:
        model.add(Dense(layer,activation=actFun,
                        use_bias=True,
                        kernel_regularizer=keras.regularizers.l1_l2(l1=0.00,l2=0.00),
                        bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
         
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
    # model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr_here),  
    #               loss = 'categorical_crossentropy',
    #               metrics=[keras.metrics.categorical_accuracy]) 
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_here),  
    #               loss = 'categorical_crossentropy',
    #               metrics=[keras.metrics.categorical_accuracy]) 
    
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
### Start saving everything, including the ANN
dirname = '/home/Zachary.Labe/Research/DetectMitigate/SavedModels/'
savename = 'ANNv2_EmissionScenario_' + variq + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
modelwrite = dirname + savename + '.h5'

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

### Classification Report
target_names = ["Class: {}".format(scenarioall[i]) for i in range(len(scenarioall))]
print(classification_report(ypred_picktest,actual_classtest,target_names=target_names))
print("\n\n")
print(savename)

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
