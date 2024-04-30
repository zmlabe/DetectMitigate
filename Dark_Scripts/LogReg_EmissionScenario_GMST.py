"""
Logistic regression to predict the SSP based on the GMt2m

Author     : Zachary M. Labe
Date       : 13 July 2023
Version    : 1
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
### Model Parameters
fac = 0.80 # 0.70 training, 0.13 validation, 0.07 for testing
random_segment_seed = 71541
random_network_seed = 87750

### Model paramaters for the same ANN equivalent
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

### Read in data for training predictions and actual hiatuses
dirname = '/work/Zachary.Labe/Research/DetectMitigate/Data/'
savename = 'ANNv4_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 

trainindices = np.asarray(np.genfromtxt(dirname + 'trainingEnsIndices_' + savename + '.txt'),dtype=int)

###############################################################################
###############################################################################
###############################################################################
### Read in data for testing predictions and actual hiatuses
testindices = np.asarray(np.genfromtxt(dirname + 'testingEnsIndices_' + savename + '.txt'),dtype=int)

###############################################################################
###############################################################################
###############################################################################
### Read in data for validation predictions and actual hiatuses
valindices = np.asarray(np.genfromtxt(dirname + 'validationEnsIndices_' + savename + '.txt'),dtype=int)

### Read in GMT2M
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/TimeSeries/'
T2M = np.load(directoryoutput + 'GMT2M_EmissionScenario.npy')
T2Mtrain = np.swapaxes(T2M[:,trainindices,:],0,1).squeeze()
T2Mtest = np.swapaxes(T2M[:,testindices,:],0,1).squeeze()
T2Mval= np.swapaxes(T2M[:,valindices,:],0,1).squeeze()

Xtrain = T2Mtrain.reshape(lenOfPicks*trainindices.shape[0]*yearsall[0].shape[0],1)
Xval = T2Mval.reshape(lenOfPicks*valindices.shape[0]*yearsall[0].shape[0],1)
Xtest = T2Mtest.reshape(lenOfPicks*testindices.shape[0]*yearsall[0].shape[0],1)
XtrainS,XtestS,XvalS,stdVals = dSS.standardize_dataVal(Xtrain,Xtest,Xval)

Ytrain = classeslnew[trainindices,:,:].reshape(lenOfPicks*trainindices.shape[0]*yearsall[0].shape[0])
Ytrain = keras.utils.to_categorical(Ytrain)
Yval = classeslnew[valindices,:,:].reshape(lenOfPicks*valindices.shape[0]*yearsall[0].shape[0])
Yval= keras.utils.to_categorical(Yval)
Ytest = classeslnew[testindices,:,:].reshape(lenOfPicks*testindices.shape[0]*yearsall[0].shape[0])
Ytest = keras.utils.to_categorical(Ytest)

### Other Model paramaters
input_shape=np.shape(Xtrain)[1]
output_shape=np.shape(Ytrain)[1]

### Create class weights
def class_weight_creator(Y):
    class_dict = {}
    weights = np.max(np.sum(Y, axis=0)) / np.sum(Y, axis=0)
    for i in range( Y.shape[-1] ):
        class_dict[i] = weights[i]               
    return class_dict
class_weight = class_weight_creator(Ytrain)

###############################################################################
###############################################################################
###############################################################################
def loadmodel(Xtrain,Xval,Ytrain,Yval,hidden,random_network_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,class_weight,input_shape,output_shape):
    print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')
    keras.backend.clear_session()
    model = keras.models.Sequential()
    tf.random.set_seed(int(np.random.randint(1,100)))
    
    if random_network_seed == None:
        np.random.seed(None)
        random_network_seed = int(np.random.randint(1, 100000))

    #### One layer model
    model.add(Dense(output_shape,input_shape=(input_shape,),activation=actFun,use_bias=True,
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.00,l2=0.00),
                    bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))

    ### Add softmax layer at the end
    model.add(Activation('softmax'))
    
    ### Compile the model
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01,
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
    # plt.savefig(directoryfigure + 'LogReg_GMST_loss.png',dpi=300)
    
    plt.subplot(1,2,2)
    plt.plot(history.history['categorical_accuracy'],label = 'training')
    plt.plot(history.history['val_categorical_accuracy'],label = 'validation')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    # plt.savefig(directoryfigure + 'LogReg_GMST_accuracy.png',dpi=300)
    
    model.summary() 
    return model,history

###############################################################################
### Compile neural network
model,history = loadmodel(XtrainS,XvalS,Ytrain,Yval,hidden,random_network_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,class_weight,input_shape,output_shape)

###############################################################################
### Actual hiatus
actual_classtrain = classeslnew[trainindices,:,:].ravel()
actual_classtest = classeslnew[testindices,:,:].ravel()
actual_classval = classeslnew[valindices,:,:].ravel()
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
dirname = '/work/Zachary.Labe/Research/DetectMitigate/savedModels'
savename = 'ANNv3_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 

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

f1_train = f1TotalTime(ypred_picktrain,actual_classtrain)     
f1_test = f1TotalTime(ypred_picktest,actual_classtest)
f1_val = f1TotalTime(ypred_pickval,actual_classval)

plt.figure()
cm = confusion_matrix(actual_classtest,ypred_picktest)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=scenarioall)
disp.plot(cmap=cmr.fall)
plt.savefig(directoryfigure + 'ConfusionMatrix_GMST_loss.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Read in OS data to make predictions
osdata = np.load(directoryoutput + 'GMT2M_EmissionScenario-OS.npy').ravel()[:,np.newaxis]
osdata_10ye = np.load(directoryoutput + 'GMT2M_EmissionScenario-OS_10ye.npy').ravel()[:,np.newaxis]

### Standardize
Xmean,Xstd = stdVals

osdataS = (osdata - Xmean)/Xstd
osdata_10yeS = (osdata_10ye - Xmean)/Xstd

os_predict = model.predict(osdataS)
os_pick = np.argmax(os_predict,axis=1)
os_10ye_predict = model.predict(osdata_10yeS)
os_10ye_pick = np.argmax(os_10ye_predict,axis=1)

print('\n',acctrain,accval,acctest)
print('\n',f1_test)
