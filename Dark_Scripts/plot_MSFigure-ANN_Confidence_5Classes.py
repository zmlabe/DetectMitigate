"""
Fourth version of classification problem for emission scenarios for LRP

Author     : Zachary M. Labe
Date       : 30 October 2023
Version    : 2 - added XAI capability, 3 - added capability for all XAI of OS ensemble members, 4 - selected architecture based on hyperparameter sweep
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import palettable.cubehelix as cm
import palettable.scientific.sequential as sss
import palettable.cartocolors.qualitative as cc
import matplotlib.colors as cccc
import cmocean as cmocean
import cmasher as cmr
import calc_Utilities as UT
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/'
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'
###############################################################################
###############################################################################
###############################################################################
### Extra functions
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
            ax.xaxis.set_ticks([]) 

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
dataset_inference = True
timeper = ['historicalforcing','futureforcing','futureforcing','futureforcing','futureforcing']
scenarioall = ['Historical','Natural','SSP5-8.5','SSP1-1.9','SSP2-4.5']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
scenarioallnames = ['Historical','Natural','SSP5-8.5','SSP1-1.9','SSP2-4.5']
osnames = ['SSP534OS','SSP534OS','SSP534OS_10ye','SSP534OS_10ye']
allvariables = ['Temperature','Precipitation','Temperature','Precipitation']
shuffletype = 'GAUSS'
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
baseline = np.arange(1951,1980+1,1)
years = np.arange(2015,2100+1,1)
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
### Model paramaters
random_segment_seed = 71541
random_network_seed = 87750

fig = plt.figure(figsize=(10,7))
for f in range(len(allvariables)):
    if f == 0:
        variq = 'T2M'
    
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
    
        savename = 'ANNv4_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
        
        ### Import confidence
        confidence = np.load(directorydata + 'overshootPredictedConfidence_' + savename+ '.npz')['overshootconf']
        meanconf = np.nanmean(confidence,axis=0)
        
        ### Import labels
        labelsall = np.genfromtxt(directorydata + 'overshootPredictedLabels_' + savename+ '.txt')
        label = sts.mode(labelsall,axis=0)[0][0]
###############################################################################        
    elif f == 1:
        variq = 'PRECT'
    
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
    
        savename = 'ANNv4_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
        
        ### Import confidence
        confidence = np.load(directorydata + 'overshootPredictedConfidence_' + savename+ '.npz')['overshootconf']
        meanconf = np.nanmean(confidence,axis=0)
        
        ### Import labels
        labelsall = np.genfromtxt(directorydata + 'overshootPredictedLabels_' + savename+ '.txt')
        label = sts.mode(labelsall,axis=0)[0][0]
###############################################################################        
    elif f == 2:
        variq = 'T2M'
    
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
    
        savename = 'ANNv4_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
        
        ### Import confidence
        confidence = np.load(directorydata + 'overshoot_10yePredictedConfidence_' + savename+ '.npz')['overshootconf']
        meanconf = np.nanmean(confidence,axis=0)
        
        ### Import labels
        labelsall = np.genfromtxt(directorydata + 'overshoot_10yePredictedLabels_' + savename+ '.txt')
        label = sts.mode(labelsall,axis=0)[0][0]
###############################################################################        
    elif f == 3:
        variq = 'PRECT'
    
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
    
        savename = 'ANNv4_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
        
        ### Import confidence
        confidence = np.load(directorydata + 'overshoot_10yePredictedConfidence_' + savename+ '.npz')['overshootconf']
        meanconf = np.nanmean(confidence,axis=0)
        
        ### Import labels
        labelsall = np.genfromtxt(directorydata + 'overshoot_10yePredictedLabels_' + savename+ '.txt')
        label = sts.mode(labelsall,axis=0)[0][0]
    
    ### Count frequency 
    freq0 = []
    freq1 = []
    freq2 = []
    freq3 = []
    freq4 = []
    for i in range(len(years)):
        count0 = np.count_nonzero(labelsall[:,i] == 0)
        count1 = np.count_nonzero(labelsall[:,i] == 1)
        count2 = np.count_nonzero(labelsall[:,i] == 2)
        count3 = np.count_nonzero(labelsall[:,i] == 3)
        count4 = np.count_nonzero(labelsall[:,i] == 4)
        
        freq0.append(count0)
        freq1.append(count1)
        freq2.append(count2)
        freq3.append(count3)
        freq4.append(count4)
        
    allfreq = np.vstack([freq0,freq1,freq2,freq3,freq4])
        
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Begin plot
    ax = plt.subplot(2,2,f+1)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.xaxis.grid(color='darkgrey',alpha=0.7,clip_on=False,linewidth=0.5)
    
    if any([f==0,f==1]):
        plt.axvline(x=2040,color='k',linestyle='-',linewidth=2)
    
    if any([f==2,f==3]):
        plt.axvline(x=2031,color='dimgrey',linestyle='--',linewidth=2,dashes=(1,0.3))
    
    color = cmr.infinity(np.linspace(0.00,0.8,len(scenarioall)))
    for i,c in zip(range(len(scenarioall)),color):
        if i == 7:
            c = 'k'
        else:
            c = c
        plt.plot(years,meanconf[:,i],color=c,linewidth=0.6,
                    label=r'\textbf{%s}' % scenarioall[i],zorder=11,
                    clip_on=False,alpha=1)
        plt.scatter(years,meanconf[:,i],color=c,s=28,zorder=12,
                    clip_on=False,alpha=0.2,edgecolors='none')
        
        for yr in range(years.shape[0]):
            la = label[yr]
            if i == la:
                plt.scatter(years[yr],meanconf[yr,i],color=c,s=28,zorder=12,
                            clip_on=False,alpha=1,edgecolors='none')
            
    
    if f == 3:
        leg = plt.legend(shadow=False,fontsize=13,loc='upper center',
                      bbox_to_anchor=(-0.1,-0.1),fancybox=True,ncol=5,frameon=False,
                      handlelength=0.8,handletextpad=0.2)
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
            
    if any([f==0,f==2]):
        ax.annotate(r'\textbf{%s}' % (osnames[f]),xy=(2015,0),xytext=(-0.25,0.50),
                  textcoords='axes fraction',color='k',fontsize=18,
                  rotation=90,ha='center',va='center')
    
    plt.xticks(np.arange(2015,2100+1,10),map(str,np.arange(2015,2100+1,10)),size=7.5)
    plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=7.5)
    plt.xlim([2015,2100])   
    plt.ylim([0,1.0])  

    plt.text(2015,1,r'\textbf{[%s]}' % letters[f],fontsize=9,color='k')       
    
    if any([f==0,f==1,f==2,f==3]):
        plt.ylabel(r'\textbf{Confidence}',color='dimgrey',fontsize=9,labelpad=8)
        
    if any([f==0,f==1]):
        plt.title(r'\textbf{%s}' % allvariables[f],fontsize=18,color='k',y=1.1)

# plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'MSFigure_ANN_Confidence_5Classes.png',dpi=600)
