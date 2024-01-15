"""
First version of classification problem for emission scenario

Author     : Zachary M. Labe
Date       : 3 January 2023
Version    : 1 (mostly for testing; now using validation data)
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
### Data preliminaries 
modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_NATURAL','SPEAR_MED','SPEAR_MED_Scenario','SPEAR_MED_Scenario']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
dataset_obs = 'ERA5_MEDS'
lenOfPicks = len(modelGCMs)
allDataLabels = modelGCMs
monthlychoice = 'annual'
variqa = ['T2M','PRECT','T2M','PRECT']
variqnames = ['Temperature','Precipitation','Temperature','Precipitation']
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
dataset_inference = True
timeper = ['historicalforcing','futureforcing','futureforcing','futureforcing','futureforcing']
scenarioall = ['historical','natural','SSP585','SSP119','SSP245']
scenarioallnames = ['Historical','Natural','SSP5-8.5','SSP1-1.9','SSP2-4.5']
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

fig = plt.figure(figsize=(10,3.5))
for f in range(len(variqa)):
    variq = variqa[f]
    
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
        
    random_segment_seed = 71541
    random_network_seed = 87750
    savename = 'ANNv4_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
    
    ### Read in which overshoot scenario it is from
    if any([f==0,f==1]):
        ### Import confidence
        confidence = np.load(directorydata + 'overshootPredictedConfidence_' + savename+ '.npz')['overshootconf']
        meanconf = np.nanmean(confidence,axis=0)
        
        ### Import labels
        labelsall = np.genfromtxt(directorydata + 'overshootPredictedLabels_' + savename+ '.txt')
    else:
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
        
    ### ['historical','natural','SSP585','SSP119','SSP245']
    ### [0,1,2,3,4]
    ### better sort [1,0,2,4,3]
    scenarioallnames = ['Natural','Historical','SSP5-8.5','SSP2-4.5','SSP1-1.9']
    allfreq = np.vstack([freq1,freq0,freq2,freq4,freq3])
    
    ###############################################################################
    ###############################################################################
    ###############################################################################                      
    ### Plot first meshgrid
    
    ax = plt.subplot(2,2,f+1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.get_xaxis().set_tick_params(direction='out', width=2,length=3,
                color='darkgrey')
    ax.get_yaxis().set_tick_params(direction='out', width=2,length=3,
                color='darkgrey')
    
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='on',      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='on',      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        labelleft='on')
    
    csm=plt.get_cmap(cmr.freeze_r)
    norm = cccc.BoundaryNorm(np.arange(0,31+1,1),csm.N)
    
    cs = plt.pcolormesh(allfreq,shading='faceted',edgecolor='darkgrey',
                        linewidth=0.01,norm=norm,cmap=csm,clip_on=False,alpha=0.8)
    
    plt.xticks(np.arange(0.5,87.5,5),np.arange(2015,2101,5),ha='center',va='center',color='k',size=7)
    if any([f==0,f==2]):
        plt.yticks(np.arange(0.5,5.5,1),scenarioallnames,ha='right',va='center',color='k',size=7)
        yax = ax.get_yaxis()
        yax.set_tick_params(pad=2)
    else:
        plt.yticks(np.arange(0.5,5.5,1),[],ha='right',va='center',color='k',size=7)
    plt.xlim([0,86])
    xax = ax.get_xaxis()
    xax.set_tick_params(pad=6)
    
    if any([f==0,f==1]):
        plt.axvline(x=np.where(years==2040)[0]+0.5,linewidth=2,color='darkslategrey',zorder=100,linestyle='--',dashes=(1,0.3))
    elif any([f==2,f==3]):
        plt.axvline(x=np.where(years==2031)[0]+0.5,linewidth=2,color='lightseagreen',zorder=100,linestyle='--',dashes=(1,0.3))
       
    if f==0:
        plt.vlines(x=np.where(years==2053)[0],ymin=2,ymax=4,linewidth=1,color='r',zorder=100)
        plt.vlines(x=np.where(years==2063)[0],ymin=2,ymax=4,linewidth=1,color='r',zorder=100)
    elif f==1:
        plt.vlines(x=np.where(years==2050)[0],ymin=2,ymax=4,linewidth=1,color='r',zorder=100)
        plt.vlines(x=np.where(years==2064)[0],ymin=2,ymax=4,linewidth=1,color='r',zorder=100)
    elif f==2:
        plt.vlines(x=np.where(years==2049)[0],ymin=2,ymax=4,linewidth=1,color='r',zorder=100)
        plt.vlines(x=np.where(years==2056)[0],ymin=2,ymax=4,linewidth=1,color='r',zorder=100)
        plt.vlines(x=np.where(years==2074)[0],ymin=3,ymax=5,linewidth=1,color='r',zorder=100)
        plt.vlines(x=np.where(years==2084)[0],ymin=3,ymax=5,linewidth=1,color='r',zorder=100)
    elif f==3:
        plt.vlines(x=np.where(years==2046)[0],ymin=2,ymax=4,linewidth=1,color='r',zorder=100)
        plt.vlines(x=np.where(years==2051)[0],ymin=2,ymax=4,linewidth=1,color='r',zorder=100)
        plt.vlines(x=np.where(years==2073)[0],ymin=3,ymax=5,linewidth=1,color='r',zorder=100)
        plt.vlines(x=np.where(years==2086)[0],ymin=3,ymax=5,linewidth=1,color='r',zorder=100)
    
    if any([f==0,f==1]):
        plt.title(r'\textbf{[%s] SPEAR_MED_SSP5-3.34OS -- %s}' % (letters[f],variq),fontsize=13,color='k')
    else: 
        plt.title(r'\textbf{[%s] SPEAR_MED_SSP5-3.34OS_10ye -- %s}' % (letters[f],variq),fontsize=13,color='k')
    
    for i in range(allfreq.shape[0]):
        for j in range(allfreq.shape[1]):
            cc = 'crimson'         
            if int(allfreq[i,j]) >= ((4/5)*numOfEns):
                plt.text(j+0.63,i+0.5,r'$\bf{\bullet}$',fontsize=6,
                    color=cc,va='center',ha='center',zorder=200)
            elif int(allfreq[i,j]) > ((1/2)*numOfEns) and int(allfreq[i,j]) < ((4/5)*numOfEns):
                plt.text(j+0.63,i+0.5,r'$\bf{\circ}$',fontsize=6,
                    color=cc,va='center',ha='center',zorder=200)

###############################################################################                
cbar_ax1 = fig.add_axes([0.28,0.11,0.5,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='neither',extendfrac=0.07,drawedges=True)
cbar.set_ticks([])
cbar.set_ticklabels([])  
cbar.ax.tick_params(axis='x', size=.001,labelsize=7)
cbar.dividers.set_color('dimgrey')
cbar.dividers.set_linewidth(1)
cbar.outline.set_edgecolor('dimgrey')
cbar.outline.set_linewidth(1)
cbar.set_label(r'\textbf{Number of Ensemble Members}',color='k',labelpad=9,fontsize=13)

cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(range(0,31,1)):
    cbar.ax.text((2 * j+2.9)/2 -0.95, 0.4,r'\textbf{%s}' % lab,ha='center',va='center',
                 size=6,color='crimson',zorder=100)

plt.text(15+1.1,-0.55,r'$\bf{\circ \rightarrow}$',fontsize=10,color=cc,va='center',ha='center')
plt.text(25+1.1,-0.55,r'$\bf{\bullet \rightarrow}$',fontsize=10,color=cc,va='center',ha='center')
plt.tight_layout()
plt.subplots_adjust(bottom=0.2,hspace=0.55,wspace=0.04,right=0.99)
plt.savefig(directoryfigure + 'MSFigure_ANN_HeatMaps_OS_v2.png',dpi=600)
