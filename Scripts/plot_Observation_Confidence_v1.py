"""
First version of classification problem for emission scenario

Author     : Zachary M. Labe
Date       : 6 February 2023
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
directorydata = '/home/Zachary.Labe/Research/DetectMitigate/Data/'
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
variq = 'T2M'
reg_name = 'NAprop'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
dataset_inference = True
timeper = ['historicalforcing','futureforcing','futureforcing','futureforcing','futureforcing']
scenarioall = ['HISTORICAL','NATURAL','SSP585','SSP119','SSP245']
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
yearsobs = np.arange(1950+window,2021+1,1)
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
hidden = [30,30,30]
n_epochs = 1500
batch_size = 128
lr_here = 0.0001
ridgePenalty = 0.1
actFun = 'relu'
random_segment_seed = 71541
random_network_seed = 87750
savename = 'ANNv2_EmissionScenario_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 

### Import confidence
confidence = np.load(directorydata + 'observationsPredictedConfidence_' + savename+ '.npz')['obsconf']

### Import labels
label = np.genfromtxt(directorydata + 'observationsPredictedLabels_' + savename+ '.txt')
 
###############################################################################
###############################################################################
###############################################################################
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
            
### Begin plot
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
# ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.3)

color = cmr.infinity(np.linspace(0.00,0.8,len(scenarioall)))
for i,c in zip(range(len(scenarioall)),color):
    if i == 7:
        c = 'k'
    else:
        c = c
    plt.plot(yearsobs,confidence[:,i],color=c,linewidth=0.3,
                label=r'\textbf{%s}' % scenarioall[i],zorder=11,
                clip_on=False,alpha=1)
    plt.scatter(yearsobs,confidence[:,i],color=c,s=28,zorder=12,
                clip_on=False,alpha=0.2,edgecolors='none')
    
    for yr in range(yearsobs.shape[0]):
        la = label[yr]
        if i == la:
            plt.scatter(yearsobs[yr],confidence[yr,i],color=c,s=28,zorder=12,
                        clip_on=False,alpha=1,edgecolors='none')
        

leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
              bbox_to_anchor=(0.5,-0.08),fancybox=True,ncol=5,frameon=False,
              handlelength=0,handletextpad=0)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2100+1,10),map(str,np.arange(1920,2100+1,10)),size=5.45)
plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=6)
plt.xlim([1950,2021])   
plt.ylim([0,1.0])           

plt.ylabel(r'\textbf{Confidence [%s], L$_{2}$=%s' % (monthlychoice,ridgePenalty),color='dimgrey',fontsize=8,labelpad=8)
plt.title(r'\textbf{ANN CONFIDENCE OF OBSERVATIONS -- %s - %s}' % (variq,reg_name),color='dimgrey',fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + '%s_OBS_Confidence.png' % (savename),dpi=300)
