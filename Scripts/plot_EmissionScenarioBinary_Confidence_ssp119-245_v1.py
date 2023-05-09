"""
First version of classification problem for emission scenario

Author     : Zachary M. Labe
Date       : 13 February 2023
Version    : 1 binary for ssp245 or ssp119
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
directorydata = '/work/Zachary.Labe/Research/DetectMitigate/Data/BinaryChoice/'
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
variq = 'PRECT'
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
dataset_inference = True
timeper = ['futureforcing','futureforcing']
scenarioall = ['SSP119','SSP245']
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
hidden = [30,30,30]
n_epochs = 1500
batch_size = 128
lr_here = 0.0001
ridgePenalty = 0.1
actFun = 'relu'
random_segment_seed = 71541
random_network_seed = 87750
savename = 'ANNv2_EmissionScenarioBinary_ssp_119-245_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 

### Import confidence
confidence = np.load(directorydata + 'overshootPredictedConfidence_' + savename+ '.npz')['overshootconf']
meanconf = np.nanmean(confidence,axis=0)

### Import labels
labelsall = np.genfromtxt(directorydata + 'overshootPredictedLabels_' + savename+ '.txt')
label = sts.mode(labelsall,axis=0)[0][0]

### Count frequency 
freq0 = []
freq1 = []
for i in range(len(years)):
    count0 = np.count_nonzero(labelsall[:,i] == 0)
    count1 = np.count_nonzero(labelsall[:,i] == 1)
    
    freq0.append(count0)
    freq1.append(count1)
    
allfreq = np.vstack([freq0,freq1])
        
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
    plt.plot(years,meanconf[:,i],color=c,linewidth=0.3,
                label=r'\textbf{%s}' % scenarioall[i],zorder=11,
                clip_on=False,alpha=1)
    plt.scatter(years,meanconf[:,i],color=c,s=28,zorder=12,
                clip_on=False,alpha=0.2,edgecolors='none')
    
    for yr in range(years.shape[0]):
        la = label[yr]
        if i == la:
            plt.scatter(years[yr],meanconf[yr,i],color=c,s=28,zorder=12,
                        clip_on=False,alpha=1,edgecolors='none')
        

leg = plt.legend(shadow=False,fontsize=7,loc='upper center',
              bbox_to_anchor=(0.5,-0.08),fancybox=True,ncol=5,frameon=False,
              handlelength=0,handletextpad=0)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(2015,2100+1,10),map(str,np.arange(2015,2100+1,10)),size=5.45)
plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=6)
plt.xlim([2015,2100])   
plt.ylim([0,1.0])           

plt.ylabel(r'\textbf{Confidence [%s], L$_{2}$=%s' % (monthlychoice,ridgePenalty),color='dimgrey',fontsize=8,labelpad=8)
plt.title(r'\textbf{ANN CONFIDENCE OF SSP534OS -- %s - %s}' % (variq,reg_name),color='dimgrey',fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + '%s_Confidence.png' % (savename),dpi=300)

###############################################################################
###############################################################################
###############################################################################
###############################################################################                      
### Plot first meshgrid
fig = plt.figure(figsize=(10,4))

ax = plt.subplot(111)
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

csm=plt.get_cmap(cmocean.cm.ice_r)
norm = cccc.BoundaryNorm(np.arange(1,30+1,1),csm.N)

cs = plt.pcolormesh(allfreq,shading='faceted',edgecolor='darkgrey',
                    linewidth=0.01,norm=norm,cmap=csm,clip_on=False)

plt.xticks(np.arange(0.5,87.5,5),np.arange(2015,2101,5),ha='center',va='center',color='k',size=6)
plt.yticks(np.arange(0.5,2.5,1),scenarioall,ha='right',va='center',color='k',size=6)
yax = ax.get_yaxis()
yax.set_tick_params(pad=2)
plt.xlim([0,86])

for i in range(allfreq.shape[0]):
    for j in range(allfreq.shape[1]):
        cc = 'crimson'         
        plt.text(j+0.5,i+0.5,r'\textbf{%s}' % int(allfreq[i,j]),fontsize=4,
            color=cc,va='center',ha='center')

###############################################################################                
cbar_ax1 = fig.add_axes([0.35,0.11,0.3,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='neither',extendfrac=0.07,drawedges=True)
cbar.set_ticks([])
cbar.set_ticklabels([])  
cbar.ax.tick_params(axis='x', size=.001,labelsize=7)
cbar.dividers.set_color('dimgrey')
cbar.dividers.set_linewidth(1)
cbar.outline.set_edgecolor('dimgrey')
cbar.outline.set_linewidth(1)
cbar.set_label(r'\textbf{NUMBER OF ENSEMBLE MEMBERS}',color='k',labelpad=7,fontsize=18)

cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(range(1,30,1)):
    cbar.ax.text((2 * j+2.9)/2 + 0.09, 0.37, lab,ha='center',va='center',
                 size=5,color='crimson')

plt.tight_layout()
plt.subplots_adjust(bottom=0.2,hspace=0.1)
plt.savefig(directoryfigure + '%s_NumberOfEnsChoices.png' % (savename),dpi=300)
