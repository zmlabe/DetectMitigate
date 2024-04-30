"""
Second version of classification problem for emission scenario, but now for 
looping hyperparameters and plotting the skill metrics

Author     : Zachary M. Labe
Date       : 18 September 2023
Version    : 1 binary for ssp245 or ssp119, 2 - added capability for all XAI of OS ensemble members
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np
import scipy.stats as sts
import calc_Utilities as UT

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

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

### Other information
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
dataset_obs = 'ERA5_MEDS'
timeper = ['futureforcing','futureforcing']
scenarioall = ['SSP119','SSP245']
num_of_class = len(scenarioall)
lenOfPicks = len(modelGCMs)
allDataLabels = modelGCMs
monthlychoice = 'annual'
variq = 'PRECT'
reg_name = 'Globe'
lat_bounds,lon_bounds = UT.regions(reg_name)
level = 'surface'

### Hyperparamters for files of the ANN model
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = False
COUNTER = 5
hiddenalltry = [[5],[20],[30],[100],[5,5],[20,20],[30,30],[100,100],[5,5,5],[20,20,20],[30,30,30],[100,100,100],[5,5,5,5],[20,20,20,20],[30,30,30,30],[100,100,100,100]]
ridgePenaltyall = [0.001,0.01,0.05,0.1,0.2,0.5,1,5]
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","o","p","q"]
combinations = COUNTER * len(hiddenalltry) * len(ridgePenaltyall)

### Directories
directoryoutputo = '/work/Zachary.Labe/Research/DetectMitigate/Data/LoopHyperparametersBinary_ssp_119_245/'
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'

### Read in hyperparameters
acc_allv = []
best_f1 = []
for hh in range(len(hiddenalltry)):
    version = hh
    hiddenall = hiddenalltry[version]
    hiddenallstr = str(len(hiddenalltry[version])) + 'x' + str(hiddenalltry[version][0])
    
    acc_l = []
    for rp in range(len(ridgePenaltyall)):
        
        acc_c = []
        for seee in range(COUNTER):
            
            ### Model paramaters
            fac = 0.80 
            random_segment_seed = None
            random_network_seed = None
            hidden = hiddenall
            n_epochs = 1500
            batch_size = 128
            lr_here = 0.0001
            ridgePenalty = ridgePenaltyall[rp]
            actFun = 'relu'
            
            ### Read in scores
            hiddenfolder = 'hidden_' + hiddenallstr + '/'
            directoryoutputs = directoryoutputo + hiddenfolder + 'Scores/'
            savename = 'ANNv2_EmissionScenarioBinary_ssp_119-245_' + variq + '_' + reg_name + '_' + monthlychoice + '_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) + '_COUNTER' + str(seee+1) 
            
            scoreModel = np.load(directoryoutputs + savename + '_SCORES_LoopHyper.npz')
            accvalq = scoreModel['accval'] * 100.
            precvalq = scoreModel['precval'][:] * 100.
            recallvalq = scoreModel['recallval'][:] * 100.
            f1valq = scoreModel['f1_val'][:] * 100.
            
            acc_c.append(accvalq)
            
            if variq == 'T2M':
                if hh == 5:
                    if rp == 2:
                        best_f1.append(f1valq)
            elif variq == 'PRECT':
                if hh == 11:
                    if rp == 2:
                        best_f1.append(f1valq)
            
        acc_l.append(acc_c)
    acc_allv.append(acc_l)
    
### Create arrays of scores
acc_all = np.asarray(acc_allv)

###############################################################################
###############################################################################
###############################################################################
### Graph for accuracy
labels = [r'\textbf{1-LAYER$_{5}$}',r'\textbf{1-LAYER$_{20}$}',r'\textbf{1-LAYER$_{30}$}',r'\textbf{1-LAYER$_{100}$}',
          r'\textbf{2-LAYERS$_{5}$}',r'\textbf{2-LAYERS$_{20}$}',r'\textbf{2-LAYERS$_{30}$}',r'\textbf{2-LAYERS$_{100}$}',
          r'\textbf{3-LAYERS$_{5}$}',r'\textbf{3-LAYERS$_{20}$}',r'\textbf{3-LAYERS$_{30}$}',r'\textbf{3-LAYERS$_{100}$}',
          r'\textbf{4-LAYERS$_{5}$}',r'\textbf{4-LAYERS$_{20}$}',r'\textbf{4-LAYERS$_{30}$}',r'\textbf{4-LAYERS$_{100}$}']

fig = plt.figure(figsize=(9,6))
for plo in range(len(hiddenalltry)):
    ax = plt.subplot(4,4,plo+1)
    
    plotdata = acc_all[plo,:,:].transpose()
    
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.tick_params(axis="x",which="both",bottom = False,top=False,
                    labelbottom=False)
    
    def set_box_color(bp, color):
        plt.setp(bp['boxes'],color='w',alpha=0)
        plt.setp(bp['whiskers'], color='w',alpha=0)
        plt.setp(bp['caps'], color='w',alpha=0)
        plt.setp(bp['medians'], color=color,linewidth=2)
        
    ### Mask nans for boxplot
    # datamask = plotdata.copy()
    # mask = ~np.isnan(datamask)
    # filtered_data = [d[m] for d, m in zip(datamask.T, mask.T)]
    
    positionsq = np.arange(len(ridgePenaltyall))
    bpl = plt.boxplot(plotdata,positions=positionsq,widths=0.6,
                      patch_artist=True,sym='')
    
    # Modify boxes
    cp= 'teal'
    set_box_color(bpl,cp)
    plt.plot([], c=cp, label=r'\textbf{Accuracy [\%]}',clip_on=False)
        
    for i in range(plotdata.shape[1]):
        y = plotdata[:,i]
        x = np.random.normal(positionsq[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkred', alpha=0.7,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
     
    if any([plo==0,plo==4,plo==8,plo==12]):
        plt.yticks(np.arange(0,101,5),list(map(str,np.round(np.arange(0,101,5),2))),
                    fontsize=6) 
        plt.ylim([80,100])
    else:
        plt.yticks(np.arange(0,101,5),list(map(str,np.round(np.arange(0,101,5),2))),
                    fontsize=6) 
        plt.ylim([80,100])
        ax.axes.yaxis.set_ticklabels([])

    if any([plo==12,plo==13,plo==14,plo==15]):
        plt.text(-0.25,77,r'\textbf{%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(0.75,77,r'\textbf{%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(1.78,77,r'\textbf{%s}' % ridgePenaltyall[2],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(2.8,77,r'\textbf{%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(3.8,77,r'\textbf{%s}' % ridgePenaltyall[4],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(4.9,77,r'\textbf{%s}' % ridgePenaltyall[5],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(5.9,77,r'\textbf{%s}' % ridgePenaltyall[6],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(6.9,77,r'\textbf{%s}' % ridgePenaltyall[7],fontsize=5,color='dimgrey',
                  ha='left',va='center')
  
    ax.yaxis.grid(zorder=100,color='darkgrey',alpha=0.7,clip_on=False,linewidth=0.5)
    plt.title(r'%s' % labels[plo],fontsize=11,color='dimgrey')
    plt.text(-0.5,101,r'\textbf{[%s]}' % letters[plo],color='k',fontsize=6)
    
    if any([plo==0,plo==4,plo==8,plo==12]):
        plt.ylabel(r'\textbf{Accuracy [\%]}',color='k',fontsize=7)
        
    if variq == 'T2M':
        variqname = 'Temperature'
    elif variq == 'PRECT':
        variqname = 'Precipitation'
        
plt.tight_layout()
plt.subplots_adjust(hspace=0.7,bottom=0.08)
plt.text(-17,71,r'\textbf{Ridge Regularization [L$_{2}$] -- SSP119 or SSP245 for %s}' % variqname,fontsize=8,color='k',
         ha='left',va='center')  
plt.savefig(directoryfigure + 'MSFigure_ANN_LoopHyperparameters_ssp_119_245_%s.png' % (variq),dpi=600)

