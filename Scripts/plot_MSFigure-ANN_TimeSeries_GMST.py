"""
Plot GMST for the different SPEAR emission scenarios

Author     : Zachary M. Labe
Date       : 5 January 2022
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
import calc_Stats as dSS
import read_BEST as B
from scipy.interpolate import griddata as g
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Directories
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/MSFigures_ANN/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_NATURAL','SPEAR_MED',
             'SPEAR_MED_Scenario','SPEAR_MED_Scenario','SPEAR_MED_Scenario',
             'SPEAR_MED_SSP534OS_10ye']
dataset_obs = 'ERA5_MEDS'
lenOfPicks = len(modelGCMs)
monthlychoice = 'annual'
allvariables = ['T2M','PRECT']
reg_name = 'Globe'
level = 'surface'
###############################################################################
###############################################################################
timeper = ['historicalforcing','futureforcing','futureforcing','futureforcing','futureforcing','futureforcing','futureforcing']
scenarioall = ['historical','natural','SSP585','SSP119','SSP245','SSP534OS','SSP534OS_10ye']
scenarioallnames = ['Historical','Natural','SSP5-8.5','SSP1-1.9','SSP2-4.5','SSP5-3.4OS','SSP5-3.34OS_10ye']
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
yearsall = [np.arange(1929+window,2014+1,1),np.arange(2015+window,2100+1,1),
            np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1),
            np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1),
            np.arange(2015+window,2100+1,1)]
yearsobs = np.arange(1929+window,2021+1,1)
###############################################################################
###############################################################################
numOfEns = 30
numOfEns_10ye = 30
lentime = len(yearsall)
###############################################################################
###############################################################################
lat_bounds,lon_bounds = UT.regions(reg_name)
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
### Adjust axes in time series plots 
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

fig = plt.figure(figsize=(10,4))
for ii in range(len(allvariables)):
    variq = allvariables[ii]
    
    ### Loop in all climate models
    data_all = []
    for no in range(len(modelGCMs)):
        dataset = modelGCMs[no]
        scenario = scenarioall[no]
        data_allq,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
        data_all.append(data_allq)
    data = np.asarray(data_all)
    
    ### Calculate historical baseline for calculating anomalies (and ensemble mean)
    historical = data[0]
    historicalyrs = yearsall[0]
    
    yearhq = np.where((historicalyrs >= baseline.min()) & (historicalyrs <= baseline.max()))[0]
    historicalc = np.nanmean(np.nanmean(historical[:,yearhq,:,:],axis=1),axis=0)
    
    ### Calculate anomalies
    data_anom = []
    for no in range(len(modelGCMs)):
        if no == 1:
            naturalhistorical,lats,lons = read_primary_dataset(variq,'SPEAR_MED_NATURAL_Historical',monthlychoice,scenario,lat_bounds,lon_bounds)
            historicalN = np.nanmean(np.nanmean(naturalhistorical[:,yearhq,:,:],axis=1),axis=0)
            anomq = data[no] - historicalN[np.newaxis,np.newaxis,:,:]
        else:
            anomq = data[no] - historicalc[np.newaxis,np.newaxis,:,:]
        data_anom.append(anomq)
    
    ### Calculate global average
    lon2,lat2 = np.meshgrid(lons,lats)
    aveall = []
    maxens = []
    minens = []
    meanens = []
    medianens = []
    for no in range(len(modelGCMs)):
        aveallq = UT.calc_weightedAve(data_anom[no],lat2)
    
        maxensq = np.nanmax(aveallq,axis=0)
        minensq = np.nanmin(aveallq,axis=0)
        meanensq = np.nanmean(aveallq,axis=0)
        medianensq = np.nanmedian(aveallq,axis=0)
        
        aveall.append(aveallq)
        maxens.append(maxensq)
        minens.append(minensq)
        meanens.append(meanensq)
        medianens.append(medianensq)

    ###############################################################################
    ###############################################################################
    ###############################################################################               
    ### Plot Figure      
    
    if ii == 0:
        ax = plt.subplot(121)
        
        adjust_spines(ax, ['left', 'bottom'])            
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_color('dimgrey')
        ax.spines['left'].set_color('dimgrey')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.tick_params('both',length=4.,width=2,which='major',color='dimgrey')
        ax.tick_params(axis='x',labelsize=6,pad=1.5)
        ax.tick_params(axis='y',labelsize=6,pad=1.5)
        ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.4)
        
        color = cmr.rainforest(np.linspace(0.00,0.85,len(aveall)))
        for i,c in zip(range(len(aveall)),color): 
            plt.fill_between(x=yearsall[i],y1=minens[i],y2=maxens[i],facecolor=c,zorder=1,
                     alpha=0.4,edgecolor='none',clip_on=False)
            plt.plot(yearsall[i],meanens[i],linestyle='-',linewidth=2,color=c,
                     label=r'\textbf{%s}' % scenarioallnames[i],zorder=2,clip_on=False)
            
            if i == 5:
                plt.axvline(x=2040,color='k',linewidth=2,linestyle='--',dashes=(1,0.3),zorder=100)
                plt.scatter(yearsall[i][np.argmax(meanens[i])],np.nanmax(meanens[i]),s=20,color='k',zorder=200)
                plt.text(yearsall[i][np.argmax(meanens[i])]+1.8,np.nanmax(meanens[i]),r'\textbf{%s}' % yearsall[i][np.argmax(meanens[i])],fontsize=7.5,color='k')
            elif i == 6:
                plt.axvline(x=2031,color='dimgrey',linewidth=2,linestyle='--',dashes=(1,0.3),zorder=100)
                plt.scatter(yearsall[i][np.argmax(meanens[i])],np.nanmax(meanens[i]),s=20,color='dimgrey',zorder=200)
                plt.text(yearsall[i][np.argmax(meanens[i])]+1.8,np.nanmax(meanens[i]),r'\textbf{%s}' % yearsall[i][np.argmax(meanens[i])],fontsize=7.5,color='dimgrey')
        
        leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
              bbox_to_anchor=(0.17,0.79),fancybox=True,ncol=1,frameon=False,
              handlelength=1,handletextpad=0.5)
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        
        plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
        plt.yticks(np.round(np.arange(-18,18.1,0.5),2),np.round(np.arange(-18,18.1,0.5),2))
        plt.xlim([1930,2100])
        plt.ylim([-0.5,5])
        
        plt.text(1930,5,r'\textbf{[a]}',fontsize=11,color='k')
        plt.text(1930,4.5,r'\textbf{TEMPERATURE}',fontsize=16,color='dimgrey')
        plt.xlabel(r'\textbf{Years}',fontsize=9,color='k')
        plt.ylabel(r'\textbf{Anomaly [$^{\circ}$C] Relative to 1951-1980}',
                   fontsize=9,color='k')
     
    elif ii == 1:
        ax = plt.subplot(122)
        
        adjust_spines(ax, ['left', 'bottom'])            
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_color('dimgrey')
        ax.spines['left'].set_color('dimgrey')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.tick_params('both',length=4.,width=2,which='major',color='dimgrey')
        ax.tick_params(axis='x',labelsize=6,pad=1.5)
        ax.tick_params(axis='y',labelsize=6,pad=1.5)
        ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.4)
        
        color = cmr.rainforest(np.linspace(0.00,0.85,len(aveall)))
        for i,c in zip(range(len(aveall)),color): 
            plt.fill_between(x=yearsall[i],y1=minens[i],y2=maxens[i],facecolor=c,zorder=1,
                      alpha=0.4,edgecolor='none',clip_on=False)
            plt.plot(yearsall[i],meanens[i],linestyle='-',linewidth=2,color=c,
                      label=r'\textbf{%s}' % scenarioallnames[i],zorder=2,clip_on=False)
            
            if i == 5:
                plt.axvline(x=2040,color='k',linewidth=2,linestyle='--',dashes=(1,0.3),zorder=100)
                plt.scatter(yearsall[i][np.argmax(meanens[i])],np.nanmax(meanens[i]),s=20,color='k',zorder=200)
                plt.text(yearsall[i][np.argmax(meanens[i])]+1.8,np.nanmax(meanens[i]),r'\textbf{%s}' % yearsall[i][np.argmax(meanens[i])],fontsize=7.5,color='k')
            elif i == 6:
                plt.axvline(x=2031,color='dimgrey',linewidth=2,linestyle='--',dashes=(1,0.3),zorder=100)
                plt.scatter(yearsall[i][np.argmax(meanens[i])],np.nanmax(meanens[i]),s=20,color='dimgrey',zorder=200)
                plt.text(yearsall[i][np.argmax(meanens[i])]+1.8,np.nanmax(meanens[i]),r'\textbf{%s}' % yearsall[i][np.argmax(meanens[i])],fontsize=7.5,color='dimgrey')
        
        
        # leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
        #       bbox_to_anchor=(0.17,0.74),fancybox=True,ncol=1,frameon=False,
        #       handlelength=1,handletextpad=0.5)
        # for line,text in zip(leg.get_lines(), leg.get_texts()):
        #     text.set_color(line.get_color())
        
        plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
        plt.yticks(np.round(np.arange(-0.5,0.51,0.05),2),np.round(np.arange(-0.5,0.51,0.05),2))
        plt.xlim([1930,2100])
        plt.ylim([-0.05,0.25])
        
        plt.text(1930,0.25,r'\textbf{[b]}',fontsize=11,color='k')
        plt.text(1930,0.222,r'\textbf{PRECIPITATION}',fontsize=16,color='dimgrey')
        plt.xlabel(r'\textbf{Years}',fontsize=9,color='k')
        plt.ylabel(r'\textbf{Anomaly [mm/day] Relative to 1951-1980}',
                   fontsize=9,color='k')
    
plt.tight_layout()
plt.savefig(directoryfigure + 'MSFigure_ANN_TimeSeries_GMST_%s.png' % monthlychoice,dpi=300)
