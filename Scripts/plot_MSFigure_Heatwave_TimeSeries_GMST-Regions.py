"""
Plot GMST for the different SPEAR emission scenarios

Author     : Zachary M. Labe
Date       : 25 October 2023
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
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/MSFigures_Heat/'
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/MitigateHeat/'

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['SPEAR_MED_Historical','SPEAR_MED_Scenario','SPEAR_MED','SPEAR_MED_Scenario','SPEAR_MED_SSP534OS_10ye']
dataset_obs = 'ERA5_MEDS'
lenOfPicks = len(modelGCMs)
monthlychoice = 'annual'
allvariables = ['T2M','T2M']
allregions = ['Globe','US']
level = 'surface'
###############################################################################
###############################################################################
timeper = ['historicalforcing','futureforcing','futureforcing','futureforcing','futureforcing']
scenarioall = ['historical','SSP245','SSP585','SSP534OS','SSP534OS_10ye']
scenarioallnames = ['Historical','SSP2-4.5','SSP5-8.5','SSP5-3.4OS','SSP5-3.34OS_10ye']
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
baseline = np.arange(1981,2010+1,1)
baselinePI = np.arange(1921,1950+1,1)
###############################################################################
###############################################################################
window = 0
yearsall = [np.arange(1921+window,2014+1,1),np.arange(2015+window,2100+1,1),
            np.arange(2015+window,2100+1,1),np.arange(2015+window,2100+1,1),
            np.arange(2015+window,2100+1,1)]
yearsobs = np.arange(1921+window,2021+1,1)
###############################################################################
###############################################################################
numOfEns = 30
numOfEns_10ye = 30
lentime = len(yearsall)
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

fig = plt.figure(figsize=(7,9))
for ii in range(len(allregions)):
    variq = allvariables[ii]
    
    ### Loop in all climate models
    data_all = []
    for no in range(len(modelGCMs)):
        reg_name = allregions[ii]
        dataset = modelGCMs[no]
        scenario = scenarioall[no]
        lat_bounds,lon_bounds = UT.regions(reg_name)
        data_allq,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
        
        if reg_name == 'US':
            data_allq, data_obs = dSS.mask_CONUS(data_allq,np.full((data_allq.shape[1],lats.shape[0],lons.shape[0]),np.nan),'MEDS',lat_bounds,lon_bounds)
            print('*Removed everything by CONUS*')
        
        data_all.append(data_allq)
    data = np.asarray(data_all)
    
    ### Calculate historical baseline for calculating anomalies (and ensemble mean)
    spear_h,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    if reg_name == 'US':
        spear_h, data_obsh = dSS.mask_CONUS(spear_h,np.full((data_allq.shape[1],lats.shape[0],lons.shape[0]),np.nan),'MEDS',lat_bounds,lon_bounds)
        print('*Removed everything by CONUS*')
    historical = spear_h
    historicalyrs = yearsall[0]
    
    if reg_name == 'US':
        yearhq = np.where((historicalyrs >= baseline.min()) & (historicalyrs <= baseline.max()))[0]
        historicalc = np.nanmean(np.nanmean(historical[:,yearhq,:,:],axis=1),axis=0)
    else:
        yearhq = np.where((historicalyrs >= baselinePI.min()) & (historicalyrs <= baselinePI.max()))[0]
        historicalc = np.nanmean(np.nanmean(historical[:,yearhq,:,:],axis=1),axis=0)        
    
    ### Calculate anomalies
    data_anom = []
    for no in range(len(modelGCMs)):
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
        
    ### Plot beginning in 2031
    yearq = np.where(yearsall[1] >= 2031)[0]
    
    ###############################################################################
    ###############################################################################
    ###############################################################################               
    ### Plot Figure      
    
    if ii == 0:
        ax = plt.subplot(211)
        
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
        
        plt.axhline(y=1.5,color='dimgrey',linestyle='--',linewidth=1,clip_on=False,alpha=1)
        plt.axhline(y=1.6,color='dimgrey',linestyle='--',linewidth=1,clip_on=False,alpha=1)
        plt.axhline(y=1.7,color='dimgrey',linestyle='--',linewidth=1,clip_on=False,alpha=1)
        plt.axhline(y=1.8,color='dimgrey',linestyle='--',linewidth=1,clip_on=False,alpha=1)
        plt.axhline(y=1.9,color='dimgrey',linestyle='--',linewidth=1,clip_on=False,alpha=1)
        plt.axhline(y=2,color='dimgrey',linestyle='--',linewidth=1,clip_on=False,alpha=1)
        
        color = ['darkorange','maroon','darkslategrey','lightseagreen']
        for i,c in zip(range(1,len(aveall)),color): 
            # if i == 3:
            #     plt.fill_between(x=yearsall[i][yearq],y1=minens[i][yearq],y2=maxens[i][yearq],facecolor=c,zorder=1,
            #              alpha=0.4,edgecolor='none',clip_on=False)
            #     plt.plot(yearsall[i][yearq],meanens[i][yearq],linestyle='--',dashes=(1,0.3),linewidth=4,color=c,
            #              label=r'\textbf{%s}' % scenarioallnames[i],zorder=2,clip_on=False)
            if i == 1:
                plt.fill_between(x=yearsall[i][yearq],y1=minens[i][yearq],y2=maxens[i][yearq],facecolor=c,zorder=1,
                         alpha=0.4,edgecolor='none',clip_on=False)
                plt.plot(yearsall[i],meanens[i],linestyle='--',dashes=(1,0.3),linewidth=4,color=c,
                         label=r'\textbf{%s}' % scenarioallnames[i],zorder=2,clip_on=False)
            else:
                plt.fill_between(x=yearsall[i],y1=minens[i],y2=maxens[i],facecolor=c,zorder=1,
                         alpha=0.4,edgecolor='none',clip_on=False)
                plt.plot(yearsall[i],meanens[i],linestyle='-',linewidth=4,color=c,
                         label=r'\textbf{%s}' % scenarioallnames[i],zorder=6,clip_on=False)
            
            if i == 3:
                plt.axvline(x=2040,color=c,linewidth=1,linestyle=':',zorder=100)
                plt.axvline(x=yearsall[i][np.argmax(meanens[i])],color=c,linewidth=2,linestyle='-',zorder=200)
            elif i == 4:
                plt.axvline(x=2031,color=c,linewidth=1,linestyle=':',zorder=100)
                plt.axvline(x=yearsall[i][np.argmax(meanens[i])],color=c,linewidth=2,linestyle='-',zorder=200)
                
            if i == 3:
                plt.text(yearsall[i][np.argmax(meanens[i])],5.5,r'\textbf{Max [OS]}',fontsize=7,color=c,ha='center')
                np.savetxt(directoryoutput + 'Max_GMST_SSP534OS_Annual.txt',[yearsall[i][np.argmax(meanens[i])],np.nanmax(meanens[i]),np.argmax(meanens[i])])
            elif i == 4:
                plt.text(yearsall[i][np.argmax(meanens[i])],5.5,r'\textbf{Max [OS_10ye]}',fontsize=7,color=c,ha='center')
                np.savetxt(directoryoutput + 'Max_GMST_SSP534OS_10ye_Annual.txt',[yearsall[i][np.argmax(meanens[i])],np.nanmax(meanens[i]),np.argmax(meanens[i])])
            
        # leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
        #       bbox_to_anchor=(0.17,0.84),fancybox=True,ncol=1,frameon=False,
        #       handlelength=1,handletextpad=0.5)
        # for line,text in zip(leg.get_lines(), leg.get_texts()):
        #     text.set_color(line.get_color())
        
        plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=9.2)
        plt.yticks(np.round(np.arange(-18,18.1,0.5),2),np.round(np.arange(-18,18.1,0.5),2),fontsize=9.2)
        plt.xlim([2015,2100])
        plt.ylim([0.5,5.5])
        
        plt.text(2015,5.5,r'\textbf{[d]}',fontsize=11,color='k')
        plt.text(2031,5.5,r'\textbf{OS_10ye}',fontsize=7,color='lightseagreen',ha='center')
        plt.text(2040,5.5,r'\textbf{OS}',fontsize=7,color='darkslategrey',ha='center')
        plt.text(2015,5,r'\textbf{GLOBAL}',fontsize=16,color='dimgrey')
        plt.ylabel(r'\textbf{Temperature Anomaly [$^{\circ}$C] Relative to 1921-1950}',
                   fontsize=10,color='dimgrey')
        
    elif ii == 1:
        ax = plt.subplot(212)
        
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
        
        color = ['darkorange','maroon','darkslategrey','lightseagreen']
        for i,c in zip(range(1,len(aveall)),color): 
            # if i == 3:
            #     plt.fill_between(x=yearsall[i][yearq],y1=minens[i][yearq],y2=maxens[i][yearq],facecolor=c,zorder=1,
            #              alpha=0.4,edgecolor='none',clip_on=False)
            #     plt.plot(yearsall[i][yearq],meanens[i][yearq],linestyle='--',dashes=(1,0.3),linewidth=4,color=c,
            #              label=r'\textbf{%s}' % scenarioallnames[i],zorder=2,clip_on=False)
            if i == 1:
                plt.fill_between(x=yearsall[i][yearq],y1=minens[i][yearq],y2=maxens[i][yearq],facecolor=c,zorder=1,
                         alpha=0.4,edgecolor='none',clip_on=False)
                plt.plot(yearsall[i],meanens[i],linestyle='--',dashes=(1,0.3),linewidth=4,color=c,
                         label=r'\textbf{%s}' % scenarioallnames[i],zorder=2,clip_on=False)
            else:
                plt.fill_between(x=yearsall[i],y1=minens[i],y2=maxens[i],facecolor=c,zorder=1,
                         alpha=0.4,edgecolor='none',clip_on=False)
                plt.plot(yearsall[i],meanens[i],linestyle='-',linewidth=4,color=c,
                         label=r'\textbf{%s}' % scenarioallnames[i],zorder=6,clip_on=False)
            
            if i == 3:
                plt.axvline(x=2040,color=c,linewidth=1,linestyle=':',zorder=100)
                plt.axvline(x=yearsall[i][np.argmax(meanens[i])],color=c,linewidth=2,linestyle='-',zorder=200)
            elif i == 4:
                plt.axvline(x=2031,color=c,linewidth=1,linestyle=':',zorder=100)
                plt.axvline(x=yearsall[i][np.argmax(meanens[i])],color=c,linewidth=2,linestyle='-',zorder=200)
               
        
        # leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
        #       bbox_to_anchor=(0.17,0.84),fancybox=True,ncol=1,frameon=False,
        #       handlelength=1,handletextpad=0.5)
        # for line,text in zip(leg.get_lines(), leg.get_texts()):
        #     text.set_color(line.get_color())
        
        plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=9.2)
        plt.yticks(np.round(np.arange(-18,18.1,0.5),2),np.round(np.arange(-18,18.1,0.5),2),fontsize=9.2)
        plt.xlim([2015,2100])
        plt.ylim([-0.5,7.5])
        
        plt.text(2015,7.5,r'\textbf{[e]}',fontsize=11,color='k')
        plt.text(2015,7,r'\textbf{CONUS}',fontsize=16,color='dimgrey')
        plt.ylabel(r'\textbf{Temperature Anomaly [$^{\circ}$C] Relative to 1921-1950}',
                   fontsize=10,color='dimgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'MSFigure_Heatwave_TimeSeries_GMST-Regions_%s.png' % monthlychoice,dpi=300)
