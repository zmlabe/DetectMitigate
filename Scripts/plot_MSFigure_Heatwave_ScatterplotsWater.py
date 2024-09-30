"""
Examine relationships with heatwaves and physical processes for MS for only
water-related variables

Author    : Zachary M. Labe
Date      : 27 August 2024
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import cmasher as cmr
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import sys
import scipy.stats as sts

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variqall = ['PRECT','EVAP','rh_ref']
variHEAT = 'TMAX'
modelpick = 'OS_10ye'
slicemonthnamen = ['JJA']
monthlychoice = slicemonthnamen[0]
monthlychoiceJJA = slicemonthnamen[0]
numOfEns = 30
numOfEns_10ye = 30
yearsh = np.arange(1921,2014+1,1)
years = np.arange(1921,2100+1)
yearsf = np.arange(2015,2100+1)

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

fig = plt.figure(figsize=(10,4))
for v in range(len(variqall)):
    variq = variqall[v]
    varcount = 'count90'
    variablesglobe = 'T2M'
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Data preliminaries 
    directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/MSFigures_Heat/'
    letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
    ###############################################################################
    ###############################################################################
    modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario','SPEAR_MED']
    slicemonthnamen = ['JJA']
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Read in climate models      
    def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
        data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
        datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
        print('\nOur dataset: ',dataset,' is shaped',data.shape)
        return datar,lats,lons 
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Read in VARIABLE data
    lat_bounds,lon_bounds = UT.regions('US')
    
    if variq == 'EF':
        spear_mALL_SH,lats,lons = read_primary_dataset('SHFLX','SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
        spear_hALL_SH,lats,lons = read_primary_dataset('SHFLX','SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
        
        if modelpick == 'OS':
            spear_osmALL_SH,lats,lons = read_primary_dataset('SHFLX','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
        elif modelpick == 'OS_10ye':
            spear_osmALL_SH,lats,lons = read_primary_dataset('SHFLX','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
        else:
            print(ValueError('wrong model!'))
            sys.exit()
    
        ### Mask over the USA
        spear_m_SH,maskobs = dSS.mask_CONUS(spear_mALL_SH,np.full((spear_mALL_SH.shape[1],spear_mALL_SH.shape[2],spear_mALL_SH.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
        spear_h_SH,maskobs = dSS.mask_CONUS(spear_hALL_SH,np.full((spear_mALL_SH.shape[1],spear_mALL_SH.shape[2],spear_mALL_SH.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
        spear_osm_SH,maskobs = dSS.mask_CONUS(spear_osmALL_SH,np.full((spear_mALL_SH.shape[1],spear_mALL_SH.shape[2],spear_mALL_SH.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
    
        #######################################################################
        spear_mALL_LLH,lats,lons = read_primary_dataset('EVAP','SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
        spear_hALL_LLH,lats,lons = read_primary_dataset('EVAP','SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
        
        if modelpick == 'OS':
            spear_osmALL_LLH,lats,lons = read_primary_dataset('EVAP','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
        elif modelpick == 'OS_10ye':
            spear_osmALL_LLH,lats,lons = read_primary_dataset('EVAP','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
        else:
            print(ValueError('wrong model!'))
            sys.exit()
    
        ### Mask over the USA
        spear_m_LLH,maskobs = dSS.mask_CONUS(spear_mALL_LLH,np.full((spear_mALL_LLH.shape[1],spear_mALL_LLH.shape[2],spear_mALL_LLH.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
        spear_h_LLH,maskobs = dSS.mask_CONUS(spear_hALL_LLH,np.full((spear_mALL_LLH.shape[1],spear_mALL_LLH.shape[2],spear_mALL_LLH.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
        spear_osm_LLH,maskobs = dSS.mask_CONUS(spear_osmALL_LLH,np.full((spear_mALL_LLH.shape[1],spear_mALL_LLH.shape[2],spear_mALL_LLH.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
    
        #######################################################################
        ### Calculate latent heat (H = p*lv*E)
        spear_mALLlal = 1000 * 2.45e6 * spear_m_LLH * (1/86400) * (1/1000)
        spear_hALLlal = 1000 * 2.45e6 * spear_h_LLH * (1/86400) * (1/1000)
        spear_osmALLlal = 1000 * 2.45e6 * spear_osm_LLH * (1/86400) * (1/1000)
    
        ### Calculate evaporative fraction
        spear_m = spear_mALLlal/(spear_m_SH + spear_mALLlal)
        spear_h = spear_hALLlal/(spear_h_SH + spear_hALLlal)
        spear_osm = spear_osmALLlal/(spear_osm_SH + spear_osmALLlal)
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    else:
        spear_mALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
        spear_hALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
    
        if modelpick == 'OS':
            spear_osmALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
        elif modelpick == 'OS_10ye':
            spear_osmALL,lats,lons = read_primary_dataset(variq,'SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
        else:
            print(ValueError('wrong model!'))
            sys.exit()
    
        ### Mask over the USA
        spear_m,maskobs = dSS.mask_CONUS(spear_mALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
        spear_h,maskobs = dSS.mask_CONUS(spear_hALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
        spear_osm,maskobs = dSS.mask_CONUS(spear_osmALL,np.full((spear_mALL.shape[1],spear_mALL.shape[2],spear_mALL.shape[3]),np.nan),'MEDS',lat_bounds,lon_bounds)
        
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Read in temperature data
    lat_bounds,lon_bounds = UT.regions('Globe')
    spear_mt,lats,lons = read_primary_dataset('T2M','SPEAR_MED',monthlychoiceJJA,'SSP585',lat_bounds,lon_bounds)
    spear_ht,lats,lons = read_primary_dataset('T2M','SPEAR_MED_ALLofHistorical',monthlychoiceJJA,'SSP585',lat_bounds,lon_bounds)
    
    if modelpick == 'OS':
        spear_osmt,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoiceJJA,'SSP534OS',lat_bounds,lon_bounds)
    elif modelpick == 'OS_10ye':
        spear_osmt,lats,lons = read_primary_dataset('T2M','SPEAR_MED_SSP534OS_10ye',monthlychoiceJJA,'SSP534OS_10ye',lat_bounds,lon_bounds)
    else:
        print(ValueError('wrong model!'))
        sys.exit()
        
    ### Calculate climatologies
    lon2,lat2 = np.meshgrid(lons,lats)
    
    yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
    climoh_spear = np.nanmean(np.nanmean(spear_h[:,yearq,:,:],axis=1),axis=0)
    climoh_speart = np.nanmean(np.nanmean(spear_ht[:,yearq,:,:],axis=1),axis=0)
    
    spear_ah = spear_h - climoh_spear[np.newaxis,np.newaxis,:,:]
    spear_am = spear_m - climoh_spear[np.newaxis,np.newaxis,:,:]
    spear_aosm = spear_osm - climoh_spear[np.newaxis,np.newaxis,:,:]
    
    spear_aht = spear_ht - climoh_speart[np.newaxis,np.newaxis,:,:]
    spear_amt = spear_mt - climoh_speart[np.newaxis,np.newaxis,:,:]
    spear_aosmt = spear_osmt - climoh_speart[np.newaxis,np.newaxis,:,:]
    
    ### Calculate global average in SPEAR_MED
    lon2,lat2 = np.meshgrid(lons,lats)
    spear_ah_globeht = UT.calc_weightedAve(spear_aht,lat2)
    spear_am_globeht = UT.calc_weightedAve(spear_amt,lat2)
    spear_osm_globeht = UT.calc_weightedAve(spear_aosmt,lat2)
    
    ### Calculate GWL for ensemble means
    gwl_spearht = np.nanmean(spear_ah_globeht,axis=0)
    gwl_spearft = np.nanmean(spear_am_globeht,axis=0)
    gwl_ost = np.nanmean(spear_osm_globeht,axis=0)
    
    ### Combined gwl
    gwl_allt = np.append(gwl_spearht,gwl_spearft,axis=0)
    
    ### Calculate maximum warming
    yrmax_allssp585 = np.argmax(gwl_allt)
    yrmax_os = np.argmax(gwl_ost)
    
    if modelpick == 'OS':
        os_yr = np.where((yearsf == 2040))[0][0]
    elif modelpick == 'OS_10ye':
        os_yr = np.where((yearsf == 2031))[0][0]
    else:
        print(ValueError('wrong model!'))
        sys.exit()
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Read in data for OS daily extremes
    directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
    name = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + variHEAT + '_' + 'SPEAR_MED' + '.nc'
    filename = directorydatah + name
    data = Dataset(filename)
    latus = data.variables['lat'][:]
    lonus = data.variables['lon'][:]
    count90 = data.variables[varcount][:,-86:,:,:]
    data.close()
    
    ### Read in SPEAR_MED_SSP534OS
    directorydatahHEAT = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
    name_osHEAT = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + variHEAT + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
    filename_osHEAT = directorydatahHEAT + name_osHEAT
    data_osHEAT = Dataset(filename_osHEAT)
    count90_osHEAT = data_osHEAT.variables[varcount][:,4:,:,:] # Need to start in 2015, not 2011
    data_osHEAT.close()     
    
    ### Read in SPEAR_MED_SSP534OS_10ye
    directorydatahHEAT = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
    name_os10yeHEAT = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + variHEAT + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
    filename_os10yeHEAT = directorydatahHEAT+ name_os10yeHEAT
    data_os10yeHEAT = Dataset(filename_os10yeHEAT)
    count90_os10yeqHEAT = data_os10yeHEAT.variables[varcount][:]
    data_os10yeHEAT.close()
    
    ### Meshgrid for the CONUS
    lonus2,latus2 = np.meshgrid(lonus,latus)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Combine heatwave timeseries
    count90_os10yeHEAT = np.append(count90_osHEAT[:,:(count90_osHEAT.shape[1]-count90_os10yeqHEAT.shape[1]),:,:],count90_os10yeqHEAT,axis=1)
    
    ### Calculate mean timeseries over CONUS
    if modelpick == 'OS':
        mean_osHEAT = UT.calc_weightedAve(count90_osHEAT,latus2)
    elif modelpick == 'OS_10ye':
        mean_osHEAT = UT.calc_weightedAve(count90_os10yeHEAT,latus2)
        count90_osHEAT = count90_os10yeHEAT
    else:
        print(ValueError('wrong model!'))
        sys.exit()
    
    ### Calculate ensembles mean
    ensmean_osHEAT = np.nanmean(mean_osHEAT,axis=0)
    
    ### Calculate maximum
    max_osHEAT = np.argmax(ensmean_osHEAT)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    nsize = 15
    before_os = count90_osHEAT[:,os_yr-nsize:os_yr,:,:]
    before_os_vari = spear_aosm[:,os_yr-nsize:os_yr,:,:]
    
    beforeMean_os = UT.calc_weightedAve(before_os,latus2)
    beforeMean_os_vari = UT.calc_weightedAve(before_os_vari,latus2)
    
    ### Calculate epochs - Peak
    afterPeak_os = count90_osHEAT[:,yrmax_os:yrmax_os+nsize,:,:]
    after_os_vari = spear_aosm[:,yrmax_os:yrmax_os+nsize,:,:]
    
    afterMean_os = UT.calc_weightedAve(afterPeak_os,latus2)
    afterMean_os_vari = UT.calc_weightedAve(after_os_vari,latus2)
    
    ### Calculate epochs - Last
    end_os = count90_osHEAT[:,-nsize:,:,:]
    end_os_vari = spear_aosm[:,-nsize:,:,:]
    
    end_spear = count90[:,-nsize:,:,:]
    end_spear_vari = spear_am[:,-nsize:,:,:]
    
    endMean_os = UT.calc_weightedAve(end_os,latus2)
    endMean_os_vari = UT.calc_weightedAve(end_os_vari,latus2)
    endMean_spear = UT.calc_weightedAve(end_spear,latus2)
    endMean_spear_vari = UT.calc_weightedAve(end_spear_vari,latus2)
    
    ### Create mask for correlations
    USdata = before_os.copy()*0.
    mask = np.isfinite(before_os)
    
    ### Calculate correlations
    ### Before period
    corr_before_os = sts.pearsonr(beforeMean_os.ravel(),beforeMean_os_vari.ravel())[0]

    slope_before_os,intercept_before_os,r_before_os,p_before_os,se_before_os = sts.linregress(beforeMean_os.ravel(),beforeMean_os_vari.ravel())
    line_before_os_sym = slope_before_os*np.arange(np.size(beforeMean_os)) + intercept_before_os

    ### After period
    corr_after_os = sts.pearsonr(afterMean_os.ravel(),afterMean_os_vari.ravel())[0]
    
    slope_after_os,intercept_after_os,r_after_os,p_after_os,se_after_os = sts.linregress(afterMean_os.ravel(),afterMean_os_vari.ravel())
    line_after_os_sym = slope_after_os*np.arange(np.size(afterMean_os)) + intercept_after_os

    ### End period
    corr_end_os = sts.pearsonr(endMean_os.ravel(),endMean_os_vari.ravel())[0]
    corr_end_spear = sts.pearsonr(endMean_spear.ravel(),endMean_spear_vari.ravel())[0]
    
    slope_end_os,intercept_end_os,r_end_os,p_end_os,se_end_os = sts.linregress(endMean_os.ravel(),endMean_os_vari.ravel())
    line_end_os_sym = slope_end_os*np.arange(np.size(endMean_os)) + intercept_end_os
    slope_end_spear,intercept_end_spear,r_end_spear,p_end_spear,se_end_spear = sts.linregress(endMean_spear.ravel(),endMean_spear_vari.ravel())
    line_end_spear_sym = slope_end_spear*np.arange(np.size(endMean_spear)) + intercept_end_spear
    
    ###############################################################################
    ###############################################################################
    ###############################################################################               
    ### Plot Figure            
    ax = plt.subplot(1,3,v+1)
    
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
    ax.grid(which='major',axis='x',linestyle='-',color='darkgrey',clip_on=False)
    
    plt.scatter(endMean_spear.ravel(),endMean_spear_vari.ravel(),marker='o',s=15,color='dimgray',
                alpha=0.2,edgecolors='dimgray',linewidth=0.3,clip_on=False,label=r'\textbf{SSP5-8.5 [2086-2100] [R=%s]}' % np.round(corr_end_os,2))
    plt.plot(line_end_spear_sym,color='k',linewidth=2,linestyle='-')
    
    if modelpick == 'OS':
        plt.scatter(beforeMean_os.ravel(),beforeMean_os_vari.ravel(),marker='o',s=15,color='teal',
                    alpha=0.2,edgecolors='teal',linewidth=0.3,clip_on=False,label=r'\textbf{2025-2039 [R=%s]}' % np.round(corr_before_os,2))
    elif modelpick == 'OS_10ye':
        plt.scatter(beforeMean_os.ravel(),beforeMean_os_vari.ravel(),marker='o',s=15,color='teal',
                    alpha=0.2,edgecolors='teal',linewidth=0.3,clip_on=False,label=r'\textbf{2016-2030 [R=%s]}' % np.round(corr_before_os,2))
    else:
        print(ValueError('wrong model!'))
        sys.exit()
    plt.plot(line_before_os_sym,color='teal',linewidth=2,linestyle='-')
    
    plt.scatter(afterMean_os.ravel(),afterMean_os_vari.ravel(),marker='o',s=15,color='maroon',
                alpha=0.2,edgecolors='maroon',linewidth=0.3,clip_on=False,label=r'\textbf{AFTER PEAK Tx90 [R=%s]}' % np.round(corr_after_os,2))
    plt.plot(line_after_os_sym,color='maroon',linewidth=2,linestyle='-')
    
    plt.scatter(endMean_os.ravel(),endMean_os_vari.ravel(),marker='o',s=15,color='darkorange',
                alpha=0.2,edgecolors='darkorange',linewidth=0.3,clip_on=False,label=r'\textbf{2086-2100 [R=%s]}' % np.round(corr_end_os,2))
    plt.plot(line_end_os_sym,color='darkorange',linewidth=2,linestyle='-')
    
    leg = plt.legend(shadow=False,fontsize=7.5,loc='upper center',
          bbox_to_anchor=(0.5,1.26),fancybox=True,ncol=2,frameon=False,
          handlelength=0.6,handletextpad=0.3)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    if variq == 'PRECT':
        plt.xticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=8)
        plt.yticks(np.arange(-3,3.1,0.2),map(str,np.round(np.arange(-3,3.1,0.2),2)),fontsize=8)
        plt.xlim([0,80])
        plt.ylim([-1.2,1.2])
    elif variq == 'EVAP':
        plt.xticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=8)
        plt.yticks(np.arange(-3,3.1,0.2),map(str,np.round(np.arange(-3,3.1,0.2),2)),fontsize=8)
        plt.xlim([0,80])
        plt.ylim([-0.6,1.0])
    elif variq == 'WA':
        plt.xticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=8)
        plt.yticks(np.arange(-3,3.1,0.2),map(str,np.round(np.arange(-3,3.1,0.2),2)),fontsize=8)
        plt.xlim([0,80])
        plt.ylim([-1,0.6])
    elif variq == 'rh_ref':
        plt.xticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=8)
        plt.yticks(np.arange(-50,51,2),map(str,np.round(np.arange(-50,51,2),2)),fontsize=8)
        plt.xlim([0,80])
        plt.ylim([-14,8])
    elif variq == 'SHFLX':
        plt.xticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=8)
        plt.yticks(np.arange(-50,51,5),map(str,np.round(np.arange(-50,51,5),2)),fontsize=8)
        plt.xlim([0,80])
        plt.ylim([-15,20])
    elif variq == 'EF':
        plt.xticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=8)
        plt.yticks(np.arange(-1,1.1,0.1),map(str,np.round(np.arange(-1,1.1,0.1),2)),fontsize=8)
        plt.xlim([0,80])
        plt.ylim([-0.2,0.2])
    elif variq == 'swup_toa':
        plt.xticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=8)
        plt.yticks(np.arange(-50,51,5),map(str,np.round(np.arange(-50,51,5),2)),fontsize=8)
        plt.xlim([0,80])
        plt.ylim([-20,10])
    
    if variHEAT == 'TMAX':
        plt.xlabel(r'\textbf{Count of Tx90 days in JJA}',fontsize=7,color='dimgrey')
    elif variHEAT == 'TMIN':
        plt.xlabel(r'\textbf{Count of Tn90 days in JJA}',fontsize=7,color='dimgrey')
    elif variHEAT == 'T2M':
        plt.xlabel(r'\textbf{Count of T90 days in JJA}',fontsize=7,color='dimgrey')
    else:
        print(ValueError('wrong model!'))
        sys.exit()
        
    ax.annotate(r'\textbf{[%s]}' % (letters[v]),xy=(0,0),xytext=(1,1.06),
              textcoords='axes fraction',color='k',fontsize=10,
              rotation=0,ha='center',va='center')
    
    if v == 0:
        plt.ylabel(r'\textbf{Precipitation Anomaly [mm/day]}',fontsize=7,color='dimgrey')
    elif v==1:
        plt.ylabel(r'\textbf{Evaporation Anomaly [mm/day]}',fontsize=7,color='dimgrey')
    elif v==2:
        plt.ylabel(r'\textbf{RH Anomaly [\%]}',fontsize=7,color='dimgrey')

plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'Scatter_Heatwaves_%s_%s_Water.png' % (variHEAT,modelpick),dpi=300)

