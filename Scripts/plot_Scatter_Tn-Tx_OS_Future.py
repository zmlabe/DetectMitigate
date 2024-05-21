"""
Examine relationship between Tn and Tx

Author    : Zachary M. Labe
Date      : 14 May 2024
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

varcount = 'count90'
variablesglobe = 'T2M'
numOfEns = 30
numOfEns_10ye = 30
yearsh = np.arange(1921,2014+1,1)
years = np.arange(1921,2100+1)
yearsf = np.arange(2015,2100+1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'Globe'

###############################################################################
###############################################################################
###############################################################################
### Read in climate models      
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons 
# ###############################################################################
# ###############################################################################
# ###############################################################################
# ### Read in data
# directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
# name = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + 'TMIN' + '_' + 'SPEAR_MED' + '.nc'
# filename = directorydatah + name
# data = Dataset(filename)
# latus = data.variables['lat'][:]
# lonus = data.variables['lon'][:]
# spear_am = data.variables[varcount][:,-86:,:,:]
# data.close()

# directorydatahHEAT = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
# name_osHEAT = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + 'TMIN' + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
# filename_osHEAT = directorydatahHEAT + name_osHEAT
# data_osHEAT = Dataset(filename_osHEAT)
# spear_aosm = data_osHEAT.variables[varcount][:,4:,:,:] # Need to start in 2015, not 2011
# latus = data_osHEAT.variables['lat'][:]
# lonus = data_osHEAT.variables['lon'][:]
# data_osHEAT.close()

# ### Read in SPEAR_MED_SSP534OS_10ye
# directorydatahHEAT = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
# name_os10yeHEAT = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + 'TMIN' + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
# filename_os10yeHEAT = directorydatahHEAT+ name_os10yeHEAT
# data_os10yeHEAT = Dataset(filename_os10yeHEAT)
# spear_aosm_10yeq = data_os10yeHEAT.variables[varcount][:]
# data_os10yeHEAT.close()

# ### Combine data
# spear_aosm_10ye = np.append(spear_aosm[:,:(spear_aosm.shape[1]-spear_aosm_10yeq.shape[1]),:,:],spear_aosm_10yeq,axis=1)

# ### Read in temperature data
# lat_bounds,lon_bounds = UT.regions(reg_name)
# spear_mt,lats,lons = read_primary_dataset('T2M','SPEAR_MED',monthlychoice,'SSP585',lat_bounds,lon_bounds)
# spear_ht,lats,lons = read_primary_dataset('T2M','SPEAR_MED_ALLofHistorical',monthlychoice,'SSP585',lat_bounds,lon_bounds)
# spear_osmt,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
# spear_osm_10yet,lats,lons = read_primary_dataset('T2M','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
# lon2,lat2 = np.meshgrid(lons,lats)

# yearq = np.where((yearsh >= 1921) & (yearsh <= 1950))[0]
# climoh_speart = np.nanmean(np.nanmean(spear_ht[:,yearq,:,:],axis=1),axis=0)

# spear_aht = spear_ht - climoh_speart[np.newaxis,np.newaxis,:,:]
# spear_amt = spear_mt - climoh_speart[np.newaxis,np.newaxis,:,:]
# spear_aosmt = spear_osmt - climoh_speart[np.newaxis,np.newaxis,:,:]
# spear_aosm_10yet = spear_osm_10yet - climoh_speart[np.newaxis,np.newaxis,:,:]

# ### Calculate global average in SPEAR_MED
# lon2,lat2 = np.meshgrid(lons,lats)
# spear_ah_globeht = UT.calc_weightedAve(spear_aht,lat2)
# spear_am_globeht = UT.calc_weightedAve(spear_amt,lat2)
# spear_osm_globeht = UT.calc_weightedAve(spear_aosmt,lat2)
# spear_osm_10ye_globeht = UT.calc_weightedAve(spear_aosm_10yet,lat2)

# ### Calculate GWL for ensemble means
# gwl_spearht = np.nanmean(spear_ah_globeht,axis=0)
# gwl_spearft = np.nanmean(spear_am_globeht,axis=0)
# gwl_ost = np.nanmean(spear_osm_globeht,axis=0)
# gwl_os_10yet = np.nanmean(spear_osm_10ye_globeht,axis=0)

# ### Combined gwl
# gwl_allt = np.append(gwl_spearht,gwl_spearft,axis=0)

# ### Calculate maximum warming
# yrmax_allssp585 = np.argmax(gwl_allt)
# yrmax_os = np.argmax(gwl_ost)
# yrmax_os_10ye = np.argmax(gwl_os_10yet)
# os_yr = np.where((yearsf == 2040))[0][0]
# os_10ye_yr = np.where((yearsf == 2031))[0][0]

# ###############################################################################
# ###############################################################################
# ###############################################################################
# ### Read in data for OS daily extremes
# directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
# name = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + 'TMAX' + '_' + 'SPEAR_MED' + '.nc'
# filename = directorydatah + name
# data = Dataset(filename)
# latus = data.variables['lat'][:]
# lonus = data.variables['lon'][:]
# count90 = data.variables[varcount][:,-86:,:,:]
# data.close()

# ### Read in SPEAR_MED_SSP534OS
# directorydatahHEAT = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
# name_osHEAT = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + 'TMAX' + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
# filename_osHEAT = directorydatahHEAT + name_osHEAT
# data_osHEAT = Dataset(filename_osHEAT)
# count90_osHEAT = data_osHEAT.variables[varcount][:,4:,:,:] # Need to start in 2015, not 2011
# latus = data_osHEAT.variables['lat'][:]
# lonus = data_osHEAT.variables['lon'][:]
# data_osHEAT.close()

# ### Read in SPEAR_MED_SSP534OS_10ye
# directorydatahHEAT = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
# name_os10yeHEAT = 'HeatStats/HeatStats' + '_JJA_' + 'US' + '_' + 'TMAX' + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
# filename_os10yeHEAT = directorydatahHEAT+ name_os10yeHEAT
# data_os10yeHEAT = Dataset(filename_os10yeHEAT)
# count90_os10yeqHEAT = data_os10yeHEAT.variables[varcount][:]
# data_os10yeHEAT.close()

# ### Meshgrid for the CONUS
# lonus2,latus2 = np.meshgrid(lonus,latus)

# ###############################################################################
# ###############################################################################
# ###############################################################################
# ### Combine heatwave timeseries
# count90_os10yeHEAT = np.append(count90_osHEAT[:,:(count90_osHEAT.shape[1]-count90_os10yeqHEAT.shape[1]),:,:],count90_os10yeqHEAT,axis=1)

# ### Calculate mean timeseries over CONUS
# mean_osHEAT = UT.calc_weightedAve(count90_osHEAT,latus2)
# mean_os10yeHEAT = UT.calc_weightedAve(count90_os10yeHEAT,latus2)

# ### Calculate ensembles mean
# ensmean_osHEAT = np.nanmean(mean_osHEAT,axis=0)
# ensmean_os10yeHEAT = np.nanmean(mean_os10yeHEAT,axis=0)

# ### Calculate maximum
# max_osHEAT = np.argmax(ensmean_osHEAT)
# max_os10yeHEAT = np.argmax(ensmean_os10yeHEAT)

# ###############################################################################
# ###############################################################################
# ###############################################################################
# nsize = 15
# before_os = count90_osHEAT[:,os_yr-nsize:os_yr,:,:]
# before_os10ye = count90_os10yeHEAT[:,os_10ye_yr-nsize:os_10ye_yr,:,:]

# before_os_vari = spear_aosm[:,os_yr-nsize:os_yr,:,:]
# before_os_10ye_vari = spear_aosm_10ye[:,os_10ye_yr-nsize:os_10ye_yr,:,:]

# beforeMean_os = UT.calc_weightedAve(before_os,latus2)
# beforeMean_os_10ye = UT.calc_weightedAve(before_os10ye,latus2)
# beforeMean_os_vari = UT.calc_weightedAve(before_os_vari,latus2)
# beforeMean_os_10ye_vari = UT.calc_weightedAve(before_os_10ye_vari,latus2)

# ### Calculate epochs - Peak
# afterPeak_os = count90_osHEAT[:,yrmax_os:yrmax_os+nsize,:,:]
# afterPeak_os10ye = count90_os10yeHEAT[:,yrmax_os_10ye:yrmax_os_10ye+nsize,:,:]

# after_os_vari = spear_aosm[:,yrmax_os:yrmax_os+nsize,:,:]
# after_os_10ye_vari = spear_aosm_10ye[:,yrmax_os_10ye:yrmax_os_10ye+nsize,:,:]

# afterMean_os = UT.calc_weightedAve(afterPeak_os,latus2)
# afterMean_os_10ye = UT.calc_weightedAve(afterPeak_os10ye,latus2)
# afterMean_os_vari = UT.calc_weightedAve(after_os_vari,latus2)
# afterMean_os_10ye_vari = UT.calc_weightedAve(after_os_10ye_vari,latus2)

# ### Calculate epochs - Last
# end_os = count90_osHEAT[:,-nsize:,:,:]
# end_os10ye = count90_os10yeHEAT[:,-nsize:,:,:]

# end_os_vari = spear_aosm[:,-nsize:,:,:]
# end_os_10ye_vari = spear_aosm_10ye[:,-nsize:,:,:]

# end_spear = count90[:,-nsize:,:,:]
# end_spear_vari = spear_am[:,-nsize:,:,:]

# endMean_os = UT.calc_weightedAve(end_os,latus2)
# endMean_os_10ye = UT.calc_weightedAve(end_os10ye,latus2)
# endMean_os_vari = UT.calc_weightedAve(end_os_vari,latus2)
# endMean_os_10ye_vari = UT.calc_weightedAve(end_os_10ye_vari,latus2)
# endMean_spear = UT.calc_weightedAve(end_spear,latus2)
# endMean_spear_vari = UT.calc_weightedAve(end_spear_vari,latus2)

# ### Create mask for correlations
# USdata = before_os.copy()*0.
# mask = np.isfinite(before_os)

# ### Calculate correlations
# ### Before period
# corr_before_os = sts.pearsonr(beforeMean_os.ravel(),beforeMean_os_vari.ravel())[0]
# corr_before_os_10ye = sts.pearsonr(beforeMean_os_10ye.ravel(),beforeMean_os_10ye_vari.ravel())[0]

# slope_before_os,intercept_before_os,r_before_os,p_before_os,se_before_os = sts.linregress(beforeMean_os.ravel(),beforeMean_os_vari.ravel())
# line_before_os_sym = slope_before_os*np.arange(np.size(beforeMean_os)) + intercept_before_os
# slope_before_os_10ye,intercept_before_os_10ye,r_before_os_10ye,p_before_os_10ye,se_before_os_10ye = sts.linregress(beforeMean_os_10ye.ravel(),beforeMean_os_10ye_vari.ravel())
# line_before_os_10ye_sym = slope_before_os_10ye*np.arange(np.size(beforeMean_os_10ye)) + intercept_before_os_10ye

# ### After period
# corr_after_os = sts.pearsonr(afterMean_os.ravel(),afterMean_os_vari.ravel())[0]
# corr_after_os_10ye = sts.pearsonr(afterMean_os_10ye.ravel(),afterMean_os_10ye_vari.ravel())[0]

# slope_after_os,intercept_after_os,r_after_os,p_after_os,se_after_os = sts.linregress(afterMean_os.ravel(),afterMean_os_vari.ravel())
# line_after_os_sym = slope_after_os*np.arange(np.size(afterMean_os)) + intercept_after_os
# slope_after_os_10ye,intercept_after_os_10ye,r_after_os_10ye,p_after_os_10ye,se_after_os_10ye = sts.linregress(afterMean_os_10ye.ravel(),afterMean_os_10ye_vari.ravel())
# line_after_os_10ye_sym = slope_after_os_10ye*np.arange(np.size(afterMean_os_10ye)) + intercept_after_os_10ye

# ### End period
# corr_end_os = sts.pearsonr(endMean_os.ravel(),endMean_os_vari.ravel())[0]
# corr_end_os_10ye = sts.pearsonr(endMean_os_10ye.ravel(),endMean_os_10ye_vari.ravel())[0]
# corr_end_spear = sts.pearsonr(endMean_spear.ravel(),endMean_spear_vari.ravel())[0]

# slope_end_os,intercept_end_os,r_end_os,p_end_os,se_end_os = sts.linregress(endMean_os.ravel(),endMean_os_vari.ravel())
# line_end_os_sym = slope_end_os*np.arange(np.size(endMean_os)) + intercept_end_os
# slope_end_os_10ye,intercept_end_os_10ye,r_end_os_10ye,p_end_os_10ye,se_end_os_10ye = sts.linregress(endMean_os_10ye.ravel(),endMean_os_10ye_vari.ravel())
# line_end_os_10ye_sym = slope_end_os_10ye*np.arange(np.size(endMean_os_10ye)) + intercept_end_os_10ye
# slope_end_spear,intercept_end_spear,r_end_spear,p_end_spear,se_end_spear = sts.linregress(endMean_spear.ravel(),endMean_spear_vari.ravel())
# line_end_spear_sym = slope_end_spear*np.arange(np.size(endMean_spear)) + intercept_end_spear

###############################################################################
###############################################################################
###############################################################################               
### Plot Figure
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
ax.grid(which='major',axis='x',linestyle='-',color='darkgrey',clip_on=False)
ax.grid(which='major',axis='y',linestyle='-',color='darkgrey',clip_on=False)

plt.plot(np.arange(0,101,1),np.arange(0,101,1),linewidth=2,linestyle='--',color='dimgrey',dashes=(1,0.3))

plt.scatter(endMean_spear.ravel(),endMean_spear_vari.ravel(),marker='o',s=30,color='r',
            alpha=0.3,edgecolors='r',linewidth=0,clip_on=False,label=r'\textbf{SSP5-8.5 [2086-2100] [R=%s]}' % np.round(corr_end_os,2))
plt.plot(line_end_spear_sym,color='r',linewidth=2,linestyle='-')

plt.scatter(beforeMean_os.ravel(),beforeMean_os_vari.ravel(),marker='o',s=30,color='teal',
            alpha=0.3,edgecolors='teal',linewidth=0,clip_on=False,label=r'\textbf{SSP5-3.4OS [BEFORE] [R=%s]}' % np.round(corr_before_os,2))
plt.plot(line_before_os_sym,color='teal',linewidth=2,linestyle='-')

plt.scatter(afterMean_os.ravel(),afterMean_os_vari.ravel(),marker='o',s=30,color='maroon',
            alpha=0.3,edgecolors='maroon',linewidth=0,clip_on=False,label=r'\textbf{SSP5-3.4OS [AFTER PEAK] [R=%s]}' % np.round(corr_after_os,2))
plt.plot(line_after_os_sym,color='maroon',linewidth=2,linestyle='-')

plt.scatter(endMean_os.ravel(),endMean_os_vari.ravel(),marker='o',s=30,color='darkorange',
            alpha=0.3,edgecolors='darkorange',linewidth=0,clip_on=False,label=r'\textbf{SSP5-3.4OS [2086-2100] [R=%s]}' % np.round(corr_end_os,2))
plt.plot(line_end_os_sym,color='darkorange',linewidth=2,linestyle='-')

leg = plt.legend(shadow=False,fontsize=7,loc='upper left',
      bbox_to_anchor=(0,1.02),fancybox=True,ncol=1,frameon=False,
      handlelength=0.5,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=10)
plt.yticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=10)
plt.xlim([0,80])
plt.ylim([0,80])

plt.xlabel(r'\textbf{Count of Tx90 days in JJA}',fontsize=11,color='dimgrey')
plt.ylabel(r'\textbf{Count of Tn90 days in JJA}',fontsize=11,color='dimgrey')

############################################################################### 
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
ax.grid(which='major',axis='x',linestyle='-',color='darkgrey',clip_on=False)
ax.grid(which='major',axis='y',linestyle='-',color='darkgrey',clip_on=False)

plt.plot(np.arange(0,101,1),np.arange(0,101,1),linewidth=2,linestyle='--',color='dimgrey',dashes=(1,0.3))

plt.scatter(endMean_spear.ravel(),endMean_spear_vari.ravel(),marker='o',s=30,color='r',
            alpha=0.3,edgecolors='r',linewidth=0,clip_on=False,label=r'\textbf{SSP5-8.5 [2086-2100] [R=%s]}' % np.round(corr_end_os,2))
plt.plot(line_end_spear_sym,color='r',linewidth=2,linestyle='-')

plt.scatter(beforeMean_os_10ye.ravel(),beforeMean_os_10ye_vari.ravel(),marker='o',s=30,color='teal',
            alpha=0.3,edgecolors='teal',linewidth=0,clip_on=False,label=r'\textbf{SSP5-3.4OS_10ye [BEFORE] [R=%s]}'% np.round(corr_before_os_10ye,2))
plt.plot(line_before_os_10ye_sym,color='teal',linewidth=2,linestyle='-')

plt.scatter(afterMean_os_10ye.ravel(),afterMean_os_10ye_vari.ravel(),marker='o',s=30,color='maroon',
            alpha=0.3,edgecolors='maroon',linewidth=0,clip_on=False,label=r'\textbf{SSP5-3.4OS_10ye [AFTER PEAK] [R=%s]}' % np.round(corr_after_os_10ye,2))
plt.plot(line_after_os_10ye_sym,color='maroon',linewidth=2,linestyle='-')

plt.scatter(endMean_os_10ye.ravel(),endMean_os_10ye_vari.ravel(),marker='o',s=30,color='darkorange',
            alpha=0.3,edgecolors='darkorange',linewidth=0,clip_on=False,label=r'\textbf{SSP5-3.4OS_10ye [2086-2100] [R=%s]}' % np.round(corr_end_os_10ye,2))
plt.plot(line_end_os_10ye_sym,color='darkorange',linewidth=2,linestyle='-')

leg = plt.legend(shadow=False,fontsize=7,loc='upper left',
      bbox_to_anchor=(0,1.02),fancybox=True,ncol=1,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=10)
plt.yticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),fontsize=10)
plt.xlim([0,80])
plt.ylim([0,80])

plt.xlabel(r'\textbf{Count of Tx90 days in JJA}',fontsize=11,color='dimgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'Scatter_Tn90-Tx90_OS_ssp585.png',dpi=300)

