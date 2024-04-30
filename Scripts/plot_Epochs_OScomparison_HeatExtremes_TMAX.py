"""
Calculate trend for OS with daily TMAX

Author    : Zachary M. Labe
Date      : 25 July 2023
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

variablesall = ['TMIN']
variq = variablesall[0]
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
reg_name = 'US'
varcount = 'count99'

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
### Read in data for OS daily extremes
### Read in SPEAR_MED_SSP534OS
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS' + '.nc'
filename_os = directorydatah + name_os
data_os = Dataset(filename_os)
count90_os = data_os.variables[varcount][:,4:,:,:] # Need to start in 2015, not 2011
data_os.close()

### Read in SPEAR_MED_SSP534OS_10ye
directorydatah = '/work/Zachary.Labe/Research/DetectMitigate/DataExtremes/'
name_os10ye = 'HeatStats/HeatStats' + '_JJA_' + reg_name + '_' + variq + '_' + 'SPEAR_MED_SSP534OS_10ye' + '.nc'
filename_os10ye = directorydatah + name_os10ye
data_os10ye = Dataset(filename_os10ye)
count90_os10yeq = data_os10ye.variables[varcount][:]
lat1 = data_os10ye.variables['lat'][:]
lon1 = data_os10ye.variables['lon'][:]
data_os10ye.close()

###############################################################################
###############################################################################
###############################################################################
### Parameters
lon2,lat2 = np.meshgrid(lon1,lat1)
os_yr = np.where((yearsf == 2040))[0][0]
os_10ye_yr = np.where((yearsf == 2031))[0][0]

### Combine heatwave timeseries
count90_os10ye = np.append(count90_os[:,:(count90_os.shape[1]-count90_os10yeq.shape[1]),:,:],count90_os10yeq,axis=1)

### Calculate mean timeseries over CONUS
mean_os = UT.calc_weightedAve(count90_os,lat2)
mean_os10ye = UT.calc_weightedAve(count90_os10ye,lat2)

### Calculate ensembles mean
ensmean_os = np.nanmean(mean_os,axis=0)
ensmean_os10ye = np.nanmean(mean_os10ye,axis=0)

### Calculate maximum
max_os = np.argmax(ensmean_os)
max_os10ye = np.argmax(ensmean_os10ye)

###############################################################################
###############################################################################
###############################################################################
### Calculate epochs before
nsize = 10
before_os = np.nanmean(count90_os[:,os_yr-nsize:os_yr,:,:],axis=(0,1))
before_os10ye = np.nanmean(count90_os10ye[:,os_10ye_yr-nsize:os_10ye_yr,:,:],axis=(0,1))

### Calculate epochs - Start
afterStart_os = np.nanmean(count90_os[:,os_yr:os_yr+nsize,:,:],axis=(0,1))
afterStart_os10ye = np.nanmean(count90_os10ye[:,os_10ye_yr:os_10ye_yr+nsize,:,:],axis=(0,1))

### Calculate epochs - Peak
afterPeak_os = np.nanmean(count90_os[:,max_os:max_os+nsize,:,:],axis=(0,1))
afterPeak_os10ye = np.nanmean(count90_os10ye[:,max_os10ye:max_os10ye+nsize,:,:],axis=(0,1))

###############################################################################
###############################################################################
###############################################################################
### Calculate differences
diff_Start_os = afterStart_os - before_os
diff_Start_os10ye = afterStart_os10ye - before_os10ye

diff_Peak_os = afterPeak_os - before_os
diff_Peak_os10ye = afterPeak_os10ye - before_os10ye

alldiff_Start = diff_Start_os - diff_Start_os10ye
alldiff_Peak = diff_Peak_os  - diff_Peak_os10ye

### Calculate statistical significance (FDR)
alpha_f = 0.05
pafterStart_os = UT.calc_FDR_ttest(np.nanmean(count90_os[:,os_yr:os_yr+nsize,:,:],axis=1),np.nanmean(count90_os[:,os_yr-nsize:os_yr,:,:],axis=1),alpha_f)
pafterStart_os10ye = UT.calc_FDR_ttest(np.nanmean(count90_os10ye[:,os_10ye_yr:os_10ye_yr+nsize,:,:],axis=1),np.nanmean(count90_os10ye[:,os_10ye_yr-nsize:os_10ye_yr,:,:],axis=1),alpha_f)

pafterPeak_os = UT.calc_FDR_ttest(np.nanmean(count90_os[:,max_os:max_os+nsize,:,:],axis=1),np.nanmean(count90_os[:,os_yr-nsize:os_yr,:,:],axis=1),alpha_f)
pafterPeak_os10ye = UT.calc_FDR_ttest(np.nanmean(count90_os10ye[:,max_os10ye:max_os10ye+nsize,:,:],axis=1),np.nanmean(count90_os10ye[:,os_10ye_yr-nsize:os_10ye_yr,:,:],axis=1),alpha_f)

palldiff_Start = UT.calc_FDR_ttest(np.nanmean(count90_os[:,os_yr:os_yr+nsize,:,:],axis=1)-np.nanmean(count90_os[:,os_yr-nsize:os_yr,:,:],axis=1),np.nanmean(count90_os10ye[:,os_10ye_yr:os_10ye_yr+nsize,:,:],axis=1)-np.nanmean(count90_os10ye[:,os_10ye_yr-nsize:os_10ye_yr,:,:],axis=1),alpha_f)
palldiff_Peak = UT.calc_FDR_ttest(np.nanmean(count90_os[:,max_os:max_os+nsize,:,:],axis=1)-np.nanmean(count90_os[:,os_yr-nsize:os_yr,:,:],axis=1),np.nanmean(count90_os10ye[:,max_os10ye:max_os10ye+nsize,:,:],axis=1)-np.nanmean(count90_os10ye[:,os_10ye_yr-nsize:os_10ye_yr,:,:],axis=1),alpha_f)

allplots = [diff_Start_os,diff_Start_os10ye,alldiff_Start,
            diff_Peak_os,diff_Peak_os10ye,alldiff_Peak]
allsig = [pafterStart_os,pafterStart_os10ye,palldiff_Start,
          pafterPeak_os,pafterPeak_os10ye,palldiff_Peak]

###############################################################################
###############################################################################
###############################################################################
### Plot figure

### Select map type
style = 'US'

if style == 'ortho':
    m = Basemap(projection='ortho',lon_0=270,
                lat_0=50,resolution='h',round=True,area_thresh=10000)
elif style == 'polar':
    m = Basemap(projection='npstere',boundinglat=67,lon_0=270,resolution='h',round=True,area_thresh=10000)
elif style == 'global':
    m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
elif style == 'US':
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
                area_thresh=10000)
    
### Colorbar limits
if any([variq == 'TMAX',variq == 'T2M']):
    barlim = np.arange(-15,16,5)
    limit = np.arange(-15,16,1)
if variq == 'TMAX':
    if varcount == 'count90':
        label = r'\textbf{Difference in Count of Tx90 [%s]}' % variq
    elif varcount == 'count95':
        label = r'\textbf{Difference in Count of Tx95 [%s]}' % variq
    elif varcount == 'count99':
        label = r'\textbf{Difference in Count of Tx99 [%s]}' % variq
if variq == 'TMIN':
    if varcount == 'count90':
        label = r'\textbf{Difference in Count of Tn90 [%s]}' % variq
    elif varcount == 'count95':
        label = r'\textbf{Difference in Count of Tn95 [%s]}' % variq
    elif varcount == 'count99':
        label = r'\textbf{Difference in Count of Tn99 [%s]}' % variq
elif variq == 'T2M':
    if varcount == 'count90':
        label = r'\textbf{Difference in Count of T2M-Tx90 [%s]}' % variq 
    elif varcount == 'count95':
        label = r'\textbf{Difference in Count of T2M-Tx95 [%s]}' % variq
    elif varcount == 'count95':
        label = r'\textbf{Difference in Count of T2M-Tx99 [%s]}' % variq
        
labelsTop = [r'\textbf{SSP5-3.4OS}',r'\textbf{SSP5-3.4OS_10ye}',r'\textbf{Difference}']

### Map world map
fig = plt.figure(figsize=(10,4))
for i in range(len(allplots)):
                  
    ax = plt.subplot(2,3,i+1)
    for txt in fig.texts:
        txt.set_visible(False)
    
    pval = allsig[i]
    
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    m.drawcoastlines(color='darkgrey',linewidth=1)
    m.drawstates(color='darkgrey',linewidth=0.5)
    m.drawcountries(color='darkgrey',linewidth=0.5)
    
    ### Make the plot continuous
    cs = m.contourf(lon2,lat2,allplots[i],limit,
                      extend='both',latlon=True)
    cs2 = m.contourf(lon2,lat2,pval,colors='none',hatches=['.....'],latlon=True)  
                    
    if any([variq == 'TMAX',variq == 'T2M']):
        cmap = cmocean.cm.balance 
    cs.set_cmap(cmap)
    
    ax.annotate(r'\textbf{[%s]}' % letters[i],xy=(0,0),xytext=(0.99,1.03),
              textcoords='axes fraction',color='k',fontsize=10,
              rotation=0,ha='center',va='center')
    
    if i < 3:
        plt.title(r'\textbf{%s}' % (labelsTop[i]),fontsize=11,color='dimgrey')
        
    if i == 0:
        ax.annotate(r'\textbf{After OS}',xy=(0,0),xytext=(-0.05,0.50),
                  textcoords='axes fraction',color='k',fontsize=15,
                  rotation=90,ha='center',va='center')
    if i == 3:
        ax.annotate(r'\textbf{After Peak}',xy=(0,0),xytext=(-0.05,0.50),
                  textcoords='axes fraction',color='k',fontsize=15,
                  rotation=90,ha='center',va='center')

cbar_ax1 = fig.add_axes([0.31,0.09,0.4,0.02])                
cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=8,color='k',labelpad=5)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

### Save figure 
plt.tight_layout()   
plt.subplots_adjust(bottom=0.15,wspace=-0.3)
plt.savefig(directoryfigure + 'Epochs_HeatExtremes_%s_%s_%s-%s.png' % (variq,seasons[0],varcount,variq),dpi=300)
