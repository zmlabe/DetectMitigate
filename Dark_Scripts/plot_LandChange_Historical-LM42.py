"""
Plot land fractions to different SPEAR runs for the historical period

Author    : Zachary M. Labe
Date      : 30 August 2023
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
from scipy.interpolate import griddata as g
import read_SPEAR_MED_LM42p2_test as SPL

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

numOfEns = 30
numOfEns_10ye = 9
years = np.arange(1921,2014+1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u"]
location = [1,2,3,4,5,6,7,8,10]
###############################################################################
###############################################################################
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'US'

epochsize = 10
simulations = ['SPEAR_MED_Historical','SPEAR_MED_LM42p2_test']
scenarios = ['SSP585','SPEAR_MED_LM42']

modelnames = ['SPEAR_MED_SSP585','SPEAR_MED_LM42_test']
variablenames = np.repeat(['frac_crop','frac_ntrl','frac_past','frac_scnd'],2)

###############################################################################
###############################################################################
###############################################################################
### Read in climate models      
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons 

### Temporary regridding function
def regrid(lat11,lon11,lat21,lon21,var,years):
    """
    Interpolated on selected grid. Reads SPEAR in as 4d with 
    [year,lat,lon]
    """
    
    lon1,lat1 = np.meshgrid(lon11,lat11)
    lon2,lat2 = np.meshgrid(lon21,lat21)
    
    varn_re = np.reshape(var,((lat1.shape[0]*lon1.shape[1])))   
    
    print('Completed: Start regridding process:')
    varn = g((np.ravel(lat1),np.ravel(lon1)),varn_re,(lat2,lon2),method='linear')
    print('Completed: End Regridding---')
    return varn

def readData(simulation,scenario,epochsize):
       
    if simulation == 'SPEAR_MED_Historical':
        lat_bounds,lon_bounds = UT.regions('Globe')
        frac_crop,lats,lons = read_primary_dataset('frac_crop',simulation,monthlychoice,scenario,lat_bounds,lon_bounds)
        frac_ntrl,lats,lons = read_primary_dataset('frac_ntrl',simulation,monthlychoice,scenario,lat_bounds,lon_bounds)
        frac_past,lats,lons = read_primary_dataset('frac_past',simulation,monthlychoice,scenario,lat_bounds,lon_bounds)
        frac_scnd,lats,lons = read_primary_dataset('frac_scnd',simulation,monthlychoice,scenario,lat_bounds,lon_bounds)
        frac_rangem = np.full((frac_scnd.shape),np.nan)
        diffranger = np.full((frac_scnd.shape),np.nan)
        
    else:
        lats,lons,frac_crop = SPL.read_SPEAR_MED_LM42p2_test('/work/Zachary.Labe/Data/','frac_crop',monthlychoice,4,np.nan,3,'historical')
        lats,lons,frac_ntrl = SPL.read_SPEAR_MED_LM42p2_test('/work/Zachary.Labe/Data/','frac_ntrl',monthlychoice,4,np.nan,3,'historical')
        lats,lons,frac_past = SPL.read_SPEAR_MED_LM42p2_test('/work/Zachary.Labe/Data/','frac_past',monthlychoice,4,np.nan,3,'historical')
        lats,lons,frac_scnd = SPL.read_SPEAR_MED_LM42p2_test('/work/Zachary.Labe/Data/','frac_scnd',monthlychoice,4,np.nan,3,'historical')
        lats,lons,frac_range = SPL.read_SPEAR_MED_LM42p2_test('/work/Zachary.Labe/Data/','frac_range',monthlychoice,4,np.nan,3,'historical')
        frac_rangem = np.nanmean(frac_range,axis=0)
        diffrange = np.nanmean(frac_rangem[-epochsize:,:,:] - frac_rangem[:epochsize,:,:],axis=0)
    
    ### Calculate ensemble mean
    frac_cropm = np.nanmean(frac_crop,axis=0)
    frac_ntrlm = np.nanmean(frac_ntrl,axis=0)
    frac_pastm = np.nanmean(frac_past,axis=0)
    frac_scndm = np.nanmean(frac_scnd,axis=0)
        
    ### Slice for 1921-2014
    yearsactual = np.arange(1921,2014+1,1)
    
    diffcrop = np.nanmean(frac_cropm[-epochsize:,:,:] - frac_cropm[:epochsize,:,:],axis=0)
    diffntrl = np.nanmean(frac_ntrlm[-epochsize:,:,:] - frac_ntrlm[:epochsize,:,:],axis=0)
    diffpast = np.nanmean(frac_pastm[-epochsize:,:,:] - frac_pastm[:epochsize,:,:],axis=0) 
    diffscnd = np.nanmean(frac_scndm[-epochsize:,:,:] - frac_scndm[:epochsize,:,:],axis=0)
    
    if simulation == 'SPEAR_MED_LM42p2_test':
        data = Dataset('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_SSP245/monthly/frac_crop/frac_crop_01_2011-2100.nc')
        latss = data.variables['lat'][:]
        lonss = data.variables['lon'][:]
        data.close()
        
        diffcropr_raw = regrid(lats,lons,latss,lonss,diffcrop,yearsactual)
        diffntrlr_raw = regrid(lats,lons,latss,lonss,diffntrl,yearsactual)
        diffpastr_raw = regrid(lats,lons,latss,lonss,diffpast,yearsactual)
        diffscndr_raw = regrid(lats,lons,latss,lonss,diffscnd,yearsactual)
        diffranger_raw = regrid(lats,lons,latss,lonss,diffrange,yearsactual)
        
        ### Need to convert by area from units m2 to %
        data = Dataset('//work/Zachary.Labe/Data/SPEAR/SPEAR_MED_LM42p2_test/monthly/Metadata/atmos.static.nc')
        area = data.variables['area'][:]
        data.close()
        
        diffcropr = diffcropr_raw/area # m2/m2
        diffntrlr = diffntrlr_raw/area # m2/m2
        diffpastr = diffpastr_raw/area # m2/m2
        diffscndr = diffscndr_raw/area # m2/m2
        diffranger = diffranger_raw/area # m2/m2
        
        lats = latss
        lons = lonss
        
    else:
        diffcropr = diffcrop
        diffntrlr = diffntrl
        diffpastr = diffpast
        diffscndr = diffscnd

    return frac_cropm,frac_ntrlm,frac_pastm,frac_scndm,frac_rangem,diffcropr,diffntrlr,diffpastr,diffscndr,diffranger,lats,lons

frac_cropm_585,frac_ntrlm_585,frac_pastm_585,frac_scndm_585,frac_rangem_585,diffcrop_585,diffntrl_585,diffpast_585,diffscnd_585,diffrange_585,lats,lons = readData(simulations[0],scenarios[0],epochsize)
frac_cropm_LM42,frac_ntrlm_LM42,frac_pastm_LM42,frac_scndm_LM42,frac_rangem_LM42,diffcrop_LM42,diffntrl_LM42,diffpast_LM42,diffscnd_LM42,diffrange_LM42,lats,lons = readData(simulations[1],scenarios[1],epochsize)

preparedata = [diffcrop_585,diffcrop_LM42,
                diffntrl_585,diffntrl_LM42,
                diffpast_585,diffpast_LM42,
                diffscnd_585,diffscnd_LM42,
                diffrange_LM42]

###############################################################################
###############################################################################
###############################################################################
### Plot differences in land fraction
fig = plt.figure(figsize=(4,7))

label = r'\textbf{Fraction = 2005-2014 minus 1921-1930}'
limit = np.arange(-0.3,0.31,0.01)
barlim = np.round(np.arange(-0.3,0.4,0.1),2)
cmap = cmocean.cm.balance

for i in range(len(preparedata)):
    ax = plt.subplot(5,2,location[i])
    
    var = preparedata[i]
    lat1 = lats
    lon1 = lons
    
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
                area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=1)
    m.drawstates(color='darkgrey',linewidth=0.5)
    m.drawcountries(color='darkgrey',linewidth=0.5)

    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgrey',
                      linewidth=0.7)
    circle.set_clip_on(False)
        
    lon2,lat2 = np.meshgrid(lon1,lat1)
    
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    
    cs1.set_cmap(cmap)
    
    ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(0.0,1.06),
              textcoords='axes fraction',color='k',fontsize=7,
              rotation=0,ha='center',va='center')
    
    if any([i==0,i==2,i==4,i==6]):
        ax.annotate(r'\textbf{%s}' % (variablenames[i]),xy=(0,0),xytext=(-0.1,0.5),
                  textcoords='axes fraction',color='dimgrey',fontsize=11,
                  rotation=90,ha='center',va='center') 
    elif i == 8:
        ax.annotate(r'\textbf{%s}' % ('frac_range'),xy=(0,0),xytext=(-0.1,0.5),
                  textcoords='axes fraction',color='dimgrey',fontsize=11,
                  rotation=90,ha='center',va='center') 
    if any([i==0,i==1]):
        ax.annotate(r'\textbf{%s}' % (modelnames[i]),xy=(0,0),xytext=(0.5,1.2),
                  textcoords='axes fraction',color='k',fontsize=7,
                  rotation=0,ha='center',va='center')       
    
cbar_ax1 = fig.add_axes([0.335,0.09,0.4,0.02])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=8,color='k',labelpad=5)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.13,top=0.93,hspace=0.15)
        
plt.savefig(directoryfigure + 'FutureLandChange_Historical-LM42_JJA.png',dpi=300)
