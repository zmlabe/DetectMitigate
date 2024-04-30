"""
Evaluate time series of GHG for overshoot runs
 
Author    : Zachary M. Labe
Date      : 22 May 2023
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
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

variablesall = ['T2M']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
years = np.arange(2011,2100+1)
yearsf = np.arange(2015,2100+1)
yearsall = np.arange(1921,2010+1,1)
yearstotal = np.arange(1921,2100+1,1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/MSFigures_Heat/'
directorydata = '/work/Zachary.Labe/Data/SPEAR/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','DIFFERENCE [a-b]']
dataset_obs = 'ERA5_MEDS'
seasons = ['annual']
slicemonthnamen = ['ANNUAL']
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

### Read in data for SPEAR_MED
data = Dataset(directorydata + 'SPEAR_MED/monthly/CO2/CO2_01_1921-2010.nc')
spear_co2_h = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED/monthly/CH4/CH4_01_1921-2010.nc')
spear_ch4_h = data.variables['CH4'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED/monthly/N2O/N2O_01_1921-2010.nc')
spear_n2o_h = data.variables['N2O'][:]
data.close()

data = Dataset(directorydata + 'SPEAR_MED/monthly/CO2/CO2_01_2011-2100.nc')
spear_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED/monthly/CH4/CH4_01_2011-2100.nc')
spear_ch4_f = data.variables['CH4'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED/monthly/N2O/N2O_01_2011-2100.nc')
spear_n2o_f = data.variables['N2O'][:]
data.close()

### Combine SPEAR_MED
co2_spear = np.append(spear_co2_h,spear_co2_f,axis=0).reshape(len(yearstotal),12)
ch4_spear = np.append(spear_ch4_h,spear_ch4_f,axis=0).reshape(len(yearstotal),12)
n2o_spear = np.append(spear_n2o_h,spear_n2o_f,axis=0).reshape(len(yearstotal),12)

### Read in OS data
data = Dataset(directorydata + 'SPEAR_MED_SSP534OS/monthly/CO2/CO2_01_2011-2100.nc')
os_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP534OS/monthly/CH4/CH4_01_2011-2100.nc')
os_ch4_f = data.variables['CH4'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP534OS/monthly/N2O/N2O_01_2011-2100.nc')
os_n2o_f = data.variables['N2O'][:]
data.close()

### Read in OS_10ye data
data = Dataset(directorydata + 'SPEAR_MED_SSP534OS_10ye/monthly/CO2/CO2_01_2031-2100.nc')
os_10ye_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP534OS_10ye/monthly/CH4/CH4_01_2031-2100.nc')
os_10ye_ch4_f = data.variables['CH4'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP534OS_10ye/monthly/N2O/N2O_01_2031-2100.nc')
os_10ye_n2o_f = data.variables['N2O'][:]
data.close()

### Read in SSP245 data
data = Dataset(directorydata + 'SPEAR_MED_SSP245/monthly/CO2/CO2_01_2011-2100.nc')
SSP245_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP245/monthly/CH4/CH4_01_2011-2100.nc')
SSP245_ch4_f = data.variables['CH4'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP245/monthly/N2O/N2O_01_2011-2100.nc')
SSP245_n2o_f = data.variables['N2O'][:]
data.close()

### Combine OS_10ye
diffmo = len(os_n2o_f) - len(os_10ye_n2o_f)
os_10ye_co2_allf = np.append(os_co2_f[:diffmo],os_10ye_co2_f,axis=0).reshape(len(yearsall),12)
os_10ye_ch4_allf = np.append(os_ch4_f[:diffmo],os_10ye_ch4_f,axis=0).reshape(len(yearsall),12)
os_10ye_n2o_allf = np.append(os_n2o_f[:diffmo],os_10ye_n2o_f,axis=0).reshape(len(yearsall),12)

### Reshape OS
os_co2_f = os_co2_f.reshape(len(yearsall),12)
os_ch4_f = os_ch4_f.reshape(len(yearsall),12)
os_n2o_f = os_n2o_f.reshape(len(yearsall),12)

### Reshape SSP245
SSP245_co2_f = SSP245_co2_f.reshape(len(yearsall),12)
SSP245_ch4_f = SSP245_ch4_f.reshape(len(yearsall),12)
SSP245_n2o_f = SSP245_n2o_f.reshape(len(yearsall),12)

### Calculate annual means
co2_spear_a = np.nanmean(co2_spear,axis=1)
ch4_spear_a = np.nanmean(ch4_spear,axis=1)
n2o_spear_a = np.nanmean(n2o_spear,axis=1)

os_10ye_co2_allf_a = np.nanmean(os_10ye_co2_allf,axis=1)
os_10ye_ch4_allf_a = np.nanmean(os_10ye_ch4_allf,axis=1)
os_10ye_n2o_allf_a = np.nanmean(os_10ye_n2o_allf,axis=1)

os_co2_f_a = np.nanmean(os_co2_f,axis=1)
os_ch4_f_a = np.nanmean(os_ch4_f,axis=1)
os_n2o_f_a = np.nanmean(os_n2o_f,axis=1)

SSP245_co2_f_a = np.nanmean(SSP245_co2_f,axis=1)
SSP245_ch4_f_a = np.nanmean(SSP245_ch4_f,axis=1)
SSP245_n2o_f_a = np.nanmean(SSP245_n2o_f,axis=1)

### Read in GMST data
lat_bounds,lon_bounds = UT.regions(reg_name)
spear_osm,lats,lons = read_primary_dataset('T2M','SPEAR_MED_Scenario',monthlychoice,'SSP534OS',lat_bounds,lon_bounds)
spear_osm_10ye,lats,lons = read_primary_dataset('T2M','SPEAR_MED_SSP534OS_10ye',monthlychoice,'SSP534OS_10ye',lat_bounds,lon_bounds)
lon2,lat2 = np.meshgrid(lons,lats)

### Calculate global means
ave_os = UT.calc_weightedAve(spear_osm,lat2)
ave_os10ye = UT.calc_weightedAve(spear_osm_10ye,lat2)

### Calculate ensemble means
meanAve_os = np.nanmean(ave_os[:,:],axis=0)
meanAve_os10ye = np.nanmean(ave_os10ye[:,:],axis=0)

### Calculate warmest year
maxWhere_os = np.argmax(meanAve_os)
maxWhere_os10ye = np.argmax(meanAve_os10ye)

yearMax_os = yearsf[maxWhere_os]
yearMax_os10ye = yearsf[maxWhere_os10ye]

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
        
fig = plt.figure(figsize=(7,9))
ax = plt.subplot(311)

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

plt.axvline(x=2040,linewidth=1.5,color='darkslategrey',linestyle=':')
plt.axvline(x=2031,linewidth=1.5,color='lightseagreen',linestyle=':')
plt.axvline(x=yearMax_os,color='darkslategrey',linewidth=2,linestyle='-')
plt.axvline(x=yearMax_os10ye,color='lightseagreen',linewidth=2,linestyle='-')

plt.plot(yearstotal,co2_spear_a,linestyle='-',linewidth=4,color='maroon',
          label=r'\textbf{SPEAR_MED_SSP585}')
plt.plot(years,SSP245_co2_f_a,linestyle='-',linewidth=2,color='darkorange',
          label=r'\textbf{SPEAR_MED_SSP245}')
plt.plot(years,os_co2_f_a,linestyle='-',linewidth=4,color='darkslategrey',
          label=r'\textbf{SPEAR_MED_SSP534OS}')
plt.plot(years[diffmo//12:],os_10ye_co2_allf_a[diffmo//12:],linestyle='--',linewidth=4,color='lightseagreen',
          label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',dashes=(1,0.3))

# leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
#       bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
#       handlelength=1,handletextpad=0.5)
# for line,text in zip(leg.get_lines(), leg.get_texts()):
#     text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=8.5)
plt.yticks(np.round(np.arange(0,2000,100),2),np.round(np.arange(0,2000,100),2),fontsize=8.5)
plt.xlim([2015,2100])
plt.ylim([300,1200])

plt.text(2015,1200,r'\textbf{[a]}',fontsize=10,color='k')
plt.title(r'\textbf{Carbon Dioxide [CO$_{2}$]}',
                    color='k',fontsize=14)
plt.ylabel(r'\textbf{concentration [ppm]}',color='dimgrey',fontsize=10)

############################################################################### 
ax = plt.subplot(312)

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

plt.axvline(x=2040,linewidth=1.5,color='darkslategrey',linestyle=':')
plt.axvline(x=2031,linewidth=1.5,color='lightseagreen',linestyle=':')
plt.axvline(x=yearMax_os,color='darkslategrey',linewidth=2,linestyle='-')
plt.axvline(x=yearMax_os10ye,color='lightseagreen',linewidth=2,linestyle='-')

plt.plot(yearstotal,ch4_spear_a,linestyle='-',linewidth=4,color='maroon',
          label=r'\textbf{SPEAR_MED_SSP585}')
plt.plot(years,SSP245_ch4_f_a,linestyle='-',linewidth=2,color='darkorange',
          label=r'\textbf{SPEAR_MED_SSP245}')
plt.plot(years,os_ch4_f_a,linestyle='-',linewidth=4,color='darkslategrey',
          label=r'\textbf{SPEAR_MED_SSP534OS}')
plt.plot(years[diffmo//12:],os_10ye_ch4_allf_a[diffmo//12:],linestyle='--',linewidth=4,color='lightseagreen',
          label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',dashes=(1,0.3))

# leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
#       bbox_to_anchor=(0.23,0.95),fancybox=True,ncol=1,frameon=False,
#       handlelength=1,handletextpad=0.7)
# for line,text in zip(leg.get_lines(), leg.get_texts()):
#     text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=8.5)
plt.yticks(np.round(np.arange(0,5000,200),2),np.round(np.arange(0,5000,200),2),fontsize=8.5)
plt.xlim([2015,2100])
plt.ylim([800,3000])

plt.text(2015,3000,r'\textbf{[b]}',fontsize=10,color='k')
plt.title(r'\textbf{Methane [CH$_{4}$]}',
                    color='k',fontsize=14)
plt.ylabel(r'\textbf{concentration [ppb]}',color='dimgrey',fontsize=10)

############################################################################### 
ax = plt.subplot(313)

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

plt.axvline(x=2040,linewidth=1.5,color='darkslategrey',linestyle=':')
plt.axvline(x=2031,linewidth=1.5,color='lightseagreen',linestyle=':')
plt.axvline(x=yearMax_os,color='darkslategrey',linewidth=2,linestyle='-')
plt.axvline(x=yearMax_os10ye,color='lightseagreen',linewidth=2,linestyle='-')

plt.plot(yearstotal,n2o_spear_a,linestyle='-',linewidth=4,color='maroon',
          label=r'\textbf{SPEAR_MED_SSP585}')
plt.plot(years,SSP245_n2o_f_a,linestyle='-',linewidth=2,color='darkorange',
          label=r'\textbf{SPEAR_MED_SSP245}')
plt.plot(years,os_n2o_f_a,linestyle='-',linewidth=4,color='darkslategrey',
          label=r'\textbf{SPEAR_MED_SSP534OS}')
plt.plot(years[diffmo//12:],os_10ye_n2o_allf_a[diffmo//12:],linestyle='--',linewidth=4,color='lightseagreen',
          label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',dashes=(1,0.3))

leg = plt.legend(shadow=False,fontsize=12.5,loc='upper center',
      bbox_to_anchor=(0.8,0.46),fancybox=True,ncol=1,frameon=False,
      handlelength=3,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=8.5)
plt.yticks(np.round(np.arange(0,5000,10),2),np.round(np.arange(0,5000,10),2),fontsize=8.5)
plt.xlim([2015,2100])
plt.ylim([320,400])

plt.text(2015,400,r'\textbf{[c]}',fontsize=10,color='k')
plt.title(r'\textbf{Nitrous Oxide [N$_{2}$O]}',
                    color='k',fontsize=14)
plt.ylabel(r'\textbf{concentration [ppb]}',color='dimgrey',fontsize=10)

plt.tight_layout()        
   
### Save figure
plt.savefig(directoryfigure+'MSFigure_Heatwave_TimeSeries_GHG-GMST.png',dpi=600)
    
