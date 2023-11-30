"""
Evaluate time series of GHG for all scenarios
 
Author    : Zachary M. Labe
Date      : 30 November 2023
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
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
directorydata = '/work/Zachary.Labe/Data/SPEAR/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
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

### Read in SSP370 data
data = Dataset(directorydata + 'SPEAR_MED_SSP370/monthly/CO2/CO2_01_2011-2100.nc')
SSP370_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP370/monthly/CH4/CH4_01_2011-2100.nc')
SSP370_ch4_f = data.variables['CH4'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP370/monthly/N2O/N2O_01_2011-2100.nc')
SSP370_n2o_f = data.variables['N2O'][:]
data.close()

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

### Read in SSP119 data
data = Dataset(directorydata + 'SPEAR_MED_SSP119/monthly/CO2/CO2_01_2011-2100.nc')
SSP119_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP119/monthly/CH4/CH4_01_2011-2100.nc')
SSP119_ch4_f = data.variables['CH4'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP119/monthly/N2O/N2O_01_2011-2100.nc')
SSP119_n2o_f = data.variables['N2O'][:]
data.close()

### Combine OS_10ye
diffmo = len(os_n2o_f) - len(os_10ye_n2o_f)
os_10ye_co2_allf = np.append(os_co2_f[:diffmo],os_10ye_co2_f,axis=0).reshape(len(yearsall),12)
os_10ye_ch4_allf = np.append(os_ch4_f[:diffmo],os_10ye_ch4_f,axis=0).reshape(len(yearsall),12)
os_10ye_n2o_allf = np.append(os_n2o_f[:diffmo],os_10ye_n2o_f,axis=0).reshape(len(yearsall),12)

### Reshape SSP370
SSP370_co2_f = SSP370_co2_f.reshape(len(yearsall),12)
SSP370_ch4_f = SSP370_ch4_f.reshape(len(yearsall),12)
SSP370_n2o_f = SSP370_n2o_f.reshape(len(yearsall),12)

### Reshape OS
os_co2_f = os_co2_f.reshape(len(yearsall),12)
os_ch4_f = os_ch4_f.reshape(len(yearsall),12)
os_n2o_f = os_n2o_f.reshape(len(yearsall),12)

### Reshape SSP245
SSP245_co2_f = SSP245_co2_f.reshape(len(yearsall),12)
SSP245_ch4_f = SSP245_ch4_f.reshape(len(yearsall),12)
SSP245_n2o_f = SSP245_n2o_f.reshape(len(yearsall),12)

### Reshape SSP245
SSP119_co2_f = SSP119_co2_f.reshape(len(yearsall),12)
SSP119_ch4_f = SSP119_ch4_f.reshape(len(yearsall),12)
SSP119_n2o_f = SSP119_n2o_f.reshape(len(yearsall),12)

############################################################################### 
###############################################################################
############################################################################### 
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

SSP119_co2_f_a = np.nanmean(SSP119_co2_f,axis=1)
SSP119_ch4_f_a = np.nanmean(SSP119_ch4_f,axis=1)
SSP119_n2o_f_a = np.nanmean(SSP119_n2o_f,axis=1)

SSP370_co2_f_a = np.nanmean(SSP370_co2_f,axis=1)
SSP370_ch4_f_a = np.nanmean(SSP370_ch4_f,axis=1)
SSP370_n2o_f_a = np.nanmean(SSP370_n2o_f,axis=1)

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

plt.plot(yearstotal,co2_spear_a,linestyle='-',linewidth=4,color='r',
          label=r'\textbf{SPEAR_MED_SSP585}')
plt.plot(years[4:],SSP370_co2_f_a[4:],linestyle='-',linewidth=2,color='maroon',
          label=r'\textbf{SPEAR_MED_SSP370}',clip_on=False)
plt.plot(years[4:],SSP245_co2_f_a[4:],linestyle='-',linewidth=2,color='darkorange',
          label=r'\textbf{SPEAR_MED_SSP245}',clip_on=False)
plt.plot(years[4:],os_co2_f_a[4:],linestyle='-',linewidth=4,color='darkslategrey',
          label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False)
plt.plot(years[diffmo//12:],os_10ye_co2_allf_a[diffmo//12:],linestyle='--',linewidth=4,color='lightseagreen',
          label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',dashes=(1,0.3),clip_on=False)
plt.plot(years[4:],SSP119_co2_f_a[4:],linestyle='-',linewidth=2,color='aqua',
          label=r'\textbf{SPEAR_MED_SSP119}',clip_on=False)

# leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
#       bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
#       handlelength=1,handletextpad=0.5)
# for line,text in zip(leg.get_lines(), leg.get_texts()):
#     text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=8.5)
plt.yticks(np.round(np.arange(0,2000,100),2),np.round(np.arange(0,2000,100),2),fontsize=8.5)
plt.xlim([2010,2100])
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

plt.plot(yearstotal,ch4_spear_a,linestyle='-',linewidth=4,color='r',
          label=r'\textbf{SPEAR_MED_SSP585}')
plt.plot(years[4:],SSP370_ch4_f_a[4:],linestyle='-',linewidth=2,color='maroon',
          label=r'\textbf{SPEAR_MED_SSP370}',clip_on=False)
plt.plot(years[4:],SSP245_ch4_f_a[4:],linestyle='-',linewidth=2,color='darkorange',
          label=r'\textbf{SPEAR_MED_SSP245}',clip_on=False)
plt.plot(years[4:],os_ch4_f_a[4:],linestyle='-',linewidth=4,color='darkslategrey',
          label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False)
plt.plot(years[diffmo//12:],os_10ye_ch4_allf_a[diffmo//12:],linestyle='--',linewidth=4,color='lightseagreen',
          label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',dashes=(1,0.3))
plt.plot(years[4:],SSP119_ch4_f_a[4:],linestyle='-',linewidth=2,color='aqua',
          label=r'\textbf{SPEAR_MED_SSP119}',clip_on=False)

# leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
#       bbox_to_anchor=(0.23,0.95),fancybox=True,ncol=1,frameon=False,
#       handlelength=1,handletextpad=0.7)
# for line,text in zip(leg.get_lines(), leg.get_texts()):
#     text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=8.5)
plt.yticks(np.round(np.arange(0,5000,500),2),np.round(np.arange(0,5000,500),2),fontsize=8.5)
plt.xlim([2010,2100])
plt.ylim([500,3500])

plt.text(2015,3400,r'\textbf{[b]}',fontsize=10,color='k')
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

plt.plot(yearstotal,n2o_spear_a,linestyle='-',linewidth=4,color='r',
          label=r'\textbf{SPEAR_MED_SSP585}')
plt.plot(years[4:],SSP370_n2o_f_a[4:],linestyle='-',linewidth=2,color='maroon',
          label=r'\textbf{SPEAR_MED_SSP370}',clip_on=False)
plt.plot(years[4:],SSP245_n2o_f_a[4:],linestyle='-',linewidth=2,color='darkorange',
          label=r'\textbf{SPEAR_MED_SSP245}',clip_on=False)
plt.plot(years[4:],os_n2o_f_a[4:],linestyle='-',linewidth=4,color='darkslategrey',
          label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False)
plt.plot(years[diffmo//12:],os_10ye_n2o_allf_a[diffmo//12:],linestyle='--',linewidth=4,color='lightseagreen',
          label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',dashes=(1,0.3),clip_on=False)
plt.plot(years[4:],SSP119_n2o_f_a[4:],linestyle='-',linewidth=2,color='aqua',
          label=r'\textbf{SPEAR_MED_SSP119}',clip_on=False)

leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
      bbox_to_anchor=(0.3,0.9),fancybox=True,ncol=2,frameon=False,
      handlelength=1.5,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10),fontsize=8.5)
plt.yticks(np.round(np.arange(0,5000,10),2),np.round(np.arange(0,5000,10),2),fontsize=8.5)
plt.xlim([2010,2100])
plt.ylim([320,420])

plt.text(2015,420,r'\textbf{[c]}',fontsize=10,color='k')
plt.title(r'\textbf{Nitrous Oxide [N$_{2}$O]}',
                    color='k',fontsize=14)
plt.ylabel(r'\textbf{concentration [ppb]}',color='dimgrey',fontsize=10)

plt.tight_layout()        
   
### Save figure
plt.savefig(directoryfigure+'TimeSeries_All-Scenarios_GHG.png',dpi=600)
    
