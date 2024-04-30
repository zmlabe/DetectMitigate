"""
Plot timeseries of GHG for presentation
 
Author    : Zachary M. Labe
Date      : 24 January 2024
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
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
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

variablesall = ['CO2']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
years = np.arange(2011,2100+1)
yearsall = np.arange(1921,2010+1,1)
yearstotal = np.arange(1921,2100+1,1)
yearsob = np.arange(1959,2023+1,1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Dark_Figures/'
directorydata = '/work/Zachary.Labe/Data/SPEAR/'
directorydata2 = '/home/Zachary.Labe/Research/DetectMitigate/Data/Observations/'
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
### Read in data for SPEAR_MED
data = Dataset(directorydata + 'SPEAR_MED/monthly/CO2/CO2_01_1921-2010.nc')
spear_co2_h = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED/monthly/CH4/CH4_01_1921-2010.nc')
spear_ch4_h = data.variables['CH4'][:]
data.close()

### Read in SSP370 data
data = Dataset(directorydata + 'SPEAR_MED_SSP370/monthly/CO2/CO2_01_2011-2100.nc')
SSP370_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP370/monthly/CH4/CH4_01_2011-2100.nc')
SSP370_ch4_f = data.variables['CH4'][:]
data.close()

### Read in SSP119 data
data = Dataset(directorydata + 'SPEAR_MED_SSP119/monthly/CO2/CO2_01_2011-2100.nc')
SSP119_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP119/monthly/CH4/CH4_01_2011-2100.nc')
SSP119_ch4_f = data.variables['CH4'][:]
data.close()

### Read in SSP245 data
data = Dataset(directorydata + 'SPEAR_MED_SSP245/monthly/CO2/CO2_01_2011-2100.nc')
SSP245_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP245/monthly/CH4/CH4_01_2011-2100.nc')
SSP245_ch4_f = data.variables['CH4'][:]
data.close()

### Read in SSP585
data = Dataset(directorydata + 'SPEAR_MED/monthly/CO2/CO2_01_2011-2100.nc')
spear_co2_585 = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED/monthly/CH4/CH4_01_2011-2100.nc')
spear_ch4_585 = data.variables['CH4'][:]
data.close()

### Combine SPEAR_MED
co2_spear = np.append(spear_co2_h,SSP370_co2_f,axis=0)
ch4_spear = np.append(spear_ch4_h,SSP370_ch4_f,axis=0)

############################################################################### 
###############################################################################
############################################################################### 
### Read in OS data
data = Dataset(directorydata + 'SPEAR_MED_SSP534OS/monthly/CO2/CO2_01_2011-2100.nc')
os_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP534OS/monthly/CH4/CH4_01_2011-2100.nc')
os_ch4_f = data.variables['CH4'][:]
data.close()

### Read in obs
obs = np.genfromtxt(directorydata2 + 'CO2_monthly_obs.txt',unpack=True,usecols=[4])

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
        
fig = plt.figure()
ax = plt.subplot(111)

adjust_spines(ax, ['left', 'bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both',length=4.,width=2,which='major',color='darkgrey')
ax.tick_params(axis='x',labelsize=5,pad=1.5)
ax.tick_params(axis='y',labelsize=5,pad=1.5)
# ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.4)


plt.plot(np.arange((1958-1921)*12,(2023-1921)*12,1),obs,linestyle='--',linewidth=2,color='r',
          label=r'\textbf{Observations}',clip_on=False,dashes=(1,0.9),zorder=30)
plt.plot(np.arange(0,len(co2_spear[:(2014-1921)*12]),1),co2_spear[:(2014-1921)*12],linestyle='-',linewidth=6,color='darkgrey',
          label=r'\textbf{Historical}',clip_on=False)
plt.plot(np.arange((2015-1921)*12,len(co2_spear),1),spear_co2_585[4*12:],linestyle='-',linewidth=5,color='darkred',
          label=r'\textbf{SSP5-8.5}',clip_on=False)
plt.plot(np.arange((2015-1921)*12,len(co2_spear),1),co2_spear[(2015-1921)*12:],linestyle='-',linewidth=5,color='tomato',
          label=r'\textbf{SSP3-7.0}',clip_on=False)
plt.plot(np.arange((2015-1921)*12,len(co2_spear),1),SSP245_co2_f[4*12:],linestyle='-',linewidth=5,color='gold',
          label=r'\textbf{SSP2-4.5}',clip_on=False)
plt.plot(np.arange((2015-1921)*12,len(co2_spear),1),SSP119_co2_f[4*12:],linestyle='-',linewidth=5,color='deepskyblue',
          label=r'\textbf{SSP1-1.9}',clip_on=False)
plt.plot(np.arange((2015-1921)*12,len(co2_spear),1),os_co2_f[4*12:],linestyle='-',linewidth=2,color='aqua',
          label=r'\textbf{SSP5-3.4OS}',clip_on=False,alpha=1)

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(0,2161,20*12),np.arange(1920,2101,20),fontsize=7)
plt.yticks(np.round(np.arange(0,2000,100),2),np.round(np.arange(0,2000,100),2),fontsize=7)
plt.xlim([0,2160])
plt.ylim([200,1200])

plt.text(2275,spear_co2_585[-1],r'\textbf{5-8.5}',color='darkred',ha='center',va='center',fontsize=9)
plt.text(2275,co2_spear[-1],r'\textbf{3-7.0}',color='tomato',ha='center',va='center',fontsize=9)
plt.text(2275,SSP245_co2_f[-1],r'\textbf{2-4.5}',color='gold',ha='center',va='center',fontsize=9)
plt.text(2275,SSP119_co2_f[-1],r'\textbf{1-1.9}',color='deepskyblue',ha='center',va='center',fontsize=9)
plt.text(2310,os_co2_f[-1],r'\textbf{5-3.4OS}',color='aqua',ha='center',va='center',fontsize=9)

plt.text(730,390,r'\textbf{Real World}',color='r',ha='center',va='center',fontsize=9,rotation=11)

plt.title(r'\textbf{CARBON DIOXIDE [CO$_{2}$]}',
                    color='w',fontsize=30)
plt.ylabel(r'\textbf{concentration [ppm]}',color='darkgrey',fontsize=10)

plt.tight_layout()
plt.savefig(directoryfigure + 'Dark_GHG-CO2.png',dpi=300)
