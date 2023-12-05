"""
Create new climate scenarios
 
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
yearstotal_extend = np.arange(1921,2101+1,1)
yearsrepeat = np.repeat(yearstotal,12)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/OS2040a/'
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
### Create climate scenarios
def draw_curve(p1, p2, p3, length):
    f = np.poly1d(np.polyfit((p1[0], p2[0], p3[0]), (p1[1], p2[1], p3[1]), 2))
    x = np.linspace(p1[0], p2[0], length)
    return x, f(x)

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

### Read in OS_10ye data
data = Dataset(directorydata + 'SPEAR_MED_SSP534OS_10ye/monthly/CO2/CO2_01_2031-2100.nc')
os_10ye_co2_f = data.variables['CO2'][:]
data.close()
data = Dataset(directorydata + 'SPEAR_MED_SSP534OS_10ye/monthly/CH4/CH4_01_2031-2100.nc')
os_10ye_ch4_f = data.variables['CH4'][:]
data.close()

### Combine OS_10ye
diffmo = len(os_co2_f) - len(os_10ye_co2_f)
os_10ye_co2_allf = np.append(os_co2_f[:diffmo],os_10ye_co2_f,axis=0)
os_10ye_ch4_allf = np.append(os_ch4_f[:diffmo],os_10ye_ch4_f,axis=0)

############################################################################### 
###############################################################################
###############################################################################
### CO2
### Hypothetical decrease 2040
yearq_S2040 = np.where(yearstotal == 2040)[0][0]*12
os_S2040 = np.where(years == 2040)[0][0]*12
diff = os_co2_f[os_S2040] - co2_spear[yearq_S2040]
scale_co2 = os_co2_f[os_S2040:] - diff
os2040a = co2_spear.copy()
os2040a[yearq_S2040:] = scale_co2
os2040a = np.append(os2040a,os2040a[-12:] - (os2040a[-24:-12] - os2040a[-12:]))

x_S2040 = np.linspace(yearq_S2040+1,len(yearstotal_extend)*12, (len(yearstotal_extend)*12) - yearq_S2040)
y_S2040 = os2040a[yearq_S2040:]

############################################################################### 
###############################################################################
############################################################################### 
### CH4
### Hypothetical decrease 2020
yearq_S2040 = np.where(yearstotal == 2040)[0][0]*12
os_S2040 = np.where(years == 2040)[0][0]*12
diff_ch4 = os_ch4_f[os_S2040] - ch4_spear[yearq_S2040]
scale_ch4 = os_ch4_f[os_S2040:] - diff_ch4
os2040a_ch4 = ch4_spear.copy()
os2040a_ch4[yearq_S2040:] = scale_ch4
os2040a_ch4 = np.append(os2040a_ch4,os2040a_ch4[-12:] - (os2040a_ch4[-24:-12] - os2040a_ch4[-12:]))

x_S2040_ch4 = np.linspace(yearq_S2040+1,len(yearstotal_extend)*12, (len(yearstotal_extend)*12) - yearq_S2040)
y_S2040_ch4 = os2040a_ch4[yearq_S2040:]


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
        
fig = plt.figure(figsize=(11,9))
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

plt.plot(np.arange((2015-1921)*12,len(co2_spear),1),os_co2_f[4*12:],linestyle='-',linewidth=4,color='darkslategrey',
          label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False,alpha=0.3)
plt.plot(np.arange((2031-1921)*12,len(co2_spear),1),os_10ye_co2_allf[diffmo:],linestyle='--',linewidth=4,color='lightseagreen',
          label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',dashes=(1,0.3),alpha=0.3)

plt.plot(co2_spear,linestyle='-',linewidth=4,color='maroon',
          label=r'\textbf{SPEAR_MED_SSP370}',clip_on=False)
plt.plot(np.arange((2015-1921)*12,len(co2_spear),1),spear_co2_585[4*12:],linestyle=':',linewidth=6,color='tomato',
          label=r'\textbf{SPEAR_MED_SSP585}',clip_on=False)
plt.plot(np.arange((2015-1921)*12,len(co2_spear),1),SSP245_co2_f[4*12:],linestyle='-',linewidth=2,color='k',
          label=r'\textbf{SPEAR_MED_SSP245}',clip_on=False)
plt.plot(x_S2040,y_S2040,linestyle='--',linewidth=2,color='peru',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2040a}',clip_on=False)

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(0,2161,20*12),np.arange(1920,2101,20),fontsize=8.5)
plt.yticks(np.round(np.arange(0,2000,100),2),np.round(np.arange(0,2000,100),2),fontsize=8.5)
plt.xlim([0,2160])
plt.ylim([200,1200])

plt.text(0,900,r'\textbf{[a]}',fontsize=10,color='k')
plt.title(r'\textbf{Carbon Dioxide [CO$_{2}$]}',
                    color='k',fontsize=14)
plt.ylabel(r'\textbf{concentration [ppm]}',color='dimgrey',fontsize=10)

############################################################################### 
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

plt.plot(np.arange((2015-1921)*12,len(ch4_spear),1),os_ch4_f[4*12:],linestyle='-',linewidth=2,color='darkslategrey',
          label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False,alpha=0.6)
plt.plot(np.arange((2031-1921)*12,len(ch4_spear),1),os_10ye_ch4_allf[diffmo:],linestyle='--',linewidth=2,color='lightseagreen',
          label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',dashes=(1,0.3),alpha=0.6)

plt.plot(ch4_spear,linestyle='-',linewidth=4,color='maroon',
          label=r'\textbf{SPEAR_MED_SSP370}',clip_on=False)
plt.plot(np.arange((2015-1921)*12,len(ch4_spear),1),spear_ch4_585[4*12:],linestyle=':',linewidth=6,color='tomato',
          label=r'\textbf{SPEAR_MED_SSP585}',clip_on=False)
plt.plot(np.arange((2015-1921)*12,len(ch4_spear),1),SSP245_ch4_f[4*12:],linestyle='-',linewidth=2,color='k',
          label=r'\textbf{SPEAR_MED_SSP245}',clip_on=False)
plt.plot(x_S2040_ch4,y_S2040_ch4,linestyle='--',linewidth=2,color='peru',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2040a}',clip_on=False)

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(0,2161,20*12),np.arange(1920,2101,20),fontsize=8.5)
plt.yticks(np.round(np.arange(0,5000,200),2),np.round(np.arange(0,5000,200),2),fontsize=8.5)
plt.xlim([0,2160])
plt.ylim([800,3400])

plt.text(0,3400,r'\textbf{[b]}',fontsize=10,color='k')
plt.title(r'\textbf{Methane [CH$_{4}$]}',
                    color='k',fontsize=14)
plt.ylabel(r'\textbf{concentration [ppb]}',color='dimgrey',fontsize=10)

plt.tight_layout()        
   
### Save figure
plt.savefig(directoryfigure+'TimeSeries_All-Scenarios_GHG_hypothetical_OS2040a.png',dpi=600)

############################################################################### 
############################################################################### 
############################################################################### 
### Calculate data files

def concFiles_co2(data,startyr,spear,yearstotal_extend):
    ### Cat 1921 to 210
    dataall = np.append(spear[:-len(data)+12],data)
    
    ### Reshape
    datamonths = np.reshape(dataall,(len(yearstotal_extend),12))
    dataannual = np.nanmean(datamonths,axis=1)
    
    ### Read in raw data
    dataco2 = np.loadtxt('/home/Zachary.Labe/Research/DetectMitigate/Data/co2_gblannualdata_ssp534os_from_wfc',skiprows=1)
    xx_co2 = dataco2[:,0]
    modelyears = dataco2[:,1]
    modelcopy = modelyears.copy()
    modelyears[1920:2101] = dataannual
    
    plt.figure()
    plt.plot(modelcopy)
    plt.plot(modelyears,color='r')
    
    plt.figure()
    plt.plot(dataannual)
    
    np.savetxt(directoryoutput + 'co2_gblannualdata_SSP370-OS%sa_from_wfc' % startyr,
               np.c_[xx_co2,modelyears],fmt='%1.4f',header='2500',comments='')
    
    return dataall,dataannual,modelyears
    
all_2040,annual_2040,modelyrs_2040 = concFiles_co2(y_S2040,2040,co2_spear,yearstotal_extend)

def concFiles_ch4(data,startyr,spear,yearstotal_extend):
    ### Cat 1921 to 210
    dataall = np.append(spear[:-len(data)+12],data)
    
    ### Reshape
    datamonths = np.reshape(dataall,(len(yearstotal_extend),12))
    dataannual = np.nanmean(datamonths,axis=1)
    
    ### Read in raw data
    datach4 = np.loadtxt('/home/Zachary.Labe/Research/DetectMitigate/Data/ch4_gblannualdata_ssp534os_from_wfc',skiprows=1)
    xx_ch4 = datach4[:,0]
    modelyears = datach4[:,1]
    modelcopy = modelyears.copy()
    modelyears[1920:2101] = dataannual
    
    plt.figure()
    plt.plot(modelcopy)
    plt.plot(modelyears,color='r')
    
    plt.figure()
    plt.plot(dataannual)
    
    np.savetxt(directoryoutput + 'ch4_gblannualdata_SSP370-OS%sa_from_wfc' % startyr,
               np.c_[xx_ch4,modelyears],fmt='%1.4f',header='2500',comments='')
    
    return dataall,dataannual,modelyears

all_2040_ch4,annual_2040_ch4,modelyrs_2040_ch4 = concFiles_ch4(y_S2040_ch4,2040,ch4_spear,yearstotal_extend)
