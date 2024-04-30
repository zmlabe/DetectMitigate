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
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/'
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

### Combine SPEAR_MED
co2_spear = np.append(spear_co2_h,SSP370_co2_f,axis=0)
ch4_spear = np.append(spear_ch4_h,SSP370_ch4_f,axis=0)

### Read in raw data
tryyears = np.arange(1850,2100+1,1)
dataco2 = np.loadtxt(directoryoutput + 'co2_gblannualdata_ssp534os_from_wfc',skiprows=1)
xx_co2 = dataco2[1980:,0]
yy_co2 = dataco2[1980:,1]
datach4 = np.loadtxt(directoryoutput + 'ch4_gblannualdata_ssp534os_from_wfc',skiprows=1)
xx_ch4 = datach4[1980:,0]
yy_ch4 = datach4[1980:,1]

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
### CO2
### Hypothetical decrease 2020
yearq_S2020 = np.where(yearstotal == 2020)[0][0]*12
p1_S2020 = [yearq_S2020+1,co2_spear[yearq_S2020]]
p2_S2020 = [len(yearstotal_extend)*12,co2_spear[0]]
length_S2020 = (len(yearstotal_extend)*12) - yearq_S2020
p3_S2020 = [yearq_S2020+120,co2_spear[yearq_S2020+120]]
x_S2020, y_S2020 = draw_curve(p1_S2020, p2_S2020, p3_S2020, length_S2020)

### 2030
yearq_S2030 = np.where(yearstotal == 2030)[0][0]*12
diff_S2030 = co2_spear[yearq_S2030] - co2_spear[yearq_S2020]
x_S2030 = x_S2020[10*12:]
y_S2030 = y_S2020[:-10*12] + diff_S2030

### 2040
yearq_S2040 = np.where(yearstotal == 2040)[0][0]*12
diff_S2040 = co2_spear[yearq_S2040] - co2_spear[yearq_S2020]
x_S2040 = x_S2020[20*12:]
y_S2040 = y_S2020[:-20*12] + diff_S2040

### 2050
yearq_S2050 = np.where(yearstotal == 2050)[0][0]*12
diff_S2050 = co2_spear[yearq_S2050] - co2_spear[yearq_S2020]
x_S2050 = x_S2020[30*12:]
y_S2050 = y_S2020[:-30*12] + diff_S2050

### 2060
yearq_S2060 = np.where(yearstotal == 2060)[0][0]*12
diff_S2060 = co2_spear[yearq_S2060] - co2_spear[yearq_S2020]
x_S2060 = x_S2020[40*12:]
y_S2060 = y_S2020[:-40*12] + diff_S2060

### 2070
yearq_S2070 = np.where(yearstotal == 2070)[0][0]*12
diff_S2070 = co2_spear[yearq_S2070] - co2_spear[yearq_S2020]
x_S2070 = x_S2020[50*12:]
y_S2070 = y_S2020[:-50*12] + diff_S2070

### 2080
yearq_S2080 = np.where(yearstotal == 2080)[0][0]*12
diff_S2080 = co2_spear[yearq_S2080] - co2_spear[yearq_S2020]
x_S2080 = x_S2020[60*12:]
y_S2080 = y_S2020[:-60*12] + diff_S2080

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
### CH4
### Hypothetical decrease 2020
yearq_S2020 = np.where(yearstotal == 2020)[0][0]*12
p1_S2020_ch4 = [yearq_S2020+1,ch4_spear[yearq_S2020]]
p2_S2020_ch4 = [len(yearstotal_extend)*12,ch4_spear[0]]
length_S2020_ch4 = (len(yearstotal_extend)*12) - yearq_S2020
p3_S2020_ch4 = [yearq_S2020+120,ch4_spear[yearq_S2020+120]]
x_S2020_ch4, y_S2020_ch4 = draw_curve(p1_S2020_ch4, p2_S2020_ch4, p3_S2020_ch4, length_S2020_ch4)

### 2030
yearq_S2030 = np.where(yearstotal == 2030)[0][0]*12
diff_S2030_ch4 = ch4_spear[yearq_S2030] - ch4_spear[yearq_S2020]
x_S2030_ch4 = x_S2020_ch4[10*12:]
y_S2030_ch4 = y_S2020_ch4[:-10*12] + diff_S2030_ch4

### 2040
yearq_S2040 = np.where(yearstotal == 2040)[0][0]*12
diff_S2040_ch4 = ch4_spear[yearq_S2040] - ch4_spear[yearq_S2020]
x_S2040_ch4 = x_S2020_ch4[20*12:]
y_S2040_ch4 = y_S2020_ch4[:-20*12] + diff_S2040_ch4

### 2050
yearq_S2050 = np.where(yearstotal == 2050)[0][0]*12
diff_S2050_ch4 = ch4_spear[yearq_S2050] - ch4_spear[yearq_S2020]
x_S2050_ch4 = x_S2020_ch4[30*12:]
y_S2050_ch4 = y_S2020_ch4[:-30*12] + diff_S2050_ch4

### 2060
yearq_S2060 = np.where(yearstotal == 2060)[0][0]*12
diff_S2060_ch4 = ch4_spear[yearq_S2060] - ch4_spear[yearq_S2020]
x_S2060_ch4 = x_S2020_ch4[40*12:]
y_S2060_ch4 = y_S2020_ch4[:-40*12] + diff_S2060_ch4

### 2070
yearq_S2070 = np.where(yearstotal == 2070)[0][0]*12
diff_S2070_ch4 = ch4_spear[yearq_S2070] - ch4_spear[yearq_S2020]
x_S2070_ch4 = x_S2020_ch4[50*12:]
y_S2070_ch4 = y_S2020_ch4[:-50*12] + diff_S2070_ch4

### 2080
yearq_S2080 = np.where(yearstotal == 2080)[0][0]*12
diff_S2080_ch4 = ch4_spear[yearq_S2080] - ch4_spear[yearq_S2020]
x_S2080_ch4 = x_S2020_ch4[60*12:]
y_S2080_ch4 = y_S2020_ch4[:-60*12] + diff_S2080_ch4

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
plt.plot(x_S2020,y_S2020,linestyle='--',linewidth=2,color='gold',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2020}',clip_on=False)
plt.plot(x_S2030,y_S2030,linestyle='--',linewidth=2,color='tan',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2030}',clip_on=False)
plt.plot(x_S2040,y_S2040,linestyle='--',linewidth=2,color='peru',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2040}',clip_on=False)
plt.plot(x_S2050,y_S2050,linestyle='--',linewidth=2,color='darkorange',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2050}',clip_on=False)
plt.plot(x_S2060,y_S2060,linestyle='--',linewidth=2,color='tomato',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2060}',clip_on=False)
plt.plot(x_S2070,y_S2070,linestyle='--',linewidth=2,color='r',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2070}',clip_on=False)
plt.plot(x_S2080,y_S2080,linestyle='--',linewidth=2,color='indigo',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2080}',clip_on=False)

leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0.5,1.02),fancybox=True,ncol=2,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(0,2161,20*12),np.arange(1920,2101,20),fontsize=8.5)
plt.yticks(np.round(np.arange(0,2000,100),2),np.round(np.arange(0,2000,100),2),fontsize=8.5)
plt.xlim([0,2160])
plt.ylim([200,900])

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

plt.plot(np.arange((2015-1921)*12,len(ch4_spear),1),os_ch4_f[4*12:],linestyle='-',linewidth=4,color='darkslategrey',
          label=r'\textbf{SPEAR_MED_SSP534OS}',clip_on=False,alpha=0.3)
plt.plot(np.arange((2031-1921)*12,len(ch4_spear),1),os_10ye_ch4_allf[diffmo:],linestyle='--',linewidth=4,color='lightseagreen',
          label=r'\textbf{SPEAR_MED_SSP534OS_10ye}',dashes=(1,0.3),alpha=0.3)

plt.plot(ch4_spear,linestyle='-',linewidth=4,color='maroon',
          label=r'\textbf{SPEAR_MED_SSP370}',clip_on=False)
plt.plot(x_S2020_ch4,y_S2020_ch4,linestyle='--',linewidth=2,color='gold',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2020}',clip_on=False)
plt.plot(x_S2030_ch4,y_S2030_ch4,linestyle='--',linewidth=2,color='tan',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2030}',clip_on=False)
plt.plot(x_S2040_ch4,y_S2040_ch4,linestyle='--',linewidth=2,color='peru',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2040}',clip_on=False)
plt.plot(x_S2050_ch4,y_S2050_ch4,linestyle='--',linewidth=2,color='darkorange',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2050}',clip_on=False)
plt.plot(x_S2060_ch4,y_S2060_ch4,linestyle='--',linewidth=2,color='tomato',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2060}',clip_on=False)
plt.plot(x_S2070_ch4,y_S2070_ch4,linestyle='--',linewidth=2,color='r',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2070}',clip_on=False)
plt.plot(x_S2080_ch4,y_S2080_ch4,linestyle='--',linewidth=2,color='indigo',dashes=(1,0.3),
          label=r'\textbf{SPEAR_MED_SSP370_OS2080}',clip_on=False)

# leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
#       bbox_to_anchor=(0.23,0.95),fancybox=True,ncol=1,frameon=False,
#       handlelength=1,handletextpad=0.7)
# for line,text in zip(leg.get_lines(), leg.get_texts()):
#     text.set_color(line.get_color())

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
plt.savefig(directoryfigure+'TimeSeries_All-Scenarios_GHG_hypothetical_v1-SameSlope.png',dpi=600)

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
    dataco2 = np.loadtxt(directoryoutput + 'co2_gblannualdata_ssp534os_from_wfc',skiprows=1)
    xx_co2 = dataco2[:,0]
    modelyears = dataco2[:,1]
    modelcopy = modelyears.copy()
    modelyears[1920:2101] = dataannual
    
    plt.figure()
    plt.plot(modelcopy)
    plt.plot(modelyears)
    
    plt.figure()
    plt.plot(dataannual)
    
    np.savetxt(directoryoutput + 'v1_SameSlope/co2_gblannualdata_ssp%sos_from_wfc' % startyr,
               np.c_[xx_co2,modelyears],fmt='%1.4f',header='2500',comments='')
    
    return dataall,dataannual,modelyears
    
all_2020,annual_2020,modelyrs_2020 = concFiles_co2(y_S2020,2020,co2_spear,yearstotal_extend)
all_2030,annual_2030,modelyrs_2030 = concFiles_co2(y_S2030,2030,co2_spear,yearstotal_extend)
all_2040,annual_2040,modelyrs_2040 = concFiles_co2(y_S2040,2040,co2_spear,yearstotal_extend)
all_2050,annual_2050,modelyrs_2050 = concFiles_co2(y_S2050,2050,co2_spear,yearstotal_extend)
all_2060,annual_2060,modelyrs_2060 = concFiles_co2(y_S2060,2060,co2_spear,yearstotal_extend)
all_2070,annual_2070,modelyrs_2070 = concFiles_co2(y_S2070,2070,co2_spear,yearstotal_extend)
all_2080,annual_2080,modelyrs_2080 = concFiles_co2(y_S2080,2080,co2_spear,yearstotal_extend)

def concFiles_ch4(data,startyr,spear,yearstotal_extend):
    ### Cat 1921 to 210
    dataall = np.append(spear[:-len(data)+12],data)
    
    ### Reshape
    datamonths = np.reshape(dataall,(len(yearstotal_extend),12))
    dataannual = np.nanmean(datamonths,axis=1)
    
    ### Read in raw data
    datach4 = np.loadtxt(directoryoutput + 'ch4_gblannualdata_ssp534os_from_wfc',skiprows=1)
    xx_ch4 = datach4[:,0]
    modelyears = datach4[:,1]
    modelcopy = modelyears.copy()
    modelyears[1920:2101] = dataannual
    
    plt.figure()
    plt.plot(modelcopy)
    plt.plot(modelyears)
    
    plt.figure()
    plt.plot(dataannual)
    
    np.savetxt(directoryoutput + 'v1_SameSlope/ch4_gblannualdata_ssp%sos_from_wfc' % startyr,
               np.c_[xx_ch4,modelyears],fmt='%1.4f',header='2500',comments='')
    
    return dataall,dataannual,modelyears

all_2020_ch4,annual_2020_ch4,modelyrs_2020_ch4 = concFiles_ch4(y_S2020_ch4,2020,ch4_spear,yearstotal_extend)
all_2030_ch4,annual_2030_ch4,modelyrs_2030_ch4 = concFiles_ch4(y_S2030_ch4,2030,ch4_spear,yearstotal_extend)
all_2040_ch4,annual_2040_ch4,modelyrs_2040_ch4 = concFiles_ch4(y_S2040_ch4,2040,ch4_spear,yearstotal_extend)
all_2050_ch4,annual_2050_ch4,modelyrs_2050_ch4 = concFiles_ch4(y_S2050_ch4,2050,ch4_spear,yearstotal_extend)
all_2060_ch4,annual_2060_ch4,modelyrs_2060_ch4 = concFiles_ch4(y_S2060_ch4,2060,ch4_spear,yearstotal_extend)
all_2070_ch4,annual_2070_ch4,modelyrs_2070_ch4 = concFiles_ch4(y_S2070_ch4,2070,ch4_spear,yearstotal_extend)
all_2080_ch4,annual_2080_ch4,modelyrs_2080_ch4 = concFiles_ch4(y_S2080_ch4,2080,ch4_spear,yearstotal_extend)
    
