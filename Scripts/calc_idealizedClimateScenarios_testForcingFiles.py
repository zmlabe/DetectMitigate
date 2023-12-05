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
directoryoutput = '/home/Zachary.Labe/Research/DetectMitigate/Data/v1_2040Slopes/'
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
### Read in data for my OS
data_slow = np.loadtxt(directoryoutput + 'co2_gblannualdata_ssp2040slowos_from_wfc',skiprows=1)
time_slow_co2 = data_slow[:,0]
co2_slow = data_slow[:,1]
data_slow = np.loadtxt(directoryoutput + 'ch4_gblannualdata_ssp2040slowos_from_wfc',skiprows=1)
time_slow_ch4 = data_slow[:,0]
ch4_slow = data_slow[:,1]

data_linear = np.loadtxt(directoryoutput + 'co2_gblannualdata_ssp2040linearos_from_wfc',skiprows=1)
time_linear_co2 = data_linear[:,0]
co2_linear = data_linear[:,1]
data_linear = np.loadtxt(directoryoutput + 'ch4_gblannualdata_ssp2040linearos_from_wfc',skiprows=1)
time_linear_ch4 = data_linear[:,0]
ch4_linear = data_linear[:,1]

data_fast = np.loadtxt(directoryoutput + 'co2_gblannualdata_ssp2040fastos_from_wfc',skiprows=1)
time_fast_co2 = data_fast[:,0]
co2_fast = data_fast[:,1]
data_fast = np.loadtxt(directoryoutput + 'ch4_gblannualdata_ssp2040fastos_from_wfc',skiprows=1)
time_fast_ch4 = data_fast[:,0]
ch4_fast = data_fast[:,1]

data_faster = np.loadtxt(directoryoutput + 'co2_gblannualdata_ssp2040fasteros_from_wfc',skiprows=1)
time_faster_co2 = data_faster[:,0]
co2_faster = data_faster[:,1]
data_faster = np.loadtxt(directoryoutput + 'ch4_gblannualdata_ssp2040fasteros_from_wfc',skiprows=1)
time_faster_ch4 = data_faster[:,0]
ch4_faster = data_faster[:,1]

###############################################################################
###############################################################################
###############################################################################
### Read in data for previous OS
tryyears = np.arange(1850,2100+1,1)
dataco2 = np.loadtxt('/home/Zachary.Labe/Research/DetectMitigate/Data/co2_gblannualdata_ssp534os_from_wfc',skiprows=1)
time_oldco2 = dataco2[:,0]
data_oldco2 = dataco2[:,1]
datach4 = np.loadtxt('/home/Zachary.Labe/Research/DetectMitigate/Data/ch4_gblannualdata_ssp534os_from_wfc',skiprows=1)
time_oldch4 = datach4[:,0]
data_oldch4 = datach4[:,1]

###############################################################################
###############################################################################
###############################################################################
### Plotting test figures
plt.figure()
plt.plot(co2_slow,color='gold')
plt.plot(co2_linear,color='r')
plt.plot(co2_fast,color='b')
plt.plot(co2_faster,color='g')

plt.figure()
plt.plot(data_oldco2,color='darkorange')

plt.figure()
plt.plot(ch4_slow,color='gold')
plt.plot(ch4_linear,color='r')
plt.plot(ch4_fast,color='b')
plt.plot(ch4_faster,color='g')

plt.figure()
plt.plot(data_oldch4,color='darkorange')

###############################################################################
###############################################################################
###############################################################################
### Plotting test figures - zoom
plt.figure()
plt.plot(co2_slow[1920:2101],color='gold')
plt.plot(co2_linear[1920:2101],color='r')
plt.plot(co2_fast[1920:2101],color='b')
plt.plot(co2_faster[1920:2101],color='g')

plt.figure()
plt.plot(data_oldco2[1920:2101],color='darkorange')

plt.figure()
plt.plot(ch4_slow[1920:2101],color='gold')
plt.plot(ch4_linear[1920:2101],color='r')
plt.plot(ch4_fast[1920:2101],color='b')
plt.plot(ch4_faster[1920:2101],color='g')

plt.figure()
plt.plot(data_oldch4[1920:2101],color='darkorange')

###############################################################################
###############################################################################
###############################################################################
### Plotting test figures - zoom x2
plt.figure()
plt.plot(co2_slow[2014:2101],color='gold')
plt.plot(co2_linear[2014:2101],color='r')
plt.plot(co2_fast[2014:2101],color='b')
plt.plot(co2_faster[2014:2101],color='g')

plt.figure()
plt.plot(data_oldco2[2014:2101],color='darkorange')

plt.figure()
plt.plot(ch4_slow[2014:2101],color='gold')
plt.plot(ch4_linear[2014:2101],color='r')
plt.plot(ch4_fast[2014:2101],color='b')
plt.plot(ch4_faster[2014:2101],color='g')

plt.figure()
plt.plot(data_oldch4[2014:2101],color='darkorange')

###############################################################################
###############################################################################
###############################################################################
### Plotting test figures - zoom x3
plt.figure()
plt.plot(co2_slow[2039:2101],color='gold')
plt.plot(co2_linear[2039:2101],color='r')
plt.plot(co2_fast[2039:2101],color='b')
plt.plot(co2_faster[2039:2101],color='g')

plt.figure()
plt.plot(data_oldco2[2039:2101],color='darkorange')

plt.figure()
plt.plot(ch4_slow[2039:2101],color='gold')
plt.plot(ch4_linear[2039:2101],color='r')
plt.plot(ch4_fast[2039:2101],color='b')
plt.plot(ch4_faster[2039:2101],color='g')

plt.figure()
plt.plot(data_oldch4[2039:2101],color='darkorange')

###############################################################################
###############################################################################
###############################################################################
### Plotting test figures - zoom x4
plt.figure()
plt.plot(co2_slow[2039:2109],color='gold')
plt.plot(co2_linear[2039:2109],color='r')
plt.plot(co2_fast[2039:2109],color='b')
plt.plot(co2_faster[2039:2109],color='g')

plt.figure()
plt.plot(data_oldco2[2039:2109],color='darkorange')

plt.figure()
plt.plot(ch4_slow[2039:2109],color='gold')
plt.plot(ch4_linear[2039:2109],color='r')
plt.plot(ch4_fast[2039:2109],color='b')
plt.plot(ch4_faster[2039:2109],color='g')

plt.figure()
plt.plot(data_oldch4[2039:2109],color='darkorange')
