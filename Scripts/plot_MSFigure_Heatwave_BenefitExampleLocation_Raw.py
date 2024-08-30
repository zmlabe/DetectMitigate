"""
Plot timeseries of a location to show how the net benefit is calculated

Author    : Zachary M. Labe
Date      : 26 February 2024
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import cmasher as cmr
import palettable.cubehelix as cm
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import sys
import scipy.stats as sts
import pandas as pd

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['TMAX']
variq = variablesall[0]
numOfEns = 30
numOfEns_10ye = 30
years = np.arange(2015,2100+1)

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/MSFigures_Heat/'
directorydata = '/home/Zachary.Labe/Research/DetectMitigate/Data/GridLocationExample/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['SPEAR_MED_Scenario','SPEAR_MED_Scenario']
experimentnames = ['SSP5-34OS','SSP5-34OS_10ye','SSP5-34OS minus SSP5-34OS_10ye']
seasons = ['JJA']
slicemonthnamen = ['JJA']
monthlychoice = seasons[0]
reg_name = 'Globe'
varcount = 'count90'
smoothingFactor = 10
YrThreshN = 10
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
### Get data
lat_bounds,lon_bounds = UT.regions(reg_name)

###############################################################################
###############################################################################
###############################################################################
### Read in data for OS daily extremes
raw = np.load(directorydata + 'gridpointExample_%s_none.npz' % varcount)
os_raw = raw['gridpoint_osX'][:]
os10ye_raw = raw['gridpoint_os_10yeX'][:]
whereBenefit_raw = raw['minwherebelowMax_X']
lats = raw['lats'][:]
lons = raw['lons'][:]
latqPoint_raw = raw['latqPoint']
lonqPoint_raw = raw['lonqPoint']
latloc = lats[latqPoint_raw]
lonloc = lons[lonqPoint_raw]

threshold = np.load(directorydata + 'gridpointExample_YrThreshold-%s_%s_none.npz' % (YrThreshN,varcount))
whereBenefit_threshold = threshold['minwherebelowMax_X']
lats = threshold['lats'][:]
lons = threshold['lons'][:]
latqPoint_threshold = threshold['latqPoint']
lonqPoint_threshold = threshold['lonqPoint']

running = np.load(directorydata + 'gridpointExample_running_%s_%s_none.npz' % (smoothingFactor,varcount))
os_running = running['gridpoint_osX'][:]
os10ye_running = running['gridpoint_os_10yeX'][:]
whereBenefit_running = running['minwherebelowMax_X']
latqPoint_running = running['latqPoint']
lonqPoint_running = running['lonqPoint']

filtering = np.load(directorydata + 'gridpointExample_savgolfilter_%s_%s_none.npz' % (smoothingFactor,varcount))
os_filtering = filtering['gridpoint_osX'][:]
os10ye_filtering = filtering['gridpoint_os_10yeX'][:]
whereBenefit_filtering = filtering['minwherebelowMax_X']
latqPoint_filtering = filtering['latqPoint']
lonqPoint_filtering = filtering['lonqPoint']

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

fig = plt.figure(figsize=(10,5))
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

plt.axvline(x=2040,color='teal',linewidth=1,linestyle=':',zorder=1)
plt.axvline(x=2031,color='maroon',linewidth=1,linestyle=':',zorder=2)
    
plt.plot(years,os_raw,linestyle='-',linewidth=3,color='teal',
          clip_on=False,zorder=3,label=r'\textbf{SSP543OS}')
plt.plot(years,os10ye_raw,linestyle='-',linewidth=3,color='maroon',
          clip_on=False,zorder=4,label=r'\textbf{SSP543OS_10ye}')
plt.plot(years,os_filtering,linestyle='--',linewidth=0.8,color='k',
          clip_on=False,zorder=5,dashes=(1,0.3))
plt.plot(years,os10ye_filtering,linestyle='--',linewidth=0.8,color='k',
          clip_on=False,zorder=6,dashes=(1,0.3))

# plt.scatter(years[np.argmax(os10ye_raw)],np.nanmax(os10ye_raw),s=27,color='r',
#             zorder=7,clip_on=False)
# plt.scatter(years[whereBenefit_raw],os_raw[whereBenefit_raw],s=27,color='r',
#             zorder=8,clip_on=False)
# plt.scatter(years[whereBenefit_threshold],os_raw[whereBenefit_threshold],s=30,edgecolor='r',color='None',
#             zorder=8,clip_on=False)
# plt.hlines(y=np.nanmax(os10ye_raw),xmin=years[np.argmax(os10ye_raw)],xmax=2100,
#           color='r',linestyle='--',clip_on=False,zorder=6)
# plt.hlines(y=np.nanmax(os10ye_raw),xmin=years[np.argmax(os10ye_raw)],xmax=years[whereBenefit_raw],
#           color='r',linestyle='-',clip_on=False,zorder=6)
# plt.fill_between(np.arange(years[np.argmax(os10ye_raw)],years[whereBenefit_raw]+1,1),0,40,
#                  alpha=0.3,color='r',edgecolor=None,zorder=1)
# plt.fill_between(np.arange(years[np.argmax(os10ye_raw)],years[whereBenefit_threshold]+1,1),0,40,
#                  alpha=0.2,color='r',edgecolor=None,zorder=1)

# plt.text(2046.5,40.5,r'\textbf{%s} (%s) \textbf{years}' % (years[whereBenefit_raw] - years[np.argmax(os10ye_raw)] - 10,years[whereBenefit_raw] - years[np.argmax(os10ye_raw)]),
#                                                                                         color='crimson',fontsize=12,
#                                                                                         alpha=0.5)
# plt.text(2070.35,40.5,r'\textbf{%s} (%s) \textbf{years}' % (years[whereBenefit_threshold] - years[np.argmax(os10ye_raw)] - 10, years[whereBenefit_threshold] - years[np.argmax(os10ye_raw)]),
#                                                 color='crimson',fontsize=12,
#                                                 alpha=0.3)
plt.text(2015,30,r'\textbf{[d]}',color='k',fontsize=12)
plt.text(2015,1.5,r'\textbf{%s$^{\circ}$N}' % (np.round(latloc,2)),
         color='dimgrey',fontsize=10,ha='left')
plt.text(2015,0,r'\textbf{%s$^{\circ}$W}' % (np.round(360-lonloc,2)),
         color='dimgrey',fontsize=10,ha='left')

# leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
#       bbox_to_anchor=(0.9,0.2),fancybox=True,ncol=1,frameon=False,
#       handlelength=1,handletextpad=0.5)
# for line,text in zip(leg.get_lines(), leg.get_texts()):
#     text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(0,41,5),2),np.round(np.arange(0,41,5),2))
plt.xlim([2015,2100])
plt.ylim([0,30])

plt.ylabel(r'\textbf{Count of heatwave days}',fontsize=10,color='k')

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

plt.axvline(x=2040,color='teal',linewidth=1,linestyle=':',zorder=1)
plt.axvline(x=2031,color='maroon',linewidth=1,linestyle=':',zorder=2)
    
plt.plot(years,os_filtering,linestyle='-',linewidth=3,color='teal',
          clip_on=False,zorder=3,label=r'\textbf{SSP5-3.4OS}')
plt.plot(years,os10ye_filtering,linestyle='-',linewidth=3,color='maroon',
          clip_on=False,zorder=4,label=r'\textbf{SSP5-3.4OS_10ye}')
plt.plot(years,os_raw,linestyle='--',linewidth=0.8,color='k',
          clip_on=False,zorder=5,dashes=(1,0.3))
plt.plot(years,os10ye_raw,linestyle='--',linewidth=0.8,color='k',
          clip_on=False,zorder=6,dashes=(1,0.3))

# plt.scatter(years[np.argmax(os10ye_filtering)],np.nanmax(os10ye_filtering),s=27,color='r',
#             zorder=7,clip_on=False)
# plt.scatter(years[whereBenefit_filtering],os_filtering[whereBenefit_filtering],s=27,color='r',
#             zorder=8,clip_on=False)
# plt.hlines(y=np.nanmax(os10ye_filtering),xmin=years[np.argmax(os10ye_filtering)],xmax=2100,
#           color='r',linestyle='--',clip_on=False,zorder=6)
# plt.hlines(y=np.nanmax(os10ye_filtering),xmin=years[np.argmax(os10ye_filtering)],xmax=years[whereBenefit_filtering],
#           color='r',linestyle='-',clip_on=False,zorder=6)
# plt.fill_between(np.arange(years[np.argmax(os10ye_filtering)],years[whereBenefit_filtering]+1,1),0,40,
#                  alpha=0.2,color='r',edgecolor=None,zorder=1)

plt.text(2015,30,r'\textbf{[e]}',color='k',fontsize=12)
# plt.text(2057,40.5,r'\textbf{%s} (%s) \textbf{years}' % (years[whereBenefit_filtering] - years[np.argmax(os10ye_filtering)] - 10, years[whereBenefit_filtering] - years[np.argmax(os10ye_filtering)]),
#                                                 color='crimson',fontsize=12,
#                                                 alpha=0.3)
plt.text(2015,1.5,r'\textbf{%s$^{\circ}$N}' % (np.round(latloc,2)),
         color='dimgrey',fontsize=10,ha='left')
plt.text(2015,0,r'\textbf{%s$^{\circ}$W}' % (np.round(360-lonloc,2)),
         color='dimgrey',fontsize=10,ha='left')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
leg = plt.legend(shadow=False,fontsize=12,loc='upper center',
      bbox_to_anchor=(0,-0.05),fancybox=True,ncol=3,frameon=False,
      handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1920,2101,10),np.arange(1920,2101,10))
plt.yticks(np.round(np.arange(0,41,5),2),np.round(np.arange(0,41,5),2))
plt.xlim([2015,2100])
plt.ylim([0,30])

# plt.ylabel(r'\textbf{Count of heatwave days}',fontsize=10,color='k')

plt.savefig(directoryfigure + 'MSFigure_Heatwave_BenefitTimeSeries_%s_ExampleLoc_%s-%s_%s_none.png' % (variq,latloc,lonloc,monthlychoice),dpi=300)
