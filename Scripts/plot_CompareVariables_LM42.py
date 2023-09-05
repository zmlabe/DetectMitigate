"""
Evaluate differences with SPEAR and SPEAR LM4.2

Author    : Zachary M. Labe
Date      : 29 August 2023
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
import read_SPEAR_MED as SP
import read_SPEAR_MED_LM42p2_test as LL
from scipy.interpolate import griddata as g

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['PRECT','EVAP','SHFLX','WA','ALBEDO','SNOW','TS','T2M']
variablesall = ['T2M','TS','ALBEDO','SNOW']
monthlychoiceall = ['JFM','AMJ','JJA','JAS','OND']

for v in range(len(variablesall)):
    for m in range(len(monthlychoiceall)):
        monthlychoice = monthlychoiceall[m]
        variq = variablesall[v]
        numOfEns = 30
        numOfEns_LM42 = 3
        years = np.arange(1921,2100+1)
        years_LM42 = np.arange(1921,2070+1,1)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directoryfigure = '/home/Zachary.Labe/Research/DetectMitigate/Figures/'
        letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
        
        ### Temporary regridding function
        def regrid(lat11,lon11,lat21,lon21,var,years):
            """
            Interpolated on selected grid. Reads SPEAR in as 4d with 
            [year,lat,lon]
            """
            
            lon1,lat1 = np.meshgrid(lon11,lat11)
            lon2,lat2 = np.meshgrid(lon21,lat21)
            
            varn_re = np.reshape(var,(var.shape[0],(lat1.shape[0]*lon1.shape[1])))   
            varn = np.empty((var.shape[0],lat2.shape[0],lon2.shape[1]))
            
            print('Completed: Start regridding process:')
            for i in range(years.shape[0]):
                z = g((np.ravel(lat1),np.ravel(lon1)),varn_re[i,:],(lat2,lon2),method='linear')
                varn[i,:,:] = z
                print('Completed: Year %s Regridding---' % (years[i]))
            return varn
        
        ### Read in data
        lat,lon,lai_s_all = SP.read_SPEAR_MED('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/',variq,monthlychoice,4,np.nan,30,'all')
        latl,lonl,lai_l = LL.read_SPEAR_MED_LM42p2_test('/work/Zachary.Labe/Data/',variq,monthlychoice,4,np.nan,3,'all')
        
        ### Meshgrid
        lon2,lat2 = np.meshgrid(lon,lat)
        
        ### Only plot data to 2070
        lai_s = lai_s_all[:,:len(years_LM42),:,:]
        
        ### Mask oceans
        lai_s[np.where(lai_s == -1)] = np.nan
        lai_l[np.where(lai_l == -1)] = np.nan
        
        ### Calculate ensemble mean
        lai_ms = np.nanmean(lai_s[:,:,:,:],axis=0)
        lai_ml = np.nanmean(lai_l[:,:,:,:],axis=0)
        
        # ### Regrid 
        if variq == 'SNOW':
            lai_mlr = regrid(latl,lonl,lat,lon,lai_ml,years_LM42)
        else:
            lai_mlr = lai_ml
        
        ### Calculate differences
        diff = lai_mlr - lai_ms
        
        ### Calculate epochs
        epoch_early = np.nanmean(diff[:30,:,:],axis=0)
        epoch_late = np.nanmean(diff[-30:,:,:],axis=0)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Plot figure
        lon2,lat2 = np.meshgrid(lon,lat)
        
        fig = plt.figure(figsize=(8,4))
        for txt in fig.texts:
            txt.set_visible(False)
        
        ### Colorbar limits
        if variq == 'PRECT':
            barlim = np.arange(-1,2,1)
            limit = np.arange(-1,1.01,0.01)
            label = r'\textbf{PRECIP Difference [mm/day]}' 
        elif variq == 'EVAP':
            barlim = np.arange(-1,2,1)
            limit = np.arange(-1,1.01,0.01)
            label = r'\textbf{EVAP Difference [mm/day]}' 
        elif variq == 'WA':
            barlim = np.arange(-1,2,1)
            limit = np.arange(-1,1.01,0.01)
            label = r'\textbf{WA Difference [mm/day]}' 
        elif variq == 'TS':
            barlim = np.arange(-5,6,1)
            limit = np.arange(-5,5.01,0.01)
            label = r'\textbf{TS Difference [$^{\circ}$C]}' 
        elif variq == 'T2M':
            barlim = np.arange(-5,6,1)
            limit = np.arange(-5,5.01,0.01)
            label = r'\textbf{T2M Difference [$^{\circ}$C]}' 
        elif variq == 'SHFLX':
            barlim = np.arange(-30,31,15)
            limit = np.arange(-30,30.1,0.1)
            label = r'\textbf{SHFLX Difference [W/m2]}' 
        elif variq == 'ALBEDO':
            barlim = np.arange(-10,11,5)
            limit = np.arange(-10,10.1,0.1)
            label = r'\textbf{ALBEDO Difference [\%]}' 
        elif variq == 'SNOW':
            barlim = np.round(np.arange(-0.05,0.051,0.05),2)
            limit = np.arange(-0.05,0.051,0.001)
            label = r'\textbf{SNOW Difference [m]}' 
        
        ax = plt.subplot(121)
            
        ### Select map type
        m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                    projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
                    area_thresh=5000)
        
        circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        m.drawcoastlines(color='darkgrey',linewidth=1)
        m.drawstates(color='darkgrey',linewidth=0.5)
        m.drawcountries(color='darkgrey',linewidth=0.5)
        
        ### Make the plot continuous
        cs = m.contourf(lon2,lat2,epoch_early,limit,
                          extend='both',latlon=True)
                        
        cmap = cmocean.cm.balance
        cs.set_cmap(cmap)
        
        plt.title(r'\textbf{(a) LM4.2 -- LM4; %s; 1921-1950 - %s}' % (variq,monthlychoice),fontsize=11,color='dimgrey')
        
        ###############################################################################
        ax = plt.subplot(122)
            
        ### Select map type
        m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                    projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='h',
                    area_thresh=5000)
        
        circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        m.drawcoastlines(color='darkgrey',linewidth=1)
        m.drawstates(color='darkgrey',linewidth=0.5)
        m.drawcountries(color='darkgrey',linewidth=0.5)
        
        ### Make the plot continuous
        cs = m.contourf(lon2,lat2,epoch_late,limit,
                          extend='both',latlon=True)
                      
        cmap = cmocean.cm.balance
        cs.set_cmap(cmap)
        
        plt.title(r'\textbf{(b) LM4.2 -- LM4; %s; 2041-2070 - %s}' % (variq,monthlychoice),fontsize=11,color='dimgrey')
        
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.355,0.1,0.3,0.03])                
        cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        plt.savefig(directoryfigure + 'Comparison%s_LM42_US_%s.png' % (variq,monthlychoice),dpi=300)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Plot figure
        
        fig = plt.figure(figsize=(8,3))
        for txt in fig.texts:
            txt.set_visible(False)
        
        ### Colorbar limits
        if variq == 'PRECT':
            barlim = np.arange(-1,2,1)
            limit = np.arange(-1,1.01,0.01)
            label = r'\textbf{PRECIP Difference [mm/day]}' 
        elif variq == 'EVAP':
            barlim = np.arange(-1,2,1)
            limit = np.arange(-1,1.01,0.01)
            label = r'\textbf{EVAP Difference [mm/day]}' 
        elif variq == 'WA':
            barlim = np.arange(-1,2,1)
            limit = np.arange(-1,1.01,0.01)
            label = r'\textbf{WA Difference [mm/day]}' 
        elif variq == 'TS':
            barlim = np.arange(-5,6,1)
            limit = np.arange(-5,5.01,0.01)
            label = r'\textbf{TS Difference [$^{\circ}$C]}' 
        elif variq == 'T2M':
            barlim = np.arange(-5,6,1)
            limit = np.arange(-5,5.01,0.01)
            label = r'\textbf{T2M Difference [$^{\circ}$C]}' 
        elif variq == 'SHFLX':
            barlim = np.arange(-30,31,15)
            limit = np.arange(-30,30.1,0.1)
            label = r'\textbf{SHFLX Difference [W/m2]}' 
        elif variq == 'ALBEDO':
            barlim = np.arange(-10,11,5)
            limit = np.arange(-10,10.1,0.1)
            label = r'\textbf{ALBEDO Difference [\%]}' 
        elif variq == 'SNOW':
            barlim = np.round(np.arange(-0.05,0.051,0.05),2)
            limit = np.arange(-0.05,0.051,0.001)
            label = r'\textbf{SNOW Difference [m]}' 
        
        ax = plt.subplot(121)
            
        ### Select map type
        m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
        
        circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        m.drawcoastlines(color='darkgrey',linewidth=0.3)
        
        ### Make the plot continuous
        cs = m.contourf(lon2,lat2,epoch_early,limit,
                          extend='both',latlon=True)
                        
        cmap = cmocean.cm.balance
        cs.set_cmap(cmap)
        
        plt.title(r'\textbf{(a) LM4.2 -- LM4; %s; 1921-1950 - %s}' % (variq,monthlychoice),fontsize=11,color='dimgrey')
        
        ###############################################################################
        ax = plt.subplot(122)
            
        ### Select map type
        m = Basemap(projection='robin',lon_0=0,resolution='h',area_thresh=10000)
        
        circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        m.drawcoastlines(color='darkgrey',linewidth=0.3)
        
        ### Make the plot continuous
        cs = m.contourf(lon2,lat2,epoch_late,limit,
                          extend='both',latlon=True)
                      
        cmap = cmocean.cm.balance
        cs.set_cmap(cmap)
        
        plt.title(r'\textbf{(b) LM4.2 -- LM4; %s; 2041-2070 - %s}' % (variq,monthlychoice),fontsize=11,color='dimgrey')
        
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.355,0.08,0.3,0.03])                
        cbar1 = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.13)
        
        plt.savefig(directoryfigure + 'Comparison%s_LM42_Globe_%s.png' % (variq,monthlychoice),dpi=300)
