"""
Plot histogram of trends over different time periods compared to SMILEs

Author    : Zachary M. Labe
Date      : 23 May 2022
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import calc_Utilities as UT
import sys
import read_ERA5_monthly as ERA
import read_SMILE_historical as SM
import read_LENS_historical as LE
import read_SPEAR_MED as SP
import read_SPEAR_MED_NOAER as AER
import read_SPEAR_MED_NATURAL as NAT
import scipy.stats as sts

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/Attribution_SpringNA/Figures/' 
directorydata = '/work/Zachary.Labe/Data/' 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
variq = 'T2M'
sliceperiod = 'FMA'
years = np.arange(1979,2022+1,1)
yearsnoaer = np.arange(1979,2020+1,1)
yearsnat = np.arange(1979,2022+1,1)
slicenan = 'nan'
addclimo = True
newon = True
datareader = True

### Read data
if datareader == True:
    lat1,lon1,obs = ERA.read_ERA5_monthly(variq,'/work/Zachary.Labe/Data/ERA5/',sliceperiod,
                                    years,3,addclimo,
                                    slicenan,newon)
    lat1,lon1,gfdlc = SM.read_SMILEhistorical('/work/Zachary.Labe/Data/SMILE/','GFDL_CM3',variq,
                                                sliceperiod,4,
                                                slicenan,20)
    lat1,lon1,gfdlem = SM.read_SMILEhistorical('/work/Zachary.Labe/Data/SMILE/','GFDL_ESM2M',variq,
                                                sliceperiod,4,
                                                slicenan,20)
    lat1,lon1,lens = LE.read_LENShistorical('/work/Zachary.Labe/Data/LENS/monthly/',variq,
                                            sliceperiod,4,
                                            slicenan,40)
    lat1s,lon1s,spear = SP.read_SPEAR_MED('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/',variq,
                                            sliceperiod,4,
                                            slicenan,30,'all')
    lat1sa,lon1sa,spearnoaer = AER.read_SPEAR_MED_NOAER('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_NOAER/monthly/',variq,
                                            sliceperiod,4,
                                            slicenan,12,'all')
    lat1snat,lon1snat,spearnat = NAT.read_SPEAR_MED_NATURAL('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_NATURAL/monthly/',variq,
                                            sliceperiod,4,
                                            slicenan,30,'satellite')
  
def calc_anomalies(years,data):
    """ 
    Calculate anomalies
    """
    
    ### Baseline - 1981-2010
    if data.ndim == 3:
        yearqold = np.where((years >= 1981) & (years <= 2010))[0]
        climold = np.nanmean(data[yearqold,:,:],axis=0)
        anoms = data - climold
    elif data.ndim == 4:
        yearqold = np.where((years >= 1981) & (years <= 2010))[0]
        climold = np.nanmean(data[:,yearqold,:,:],axis=1)
        anoms = data - climold[:,np.newaxis,:,:]
    
    return anoms

def calc_regionalAve(regionn,lat1,lon1,data):
    if regionn == 'CANMID':
        # la1 = 43
        # la2 = 63
        # lo1 = 238
        # lo2 = 270
        la1 = 43
        la2 = 60
        lo1 = 240
        lo2 = 295
        lat1q = np.where((lat1 >= la1) & (lat1 <= la2))[0]
        lon1q = np.where((lon1 >= lo1) & (lon1 <= lo2))[0]
        
    if data.ndim == 3:
        meanlat = data[:,lat1q,:]
        meanbox = meanlat[:,:,lon1q]
        lon1a = lon1[lon1q]
        lat1a = lat1[lat1q]
        lon2q,lat2q = np.meshgrid(lon1a,lat1a)
    elif data.ndim == 4:
        meanlat = data[:,:,lat1q,:]
        meanbox = meanlat[:,:,:,lon1q]
        lon1a = lon1[lon1q]
        lat1a = lat1[lat1q]
        lon2q,lat2q = np.meshgrid(lon1a,lat1a)
        
    ### Calculate timeseries
    mean = UT.calc_weightedAve(meanbox,lat2q)

    return mean,lat1a,lat2q,lon1a,lon2q

def calcTrends(years,yrmin,yrmax,data):
    yearq = np.where((years >= yrmin) & (years <= yrmax))[0]
    yearTrend = years[yearq]
    print(yearTrend)
    
    if data.ndim == 1:
        dataTrend = data[yearq]
        slope,intercept,r,p,se = sts.linregress(yearTrend,dataTrend)
        trendline = slope*yearTrend + intercept
    elif data.ndim == 2:
        dataTrend = data[:,yearq]
        
        slope = np.empty((data.shape[0]))
        intercept = np.empty((data.shape[0]))
        r = np.empty((data.shape[0]))
        p = np.empty((data.shape[0]))
        se = np.empty((data.shape[0]))
        trendline = np.empty((data.shape[0],len(yearTrend)))
        for i in range(data.shape[0]):
            slopeq,interceptq,rq,pq,seq = sts.linregress(yearTrend,dataTrend[i,:])
            slope[i] = slopeq
            intercept[i] = interceptq
            r[i] = rq
            p[i] = pq
            se[i] = seq
            trendline[i,:] = slopeq*yearTrend + interceptq
    else:
        print(ValueError('WRONG DIMENSIONS OF TREND INPUT!!!'))
        
    return slope,intercept,r,p,se,trendline,yearTrend

obs_anom = calc_anomalies(years,obs)
gfdlc_anom = calc_anomalies(years,gfdlc)
gfdlem_anom = calc_anomalies(years,gfdlem)
lens_anom = calc_anomalies(years,lens)
spear_anom = calc_anomalies(years,spear)
spearnoaer_anom = calc_anomalies(yearsnoaer,spearnoaer)
spearnat_anom = calc_anomalies(yearsnat,spearnat)

obs_reg,latr1,latr2,lonr1,lonr2 = calc_regionalAve('CANMID',lat1,lon1,obs_anom)
gfdlc_reg,latr1,latr2,lonr1,lonr2 = calc_regionalAve('CANMID',lat1,lon1,gfdlc_anom)
gfdlem_reg,latr1,latr2,lonr1,lonr2 = calc_regionalAve('CANMID',lat1,lon1,gfdlem_anom)
lens_reg,latr1,latr2,lonr1,lonr2 = calc_regionalAve('CANMID',lat1,lon1,lens_anom)
spear_reg,latrs1,latrs2,lonrs1,lonrs2 = calc_regionalAve('CANMID',lat1s,lon1s,spear_anom)
spearnoaer_reg,latrsa1,latrsa2,lonrsa1,lonrsa2 = calc_regionalAve('CANMID',lat1sa,lon1sa,spearnoaer_anom)
spearnat_reg,latrsnat1,latrsnat2,lonrsnat1,lonrsnat2 = calc_regionalAve('CANMID',lat1snat,lon1snat,spearnat_anom)

yrmin1 = 1979
yrmax1 = 2014
slope_obs1,intercept_obs1,r_obs1,p_obs1,se_obs1,trendline_obs1,yearTrend_obs1 = calcTrends(years,yrmin1,yrmax1,obs_reg)
slope_gfdlc1,intercept_gfdlc1,r_gfdlc1,p_gfdlc1,se_gfdlc1,trendline_gfdlc1,yearTrend_gfdlc1 = calcTrends(years,yrmin1,yrmax1,gfdlc_reg)
slope_gfdlem1,intercept_gfdlem1,r_gfdlem1,p_gfdlem1,se_gfdlem1,trendline_gfdlem1,yearTrend_gfdlem1 = calcTrends(years,yrmin1,yrmax1,gfdlem_reg)
slope_lens1,intercept_lens1,r_lens1,p_lens1,se_lens1,trendline_lens1,yearTrend_lens1 = calcTrends(years,yrmin1,yrmax1,lens_reg)
slope_spear1,intercept_spear1,r_spear1,p_spear1,se_spear1,trendline_spear1,yearTrend_spear1 = calcTrends(years,yrmin1,yrmax1,spear_reg)
slope_spearnoaer1,intercept_spearnoaer1,r_spearnoaer1,p_spearnoaer1,se_spearnoaer1,trendline_spearnoaer1,yearTrend_spearnoaer1 = calcTrends(yearsnoaer,yrmin1,yrmax1,spearnoaer_reg)
slope_spearnat1,intercept_spearnat1,r_spearnat1,p_spearnat,se_spearnat1,trendline_spearnat1,yearTrend_spearnat1 = calcTrends(yearsnat,yrmin1,yrmax1,spearnat_reg)

yrmin2 = 2000
yrmax2 = 2014
slope_obs2,intercept_obs2,r_obs2,p_obs2,se_obs2,trendline_obs2,yearTrend_obs2 = calcTrends(years,yrmin2,yrmax2,obs_reg)
slope_gfdlc2,intercept_gfdlc2,r_gfdlc2,p_gfdlc2,se_gfdlc2,trendline_gfdlc2,yearTrend_gfdlc2 = calcTrends(years,yrmin2,yrmax2,gfdlc_reg)
slope_gfdlem2,intercept_gfdlem2,r_gfdlem2,p_gfdlem2,se_gfdlem2,trendline_gfdlem2,yearTrend_gfdlem2 = calcTrends(years,yrmin2,yrmax2,gfdlem_reg)
slope_lens2,intercept_lens2,r_lens2,p_lens2,se_lens2,trendline_lens2,yearTrend_lens2 = calcTrends(years,yrmin2,yrmax2,lens_reg)
slope_spear2,intercept_spear2,r_spear2,p_spear2,se_spear2,trendline_spear2,yearTrend_spear2 = calcTrends(years,yrmin2,yrmax2,spear_reg)
slope_spearnoaer2,intercept_spearnoaer2,r_spearnoaer2,p_spearnoaer2,se_spearnoaer2,trendline_spearnoaer2,yearTrend_spearnoaer2 = calcTrends(yearsnoaer,yrmin2,yrmax2,spearnoaer_reg)
slope_spearnat2,intercept_spearnat2,r_spearnat2,p_spearnat2,se_spearnat2,trendline_spearnat2,yearTrend_spearnat2 = calcTrends(yearsnat,yrmin2,yrmax2,spearnat_reg)

yrmin3 = 2012
yrmax3 = 2014
slope_obs3,intercept_obs3,r_obs3,p_obs3,se_obs3,trendline_obs3,yearTrend_obs3 = calcTrends(years,yrmin3,yrmax3,obs_reg)
slope_gfdlc3,intercept_gfdlc3,r_gfdlc3,p_gfdlc3,se_gfdlc3,trendline_gfdlc3,yearTrend_gfdlc3 = calcTrends(years,yrmin3,yrmax3,gfdlc_reg)
slope_gfdlem3,intercept_gfdlem3,r_gfdlem3,p_gfdlem3,se_gfdlem3,trendline_gfdlem3,yearTrend_gfdlem3 = calcTrends(years,yrmin3,yrmax3,gfdlem_reg)
slope_lens3,intercept_lens3,r_lens3,p_lens3,se_lens3,trendline_lens3,yearTrend_lens3 = calcTrends(years,yrmin3,yrmax3,lens_reg)
slope_spear3,intercept_spear3,r_spear3,p_spear3,se_spear3,trendline_spear3,yearTrend_spear3 = calcTrends(years,yrmin3,yrmax3,spear_reg)
slope_spearnoaer3,intercept_spearnoaer3,r_spearnoaer3,p_spearnoaer3,se_spearnoaer3,trendline_spearnoaer3,yearTrend_spearnoaer3 = calcTrends(yearsnoaer,yrmin3,yrmax3,spearnoaer_reg)
slope_spearnat3,intercept_spearnat3,r_spearnat3,p_spearnat3,se_spearnoaer3,trendline_spearnat3,yearTrend_spearnat3 = calcTrends(yearsnat,yrmin3,yrmax3,spearnat_reg)

directoryoutput = '/home/Zachary.Labe/Research/Attribution_SpringNA/Data/'
np.savetxt(directoryoutput + 'Slopes_%s_obs_%s-%s.txt' % (sliceperiod,yrmin1,yrmax1),[slope_obs1])
np.savetxt(directoryoutput + 'Slopes_%s_obs_%s-%s.txt' % (sliceperiod,yrmin2,yrmax2),[slope_obs2])
np.savetxt(directoryoutput + 'Slopes_%s_obs_%s-%s.txt' % (sliceperiod,yrmin3,yrmax3),[slope_obs3])

np.savetxt(directoryoutput + 'Slopes_%s_gfdlc_%s-%s.txt' % (sliceperiod,yrmin1,yrmax1),slope_gfdlc1)
np.savetxt(directoryoutput + 'Slopes_%s_gfdlc_%s-%s.txt' % (sliceperiod,yrmin2,yrmax2),slope_gfdlc2)
np.savetxt(directoryoutput + 'Slopes_%s_gfdlc_%s-%s.txt' % (sliceperiod,yrmin3,yrmax3),slope_gfdlc3)

np.savetxt(directoryoutput + 'Slopes_%s_gfdlem_%s-%s.txt' % (sliceperiod,yrmin1,yrmax1),slope_gfdlem1)
np.savetxt(directoryoutput + 'Slopes_%s_gfdlem_%s-%s.txt' % (sliceperiod,yrmin2,yrmax2),slope_gfdlem2)
np.savetxt(directoryoutput + 'Slopes_%s_gfdlem_%s-%s.txt' % (sliceperiod,yrmin3,yrmax3),slope_gfdlem3)

np.savetxt(directoryoutput + 'Slopes_%s_lens_%s-%s.txt' % (sliceperiod,yrmin1,yrmax1),slope_lens1)
np.savetxt(directoryoutput + 'Slopes_%s_lens_%s-%s.txt' % (sliceperiod,yrmin2,yrmax2),slope_lens2)
np.savetxt(directoryoutput + 'Slopes_%s_lens_%s-%s.txt' % (sliceperiod,yrmin3,yrmax3),slope_lens3)

np.savetxt(directoryoutput + 'Slopes_%s_spear_%s-%s.txt' % (sliceperiod,yrmin1,yrmax1),slope_spear1)
np.savetxt(directoryoutput + 'Slopes_%s_spear_%s-%s.txt' % (sliceperiod,yrmin2,yrmax2),slope_spear2)
np.savetxt(directoryoutput + 'Slopes_%s_spear_%s-%s.txt' % (sliceperiod,yrmin3,yrmax3),slope_spear3)

np.savetxt(directoryoutput + 'Slopes_%s_noaerspear_%s-%s.txt' % (sliceperiod,yrmin1,yrmax1),slope_spearnoaer1)
np.savetxt(directoryoutput + 'Slopes_%s_noaerspear_%s-%s.txt' % (sliceperiod,yrmin2,yrmax2),slope_spearnoaer2)
np.savetxt(directoryoutput + 'Slopes_%s_noaerspear_%s-%s.txt' % (sliceperiod,yrmin3,yrmax3),slope_spearnoaer3)

np.savetxt(directoryoutput + 'Slopes_%s_natural_%s-%s.txt' % (sliceperiod,yrmin1,yrmax1),slope_spearnat1)
np.savetxt(directoryoutput + 'Slopes_%s_natural_%s-%s.txt' % (sliceperiod,yrmin2,yrmax2),slope_spearnat2)
np.savetxt(directoryoutput + 'Slopes_%s_natural_%s-%s.txt' % (sliceperiod,yrmin3,yrmax3),slope_spearnat3)

