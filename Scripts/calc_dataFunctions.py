"""
Functions are useful untilities for data processing in the NN
 
Notes
-----
    Author : Zachary Labe
    Date   : 7 September 2022
    
Usage
-----
    [1] readFiles(variq,dataset,monthlychoice)
    [2] getRegion(data,lat1,lon1,lat_bounds,lon_bounds)
"""

def readFiles(variq,dataset,monthlychoice,scenario):
    """
    Function reads in data for selected dataset

    Parameters
    ----------
    variq : string
        variable for analysis
    dataset : string
        name of data set for primary data
    monthlychoice : string
        time period of analysis
    scenario : string
         SSP119 or SSP245 or SSP370 or SSP585 or SSP534OS
        
    Returns
    -------
    data : numpy array
        data from selected data set
    lat1 : 1d numpy array
        latitudes
    lon1 : 1d numpy array
        longitudes

    Usage
    -----
    data,lat1,lon1 = readFiles(variq,dataset,monthlychoice)
    """
    print('\n>>>>>>>>>> Using readFiles function!')
    
    ### Import modules
    import numpy as np
    import sys
    
    if dataset == '20CRv3':
        import read_20CRv3_monthly as CR
        directorydataCR = '/work/Zachary.Labe/Data/20CRv3/'
        sliceyearCR = np.arange(1836,2015+1,1)
        if monthlychoice == 'none':
            sliceshapeCR = 4
        else:
            sliceshapeCR = 3
        slicenanCR = 'nan'
        lat1,lon1,data = CR.read_20CRv3_monthly(variq,directorydataCR,monthlychoice,
                                                sliceyearCR,sliceshapeCR,slicenanCR)
    elif dataset == '20CRv3_LOWS':
        import read_20CRv3_monthlyLOWS as CRL
        directorydataCRL = '/work/Zachary.Labe/Data/20CRv3_LOWS/'
        sliceyearCRL = np.arange(1836,2015+1,1)
        if monthlychoice == 'none':
            sliceshapeCRL = 4
        else:
            sliceshapeCRL = 3
        slicenanCRL = 'nan'
        lat1,lon1,data = CRL.read_20CRv3_monthlyLOWS(variq,directorydataCRL,monthlychoice,
                                                     sliceyearCRL,sliceshapeCRL,slicenanCRL)
    elif dataset == 'NClimGrid':
        import read_NClimGrid_monthly as NC
        directorydataNC = '/work/Zachary.Labe/Data/'
        sliceyearNC = np.arange(1895,2021+1,1)
        if monthlychoice == 'none':
            sliceshapeNC = 4
        else:
            sliceshapeNC = 3
        slicenanNC = 'nan'
        lat1,lon1,data = NC.read_NClimGrid_monthly(variq,directorydataNC,monthlychoice,
                                                   sliceyearNC,sliceshapeNC,slicenanNC)
    elif dataset == 'NClimGrid_LOWS':
        import read_NClimGrid_monthlyLOWS as NCL
        directorydataNCL = '/work/Zachary.Labe/Data/'
        sliceyearNCL = np.arange(1895,2021+1,1)
        if monthlychoice == 'none':
            sliceshapeNCL = 4
        else:
            sliceshapeNCL = 3
        slicenanNCL = 'nan'
        lat1,lon1,data = NCL.read_NClimGrid_monthlyLOWS(variq,directorydataNCL,monthlychoice,
                                                   sliceyearNCL,sliceshapeNCL,slicenanNCL)
    elif dataset == 'NClimGrid_MEDS':
        import read_NClimGrid_monthlyMEDS as NCM
        directorydataNCM = '/work/Zachary.Labe/Data/'
        sliceyearNCM = np.arange(1895,2021+1,1)
        if monthlychoice == 'none':
            sliceshapeNCM = 4
        else:
            sliceshapeNCM = 3
        slicenanNCM = 'nan'
        lat1,lon1,data = NCM.read_NClimGrid_monthlyMEDS(variq,directorydataNCM,monthlychoice,
                                                   sliceyearNCM,sliceshapeNCM,slicenanNCM)
    elif dataset == 'NClimGrid_HIGHS':
        import read_NClimGrid_monthlyHIGHS as NCH
        directorydataNCH = '/work/Zachary.Labe/Data/'
        sliceyearNCH = np.arange(1895,2021+1,1)
        if monthlychoice == 'none':
            sliceshapeNCH = 4
        else:
            sliceshapeNCH = 3
        slicenanNCH = 'nan'
        lat1,lon1,data = NCH.read_NClimGrid_monthlyHIGHS(variq,directorydataNCH,monthlychoice,
                                                   sliceyearNCH,sliceshapeNCH,slicenanNCH)
    elif dataset == 'ERA5_025x025':
        import read_ERA5_monthly025x025 as ERAH
        directorydataERAH = '/work/Zachary.Labe/Data/'
        sliceyearERAH = np.arange(1979,2021+1,1)
        sliceshapeERAH = 3
        slicenanERAH = 'nan'
        levelERAH = 'nan'
        lat1,lon1,data = ERAH.read_ERA5_monthly025x025(variq,directorydataERAH,monthlychoice,
                                                       sliceyearERAH,sliceshapeERAH,True,
                                                       slicenanERAH,levelERAH)
    elif dataset == 'ERA5_1x1':
        import read_ERA5_monthly1x1 as ERA
        directorydataERA = '/work/Zachary.Labe/Data/'
        sliceyearERA = np.arange(1979,2021+1,1)
        if monthlychoice == 'none':
            sliceshapeERA = 4
        else:
            sliceshapeERA = 3
        slicenanERA = 'nan'
        levelERA = 'nan'
        lat1,lon1,lev1,data = ERA.read_ERA5_monthly1x1(variq,directorydataERA,monthlychoice,
                                                  sliceyearERA,sliceshapeERA,True,
                                                  slicenanERA,levelERA)
    elif dataset == 'ERA5_LOWS':
        import read_ERA5_monthlyLOWS as ERAL
        directorydataERAL = '/work/Zachary.Labe/Data/'
        sliceyearERAL = np.arange(1979,2021+1,1)
        if monthlychoice == 'none':
            sliceshapeERAL = 4
        else:
            sliceshapeERAL = 3
        slicenanERAL = 'nan'
        levelERAL = 'surface'
        scenario = 'nan'
        lat1,lon1,lev1,data = ERAL.read_ERA5_monthlyLOWS(variq,directorydataERAL,monthlychoice,
                                                  sliceyearERAL,sliceshapeERAL,True,
                                                  slicenanERAL,levelERAL)
    elif dataset == 'ERA5_MEDS':
        import read_ERA5_monthlyMEDS as ERAM
        directorydataERAM = '/work/Zachary.Labe/Data/'
        sliceyearERAM = np.arange(1979,2021+1,1)
        if monthlychoice == 'none':
            sliceshapeERAM = 4
        else:
            sliceshapeERAM = 3
        slicenanERAM = 'nan'
        levelERAM = 'surface'
        scenario = 'nan'
        lat1,lon1,lev1,data = ERAM.read_ERA5_monthlyMEDS(variq,directorydataERAM,monthlychoice,
                                                  sliceyearERAM,sliceshapeERAM,True,
                                                  slicenanERAM,levelERAM)
    elif dataset == 'ERA5_HIGHS':
        import read_ERA5_monthlyHIGHS as ERAH
        directorydataERAH = '/work/Zachary.Labe/Data/'
        sliceyearERAH = np.arange(1979,2021+1,1)
        if monthlychoice == 'none':
            sliceshapeERAH = 4
        else:
            sliceshapeERAH = 3
        slicenanERAH = 'nan'
        levelERAH = 'surface'
        scenario = 'nan'
        lat1,lon1,lev1,data = ERAH.read_ERA5_monthlyHIGHS(variq,directorydataERAH,monthlychoice,
                                                  sliceyearERAH,sliceshapeERAH,True,
                                                  slicenanERAH,levelERAH)
    elif dataset == 'LENS1':
        import read_LENS1 as LEO
        directorydataLEO = '/work/Zachary.Labe/Data/'
        sliceyearLEO = np.arange(1920,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeLEO = 5
        else:
            sliceshapeLEO = 4
        slicenanLEO = 'nan'
        levelLEO = 'surface'
        numOfEnsLEO = 40
        scenario = 'RCP85'
        lat1,lon1,data = LEO.read_LENS1(directorydataLEO,variq,monthlychoice,
                                            sliceshapeLEO,slicenanLEO,numOfEnsLEO,'futureforcing')
    elif dataset == 'LENS1_LOWS':
        import read_LENS1_LOWS as LEOL
        directorydataLEOL = '/work/Zachary.Labe/Data/'
        sliceyearLEOL = np.arange(1920,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeLEOL = 5
        else:
            sliceshapeLEOL = 4
        slicenanLEOL = 'nan'
        levelLEOL = 'surface'
        numOfEnsLEOL = 40
        scenario = 'RCP85'
        lat1,lon1,data = LEOL.read_LENS1_LOWS(directorydataLEOL,variq,monthlychoice,
                                              sliceshapeLEOL,slicenanLEOL,numOfEnsLEOL,'futureforcing')
    elif dataset == 'LENS2':
        import read_LENS2 as LE
        directorydataLE = '/work/Zachary.Labe/Data/LENS2/monthly/'
        sliceyearLE = np.arange(1850,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeLE = 5
        else:
            sliceshapeLE = 4
        slicenanLE = 'nan'
        levelLE = 'surface'
        numOfEnsLE = 100
        scenario = 'SSP370'
        lat1,lon1,data = LE.read_LENS2(directorydataLE,variq,monthlychoice,
                                            sliceshapeLE,slicenanLE,numOfEnsLE,'futureforcing')
    elif dataset == 'LENS2_smoothed':
        import read_LENS2_smoothed as LEsm
        directorydataLEsm = '/work/Zachary.Labe/Data/'
        sliceyearLEsm = np.arange(1850,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeLEsm = 5
        else:
            sliceshapeLEsm = 4
        slicenanLEsm = 'nan'
        levelLEsm = 'surface'
        numOfEnsLEsm = 50
        scenario = 'SSP370'
        lat1,lon1,data = LEsm.read_LENS2_smoothed(directorydataLEsm,variq,monthlychoice,
                                                sliceshapeLEsm,slicenanLEsm,numOfEnsLEsm,'futureforcing')
    elif dataset == 'LENS2_cmip6bb':
        import read_LENS2_cmip6bb as LEcmip
        directorydataLEcmip = '/work/Zachary.Labe/Data/'
        sliceyearLEcmip = np.arange(1850,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeLEcmip = 5
        else:
            sliceshapeLEcmip = 4
        slicenanLEcmip = 'nan'
        levelLEcmip = 'surface'
        numOfEnsLEcmip = 50
        scenario = 'SSP370'
        lat1,lon1,data = LEcmip.read_LENS2_cmip6bb(directorydataLEcmip,variq,monthlychoice,
                                                sliceshapeLEcmip,slicenanLEcmip,numOfEnsLEcmip,'futureforcing')
    elif dataset == 'LENS2_LOWS':
        import read_LENS2_LOWS as LEL
        directorydataLEL = '/work/Zachary.Labe/Data/LENS2_LOWS/monthly/'
        sliceyearLEL = np.arange(1850,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeLEL = 5
        else:
            sliceshapeLEL = 4
        slicenanLEL = 'nan'
        levelLEL = 'surface'
        numOfEnsLEL = 100
        scenario = 'SSP370'
        lat1,lon1,data = LEL.read_LENS2_LOWS(directorydataLEL,variq,monthlychoice,
                                             sliceshapeLEL,slicenanLEL,numOfEnsLEL,'futureforcing')
    elif dataset == 'LENS2_smoothed_LOWS':
        import read_LENS2_smoothed_LOWS as LEsml
        directorydataLEsml = '/work/Zachary.Labe/Data/'
        sliceyearLEsml = np.arange(1850,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeLEsml = 5
        else:
            sliceshapeLEsml = 4
        slicenanLEsml = 'nan'
        levelLEsml = 'surface'
        numOfEnsLEsml = 50
        scenario = 'SSP370'
        lat1,lon1,data = LEsml.read_LENS2_smoothed_LOWS(directorydataLEsml,variq,monthlychoice,
                                                sliceshapeLEsml,slicenanLEsml,numOfEnsLEsml,'futureforcing')
    elif dataset == 'LENS2_cmip6bb_LOWS':
        import read_LENS2_cmip6bb_LOWS as LEcmipl
        directorydataLEcmipl = '/work/Zachary.Labe/Data/'
        sliceyearLEcmipl = np.arange(1850,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeLEcmipl = 5
        else:
            sliceshapeLEcmipl = 4
        slicenanLEcmipl = 'nan'
        levelLEcmipl = 'surface'
        numOfEnsLEcmipl = 50
        scenario = 'SSP370'
        lat1,lon1,data = LEcmipl.read_LENS2_cmip6bb_LOWS(directorydataLEcmipl,variq,monthlychoice,
                                                sliceshapeLEcmipl,slicenanLEcmipl,numOfEnsLEcmipl,'futureforcing')
    elif dataset == 'MIROC6_LE_LOWS':
        import read_MIROC6_LE_LOWS as MIL
        directorydataMIL = '/work/Zachary.Labe/Data/'
        sliceyearMIL = np.arange(1850,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeMIL = 5
        else:
            sliceshapeMIL = 4
        slicenanMIL = 'nan'
        levelMIL = 'surface'
        numOfEnsMIL = 50
        scenario = 'SSP585'
        lat1,lon1,data = MIL.read_MIROC6_LE_LOWS(directorydataMIL,variq,monthlychoice,
                                                 sliceshapeMIL,slicenanMIL,numOfEnsMIL,'futureforcing')
    elif dataset == 'SMHI_LE_LOWS':
        import read_SMHI_LE_LOWS as SML
        directorydataSML = '/work/Zachary.Labe/Data/'
        sliceyearSML = np.arange(1970,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeSML = 5
        else:
            sliceshapeSML = 4
        slicenanSML = 'nan'
        levelSML = 'surface'
        numOfEnsSML = 50
        scenario = 'SSP585'
        lat1,lon1,data = SML.read_SMHI_LE_LOWS(directorydataSML,variq,monthlychoice,
                                               sliceshapeSML,slicenanSML,numOfEnsSML,'futureforcing')
    elif dataset == 'MPI_ESM12_HR_LOWS':
        import read_MPI_ESM12_HR_LOWS as MPIHRL
        directorydataMPIHRL = '/work/Zachary.Labe/Data/'
        sliceyearMPIHRL = np.arange(1850,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeMPIHRL = 5
        else:
            sliceshapeMPIHRL = 4
        slicenanMPIHRL = 'nan'
        levelMPIHRL = 'surface'
        numOfEnsMPIHRL = 10
        scenario = 'SSP370'
        lat1,lon1,data = MPIHRL.read_MPI_ESM12_HR_LOWS(directorydataMPIHRL,variq,monthlychoice,
                                                       sliceshapeMPIHRL,slicenanMPIHRL,numOfEnsMPIHRL,'futureforcing')
    elif dataset == 'MPI_ESM12_LE':
        import read_MPI_ESM12_LE as MPILR
        directorydataMPILR = '/work/Zachary.Labe/Data/'
        sliceyearMPILR = np.arange(1850,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeMPILR = 5
        else:
            sliceshapeMPILR = 4
        slicenanMPILR = 'nan'
        levelMPILR = 'surface'
        numOfEnsMPILR = 30
        scenario = 'SSP585'
        lat1,lon1,data = MPILR.read_MPI_ESM12_LE(directorydataMPILR,variq,monthlychoice,
                                                       sliceshapeMPILR,slicenanMPILR,numOfEnsMPILR,'futureforcing')
    elif dataset == 'MPI_ESM12_HR_LOWS':
        import read_MPI_ESM12_HR_LOWS as MPIHRL
        directorydataMPIHRL = '/work/Zachary.Labe/Data/'
        sliceyearMPIHRL = np.arange(1850,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeMPIHRL = 5
        else:
            sliceshapeMPIHRL = 4
        slicenanMPIHRL = 'nan'
        levelMPIHRL = 'surface'
        numOfEnsMPIHRL = 10
        scenario = 'SSP370'
        lat1,lon1,data = MPIHRL.read_MPI_ESM12_HR_LOWS(directorydataMPIHRL,variq,monthlychoice,
                                                       sliceshapeMPIHRL,slicenanMPIHRL,numOfEnsMPIHRL,'futureforcing')
    elif dataset == 'FLOR':
        import read_FLOR as F
        directorydataF = '/work/Zachary.Labe/Data/'
        sliceyearF = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeF = 5
        else:
            sliceshapeF = 4
        slicenanF = 'nan'
        levelF = 'surface'
        numOfEnsF = 30
        scenario = 'RCP85'
        lat1,lon1,data = F.read_FLOR(directorydataF,variq,monthlychoice,
                                            sliceshapeF,slicenanF,numOfEnsF,'futureforcing')
    elif dataset == 'FLOR_LOWS':
        import read_FLOR_LOWS as FL
        directorydataFL = '/work/Zachary.Labe/Data/'
        sliceyearFL = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeFL = 5
        else:
            sliceshapeFL = 4
        slicenanFL = 'nan'
        levelFL = 'surface'
        numOfEnsFL = 30
        scenario = 'RCP85'
        lat1,lon1,data = FL.read_FLOR_LOWS(directorydataFL,variq,monthlychoice,
                                            sliceshapeFL,slicenanFL,numOfEnsFL,'futureforcing')
    elif dataset == 'SPEAR_LOW':
        import read_SPEAR_LOW as SPL
        directorydataSPL = '/work/Zachary.Labe/Data/SPEAR/SPEAR_LOW/monthly/'
        sliceyearSPL = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeSPL = 5
        else:
            sliceshapeSPL = 4
        slicenanSPL = 'nan'
        levelSPL = 'surface'
        numOfEnsSPL = 30
        scenario = 'SSP585'
        lat1,lon1,data = SPL.read_SPEAR_LOW(directorydataSPL,variq,monthlychoice,
                                            sliceshapeSPL,slicenanSPL,numOfEnsSPL,'futureforcing')
    elif dataset == 'SPEAR_MED':
        import read_SPEAR_MED as SPM
        directorydataSPM = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/'
        sliceyearSPM = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeSPM = 5
        else:
            sliceshapeSPM = 4
        slicenanSPM = 'nan'
        levelSPM = 'surface'
        numOfEnsSPM = 30
        scenario = 'SSP585'
        lat1,lon1,data = SPM.read_SPEAR_MED(directorydataSPM,variq,monthlychoice,
                                            sliceshapeSPM,slicenanSPM,numOfEnsSPM,'futureforcing')
    elif dataset == 'SPEAR_MED_Historical':
        import read_SPEAR_MED_Historical as SPMh
        directorydataSPMh = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/'
        sliceyearSPMh = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeSPMh = 5
        else:
            sliceshapeSPMh = 4
        slicenanSPMh = 'nan'
        levelSPMh = 'surface'
        numOfEnsSPMh = 30
        scenario = 'SSP585'
        lat1,lon1,data = SPMh.read_SPEAR_MED_Historical(directorydataSPMh,variq,monthlychoice,
                                            sliceshapeSPMh,slicenanSPMh,numOfEnsSPMh,'historicalforcing')
    elif dataset == 'SPEAR_MED_ALLofHistorical':
        import read_SPEAR_MED_Historical as SPMhall
        directorydataSPMhall = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/'
        sliceyearSPMhall = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeSPMhall = 5
        else:
            sliceshapeSPMhall = 4
        slicenanSPMhall = 'nan'
        levelSPMhall = 'surface'
        numOfEnsSPMhall = 30
        scenario = 'SSP585'
        lat1,lon1,data = SPMhall.read_SPEAR_MED_Historical(directorydataSPMhall,variq,monthlychoice,
                                            sliceshapeSPMhall,slicenanSPMhall,numOfEnsSPMhall,'ALLofhistoricalforcing')
    elif dataset == 'SPEAR_MED_FA':
        import read_SPEAR_MED_FA as SPMF
        directorydataSPMF = '/work/Zachary.Labe/Data/'
        sliceyearSPMF = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeSPMF = 5
        else:
            sliceshapeSPMF = 4
        slicenanSPMF = 'nan'
        levelSPMF = 'surface'
        numOfEnsSPMF = 30
        scenario = 'SSP585'
        lat1,lon1,data = SPMF.read_SPEAR_MED_FA(directorydataSPMF,variq,monthlychoice,
                                            sliceshapeSPMF,slicenanSPMF,numOfEnsSPMF,'futureforcing')
    elif dataset == 'SPEAR_MED_Scenario':
        import read_SPEAR_MED_Scenario as SPSS
        directorydataSPSS = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/'
        sliceyearSPSS = np.arange(1921,2100+1,1)
        sliceshapeSPSS = 4
        slicenanSPSS = 'nan'
        levelSPSS = 'surface'
        numOfEnsSPSS = 30
        scenario = scenario
        lat1,lon1,data = SPSS.read_SPEAR_MED_Scenario(directorydataSPSS,scenario,variq,monthlychoice,
                                            sliceshapeSPSS,slicenanSPSS,numOfEnsSPSS,'futureforcing')
    elif dataset == 'SPEAR_MED_SSP534OS_10ye':
        import read_SPEAR_MED_SSP534OS_10ye as SPSS10
        directorydataSPSS10 = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_SSP534OS_10ye/monthly/'
        sliceyearSPSS10 = np.arange(1921,2100+1,1)
        sliceshapeSPSS10 = 4
        slicenanSPSS10 = 'nan'
        levelSPSS10 = 'surface'
        numOfEnsSPSS10 = 30
        lat1,lon1,data = SPSS10.read_SPEAR_MED_SSP534OS_10ye(directorydataSPSS10,variq,monthlychoice,
                                                             sliceshapeSPSS10,slicenanSPSS10,numOfEnsSPSS10,'futureforcing')
    elif dataset == 'SPEAR_MED_LM42p2_test':
        import read_SPEAR_MED_LM42p2_test as LM42p2test
        directorydataLM42p2test = '/work/Zachary.Labe/Data/'
        sliceyearLM42p2test = np.arange(1921,2070+1,1)
        sliceshapeLM42p2test = 4
        slicenanLM42p2test = 'nan'
        levelLM42p2test = 'surface'
        numOfEnsLM42p2test = 3
        lat1,lon1,data = LM42p2test.read_SPEAR_MED_LM42p2_test(directorydataLM42p2test,variq,monthlychoice,
                                                             sliceshapeLM42p2test,slicenanLM42p2test,numOfEnsLM42p2test,'futureforcing')
    elif dataset == 'SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv':
        import read_SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv as SPSS10amoc
        directorydataSPSS10amoc = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv/monthly/'
        sliceyearSPSS10amoc = np.arange(1921,2100+1,1)
        sliceshapeSPSS10amoc = 4
        slicenanSPSS10amoc = 'nan'
        levelSPSS10amoc = 'surface'
        numOfEnsSPSS10amoc = 9
        lat1,lon1,data = SPSS10amoc.read_SPEAR_MED_SSP534OS_STRONGAMOC_p1Sv(directorydataSPSS10amoc,variq,monthlychoice,
                                                                            sliceshapeSPSS10amoc,slicenanSPSS10amoc,numOfEnsSPSS10amoc,'futureforcing')
    elif dataset == 'SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv':
        import read_SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv as SPSS10amoc2
        directorydataSPSS10amoc2 = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv/monthly/'
        sliceyearSPSS10amoc2 = np.arange(1921,2100+1,1)
        sliceshapeSPSS10amoc2 = 4
        slicenanSPSS10amoc2 = 'nan'
        levelSPSS10amoc2 = 'surface'
        numOfEnsSPSS10amoc2 = 9
        lat1,lon1,data = SPSS10amoc2.read_SPEAR_MED_SSP534OS_STRONGAMOC_p2Sv(directorydataSPSS10amoc2,variq,monthlychoice,
                                                                            sliceshapeSPSS10amoc2,slicenanSPSS10amoc2,numOfEnsSPSS10amoc2,'futureforcing')
    elif dataset == 'SPEAR_MED_NATURAL':
        import read_SPEAR_MED_NATURAL as SPN
        directorydataSPN = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_NATURAL/monthly/'
        sliceyearSPN = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeSPN = 5
        else:
            sliceshapeSPN = 4
        slicenanSPN = 'nan'
        levelSPN = 'surface'
        numOfEnsSPN = 30
        scenario = 'natural'
        lat1,lon1,data = SPN.read_SPEAR_MED_NATURAL(directorydataSPN,variq,monthlychoice,
                                            sliceshapeSPN,slicenanSPN,numOfEnsSPN,'futureforcing')
    elif dataset == 'SPEAR_MED_NATURAL_ALLYRS':
        import read_SPEAR_MED_NATURAL as SPNa
        directorydataSPNa = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_NATURAL/monthly/'
        sliceyearSPNa = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeSPNa = 5
        else:
            sliceshapeSPNa = 4
        slicenanSPNa = 'nan'
        levelSPNa = 'surface'
        numOfEnsSPNa = 30
        scenario = 'natural'
        lat1,lon1,data = SPNa.read_SPEAR_MED_NATURAL(directorydataSPNa,variq,monthlychoice,
                                            sliceshapeSPNa,slicenanSPNa,numOfEnsSPNa,'alldet')
    elif dataset == 'SPEAR_MED_NOAER':
        import read_SPEAR_MED_NOAER as SPNO
        directorydataSPNO = '/work/Zachary.Labe/Data/SPEAR/SPEAR_MED_NOAER/monthly/'
        sliceyearSPNO = np.arange(1921,2020+1,1)
        if monthlychoice == 'none':
            sliceshapeSPNO = 5
        else:
            sliceshapeSPNO = 4
        slicenanSPNO = 'nan'
        levelSPNO = 'surface'
        numOfEnsSPNO = 12
        scenario = 'SSP585'
        lat1,lon1,data = SPNO.read_SPEAR_MED_NOAER(directorydataSPNO,variq,monthlychoice,
                                            sliceshapeSPNO,slicenanSPNO,numOfEnsSPNO,'futureforcing')
    elif dataset == 'SPEAR_HIGH':
        import read_SPEAR_HIGH as SPH
        directorydataSPH = '/work/Zachary.Labe/Data/'
        sliceyearSPH = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeSPH = 5
        else:
            sliceshapeSPH = 4
        slicenanSPH = 'nan'
        levelSPH = 'surface'
        numOfEnsSPH = 10
        scenario = 'SSP585'
        lat1,lon1,data = SPH.read_SPEAR_HIGH(directorydataSPH,variq,monthlychoice,
                                            sliceshapeSPH,slicenanSPH,numOfEnsSPH,'futureforcing')
    elif dataset == 'MMLEA':
        import read_MMLEA as MMLEA
        directorydataMMLEA = '/work/Zachary.Labe/Data/'
        sliceyearMMLEA = np.arange(1921,2100+1,1)
        if monthlychoice == 'none':
            sliceshapeMMLEA = 5
        else:
            sliceshapeMMLEA = 4
        slicenanMMLEA = 'nan'
        lat1,lon1,data = MMLEA.read_MMLEA(directorydataMMLEA,variq,monthlychoice,
                                         sliceshapeMMLEA,slicenanMMLEA,'futureforcing')

    else:
        print(ValueError('WRONG DATA SET SELECTED!'))
        sys.exit()
        
    print('>>>>>>>>>> Completed: Finished readFiles function!')
    return data,lat1,lon1  

def getRegion(data,lat1,lon1,lat_bounds,lon_bounds):
    """
    Function masks out region for data set

    Parameters
    ----------
    data : 2d+ numpy array
        original data set
    lat1 : 1d array
        latitudes
    lon1 : 1d array
        longitudes
    lat_bounds : 2 floats
        (latmin,latmax)
    lon_bounds : 2 floats
        (lonmin,lonmax)
        
    Returns
    -------
    data : numpy array
        MASKED data from selected data set
    lat1 : 1d numpy array
        latitudes
    lon1 : 1d numpy array
        longitudes

    Usage
    -----
    data,lats,lons = getRegion(data,lat1,lon1,lat_bounds,lon_bounds)
    """
    print('\n>>>>>>>>>> Using get_region function!')
    
    ### Import modules
    import numpy as np
    
    ### Note there is an issue with 90N latitude (fixed!)
    lat1 = np.round(lat1,3)
    
    ### Mask latitudes
    if data.ndim == 2:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[latq,:] 
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,lonq]
        
    elif data.ndim == 3:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:,latq,:] 
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,:,lonq]
        
    elif data.ndim == 4:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:,:,latq,:]        
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,:,:,lonq]
        
    elif data.ndim == 5:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:,:,:,latq,:]
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,:,:,:,lonq]
        
    elif data.ndim == 6:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:,:,:,:,latq,:]
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,:,:,:,:,lonq]
    
    ### New variable name
    datanew = datalonq
    
    print('>>>>>>>>>> Completed: getRegion function!')
    return datanew,latn,lonn   

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# import calc_Stats as SSS

# variq = 'T2M'
# dataset = 'NClimGrid_MEDS'
# monthlychoice = 'JJA'
# scenario = 'SSP534OS'
# data,lat1,lon1 = readFiles(variq,'SPEAR_MED_Scenario',monthlychoice,scenario)
# # obs,lat1n,lon1n = readFiles(variq,'NClimGrid_MEDS',monthlychoice,scenario)

# lon2,lat2 = np.meshgrid(lon1,lat1)
# ave = UT.calc_weightedAve(data,lat2)
