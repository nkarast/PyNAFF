##################################################################################################################
#
#   PyNAFF  -  A Cython module for calculating the Fundamental Frequencies of Quasi-Periodic Signals
#              Based on the work of J. Lashkar : Numerical Analysis of Fundamental Frequencies
#              (http://www.sciencedirect.com/science/article/pii/001910359090084M)
#
#   Authors : Nikos Karastathis ( nkarast .at. cern .dot. ch )
#             Panagiotis Zisopoulos ( psizopou .at. cern .dot. ch)
#
#   Version : 1.0 (13/03/2017)
#
##################################################################################################################
""" PyNAFF

A module that calculates the fundamental frequencies of Quasi-Periodic Signals
based pm the work of J. Lashkar: Numerical Analysis of Fundamental Frequencies

@  Authors : Nikos Karastathis ( nkarast .at. cern .dot. ch )
             Panagiotis Zisopoulos ( psizopou .at. cern .dot. ch)

@  Version : 1.0 (13/03/2017)


The module at the moment resides in a single file. This contains the main PyNAFF class
that has all the functions to calculate the fundamanetal frequencies of the signal.

The driving routine is called `naff()` and is used for data that are given either in
textual form (input tabulated file), or a numpy array.

Finally the code can be ran as a script:
$ > python PyNAFF.py filename icol icx icy nulin KTABS nterm iw

"""
import sys
import numpy as np
cimport numpy as np
from logging import *
from cpython cimport bool

# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
cdef double __version__       = 1.0
cdef list __authors__       = ["Nikos Karastathis (nkarast .at. cern .dot. ch)", "Panagiotis Zisopoulos (pzisopou .at. cern .dot. ch)"]
cdef str __pythonVersion__ = '2.7.11+'

# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
cdef class PyNAFF(object):
    """PyNAFF class()

        The class that holds a NAFF study for a fundamental frequencies of a quasiperiodic signal.
        -- Members of the Class defined in the constructor:

            *  logfile
            *  loglevel
            *  inputParamFilename
            *  parametersFilename
            *  outputFilename
            *  dataFilename
            *  solutionFilename
            *  tabFilename

            *  DTOUR      # length of a turn in units under considertation
            *  T0         # initial data
            *  XH         # step (positive or neg)
            *  KTABS      # how many points for naff
            *  NEPS       # precision for freq search
            *  NTERM      # number of terms asked for
            *  ICPLX      # 1=data is complex
            *  IW         # should I add window?
            *  NULIN      # lines skipped in file
            *  ICOL       # number of columns in file
            *  ICX        # column number for x data (starting from 1)
            *  ICY        # column number for y data (starting from 1)
            *  IPRNAF     # print staff in mftnaff
            *  IFTES      # no idea
            *  IFSAV      # a secret lost in the time
            *  IFCOMP     # compare frequencies
            *  IFRES      # do residuals
            *  IFSTAB     # save tab file
            *  IMODT      # another secret
            *  TMF1       # yet another secret
            *  TMF2       # you guess...
            *  TNT        # you guess...

            *  batch      # batch mode
            *  debug      # debug mode
            *  saveFile   # save output files
            *  isFile     # flag to define if data (i.e. self.filename) is file
            *  result     # list to keep the results to be returned


        -- Functions of the Class:
            * __init__()
            * initLogger()
            * setParametersFilename()
            * setOutputFilename()
            * setDataFilename()
            * setSolutionFilename()
            * setTabFilename()
            * setDTOUR()
            * setT0()
            * setXH()
            * setKTABS()
            * setNEPS()
            * setNTERM()
            * setICPLX()
            * setIW()
            * setNULIN()
            * setICOL()
            * setICX()
            * setICY()
            * setIPRNAF()
            * setIPRNAF()
            * setIFTES()
            * setIFSAV()
            * setIFCOMP()
            * setIFRES()
            * setIFRES()
            * setIFSTAB()
            * setIMODT()
            * setTMF1()
            * setTMF2()
            * setTNT()
            * readParamsFile()
            * printConfig()
            * inipar()
            * prtpar()
            * initAngleStep()
            * iniwin()
            * loadArray()
            * loadsol()
            * prtabs()
            * inifre()
            * puiss2()
            * maxx()
            * fftmax()
            * frefin()
            * maxiqua()
            * func()
            * profre()
            * ztpow2()
            * ztpow()
            * ztpow2a()
            * zardyd()
            * fretes()
            * proscaa()
            * gramsc()
            * modfre()
            * mftnaf()
            * savsol()
            * appendToSol()
            * tessol()
            * compso()
            * savtab()
            * run()
    """


    cdef str logfile, inputParamFilename, parametersFilename, outputFilename, solutionFilename, tabFilename, filename, dataFilename
    cdef unsigned int loglevel, KTABS, NTERM, ICPLX, NULIN, ICOL, ICX, IPRNAF, IFTES, IFSAV, IFCOMP, IFRES, IFSTAB, IMODT, IKT, NFR, KTABSM, NFS
    cdef signed int IW, ICY
    cdef double DTOUR, T0, XH, NEPS, UNIANG, FREFON, EPS, AF, BF
    cdef double [:] TMF1, TMF2, TNT, TFS, TFSR, TWIN
    cdef double complex [:] ZTABS, ZRTABS, ZAMP, ZAMPR, dataArray
    cdef double complex [:,:] ZALP
    cdef list result

    cdef bool isFile, batch, debug, saveFiles

    def __init__(self, double complex [:] data_array = None, str filename='track.obs0001.p0001', bool isFile=True, str inputParamFile=None, bool saveFiles=False, str solutionFile='track.sol', str parametersOutLog='nafpar.par',
        str outBuffFile='naf.out', str tabFile='naf.tab', unsigned int KTABS=300, bool batch=False, bool debug=False, str logfile=None, unsigned int loglevel=10, double neps=100000000.,
        unsigned int nterm=1, unsigned int icplx=1, signed int iw=+1, unsigned int nulin=0, unsigned int icol=1, unsigned int icx=1, signed int icy=1, unsigned int iprnaf=0, unsigned int iftes=0, 
        unsigned int ifsav=0, unsigned int ifcomp=0, unsigned int ifres=0, unsigned int ifstab=1, unsigned int imodt=0):

    # def __init__(self, data_array = None, filename='track.obs0001.p0001', isFile=True, inputParamFile=None, saveFiles=False,  solutionFile='track.sol',  parametersOutLog='nafpar.par',
    #     outBuffFile='naf.out',  tabFile='naf.tab', KTABS=300,  batch=False, debug=False,  logfile=None,  loglevel=10,  neps=100000000., nterm=1, icplx=1, iw=+1, nulin=0, 
    #     icol=1, icx=1, icy=1, iprnaf=0, iftes=0, ifsav=0, ifcomp=0, ifres=0, ifstab=1,imodt=0):

        self.logfile  = logfile
        self.loglevel = loglevel
        self.initLogger(logfile, loglevel)

        self.NFS = 0
        self.TFS = np.zeros(self.NFR).astype('float64')
        self.ZAMP = np.zeros(self.NFR).astype('complex128') #np.array([np.complex(0.,0.) for i in xrange(self.NFR)])
        self.ZALP = np.zeros((self.NFR,self.NFR)).astype('complex128')

        self.inputParamFilename     = inputParamFile
        self.parametersFilename     = parametersOutLog
        self.outputFilename         = outBuffFile
        self.dataFilename           = filename
        self.dataArray              = data_array
        self.solutionFilename       = solutionFile
        self.tabFilename            = tabFile

        self.DTOUR                  = np.float64(6.283185307179586476925)   # length of a turn in units under considertation
        self.T0                     = 0.0                                   # initial data
        self.XH                     = np.float64(6.283185307179586476925)   # step (positive or neg)
        self.KTABS                  = int(KTABS) #3300                      # how many points for naff
        self.NEPS                   = np.float64(neps)                      # precision for freq search
        self.NTERM                  = nterm                                 # number of terms asked for
        self.ICPLX                  = icplx                                 # 1=data is complex
        self.IW                     = iw                                    # should I add window?
        self.NULIN                  = nulin                                 # lines skipped in file
        self.ICOL                   = icol                                  # number of columns in file
        self.ICX                    = icx                                   # column number for x data (starting from 1)
        self.ICY                    = icy                                   # column number for y data (starting from 1)
        self.IPRNAF                 = iprnaf                                # print staff in mftnaff
        self.IFTES                  = iftes                                 # no idea
        self.IFSAV                  = ifsav                                 # a secret lost in the time
        self.IFCOMP                 = ifcomp                                # compare frequencies
        self.IFRES                  = ifres                                 # do residuals
        self.IFSTAB                 = ifstab                                # save tab file
        self.IMODT                  = imodt                                 # another secret
        self.TMF1                   = np.array([-100. , 35., 70., -35.])    # yet another secret
        self.TMF2                   = np.array([-5. , 70., 100., -24.])     # you guess...
        self.TNT                    = np.array([10., 10., 10., 10.])        # you guess...

        self.batch                  = batch                                 # batch mode
        self.debug                  = debug                                 # debug mode
        self.saveFiles              = saveFiles                             # save output files
        self.isFile                 = isFile                                # flag to define if data (i.e. self.filename) is file
        if not isFile:
            self.result             = []#np.array([] , dtype=np.float64)                                    # list to keep the results to be returned

        self.UNIANG = 0.0                                           # unit of the angle = 1rd = 1 uniang unit of the angle (if the unit is in seconds of the arc then UNIANG = 206264.80624709)
        self.FREFON = 0.0                                          # Fundamental Frequency : FREFON = 2*pi/(KTABS*XH) or in seconds FREFON = 360*3600 / (KTABS*XH)
        self.EPS    = 0.0                                          # Precision
        self.IKT    = 0                                          # Step
        self.AF = 0.0                                              # Real part of amplitude
        self.BF = 0.0                                              # Imaginary part of amplitude

        # these are used for the main function
        self.NFR    = 100                                           # maximum number of frequencies allowed  # @TODO this is useless
        self.KTABSM = 500000                                        # maximum number of turns allowed        # @TODO this is useless
        self.ZRTABS = np.zeros(self.KTABSM+1).astype('complex128')  # Temporary real tab array  (used for comparison of frequencies)
        self.ZAMPR  = np.zeros(self.NFR).astype('complex128')       # Temporary amplitude array (used for comparison of frequencies)
        self.TFSR   = np.zeros(self.NFR)                            # Temporary frequency array (used for comparison of frequencies)

        self.TWIN   = np.zeros(self.KTABS+1)                        # Array with the window values
        self.ZTABS  = np.zeros(self.KTABS+1).astype('complex128')   # Array with the complex data
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void initLogger(self, str logfile, int loglevel):
        '''Function to initialize the python logger.
        Input:  logfile  : None -> If logfile is not None then a file will be created
                loglevel : Level of output to be printed out:
                        -> 0: NOT SET, 10: DEBUG, 20: INFO, 30: WARN, 40: ERROR, 50: FATAL
        '''
        cdef str FORMAT 

        FORMAT = '%(asctime)s %(levelname)s : %(message)s'
        basicConfig(format=FORMAT, filename=logfile, level=loglevel)
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setParametersFilename(self, str paramsFname):
        '''Accessor function
        '''
        self.parametersFilename = paramsFname
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setOutputFilename(self, str outFname):
        '''Accessor function
        '''
        self.outputFilename = outFname
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setDataFilename(self, str dataFname):
        '''Accessor function
        '''
        self.dataFilename = dataFname
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setSolutionFilename(self, str solFname):
        '''Accessor function
        '''
        self.solutionFilename = solFname
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setTabFilename(self, str tabFname):
        '''Accessor function
        '''
        self.tabFilename = tabFname
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setDTOUR(self, double nDTOUR):
        '''Accessor function
        '''
        self.DTOUR = np.float64(nDTOUR)
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setT0(self, double nT0):
        '''Accessor function
        '''
        self.T0 = nT0
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setXH(self, double nXH):
        '''Accessor function
        '''
        self.XH = nXH
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setKTABS(self, int nKTABS):
        '''Accessor function
        '''
        self.KTABS = int(nKTABS)
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setNEPS(self, double nNEPS):
        '''Accessor function
        '''
        self.NEPS = nNEPS
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setNTERM(self, unsigned int nNTERM):
        '''Accessor function
        '''
        self.NTERM = nNTERM
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setICPLX(self, unsigned int nICPLX):
        '''Accessor function
        '''
        self.ICPLX = nICPLX
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setIW(self, signed int nIW):
        '''Accessor function
        '''
        self.IW = nIW
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setNULIN(self, unsigned int nNULIN):
        '''Accessor function
        '''
        self.NULIN = nNULIN
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setICOL(self, unsigned int nICOL):
        '''Accessor function
        '''
        self.ICOL = nICOL
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setICX(self, unsigned int nICX):
        '''Accessor function
        '''
        self.ICX = nICX
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setICY(self, signed int nICY):
        '''Accessor function
        '''
        self.ICY = nICY
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setIPRNAF(self, unsigned int nIPRNAF):
        '''Accessor function
        '''
        self.IPRNAF = nIPRNAF
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setIFTES(self, unsigned int nIFTES):
        '''Accessor function
        '''
        self.IFTES = nIFTES
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setIFSAV(self, unsigned int nIFSAV):
        '''Accessor function
        '''
        self.IFSAV = nIFSAV
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setIFCOMP(self, unsigned int nIFCOMP):
        '''Accessor function
        '''
        self.IFCOMP = nIFCOMP
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setIFRES(self, unsigned int nIFRES):
        '''Accessor function
        '''
        self.IFRES = nIFRES
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setIFSTAB(self, unsigned int nIFSTAB):
        '''Accessor function
        '''
        self.IFSTAB = nIFSTAB
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setIMODT(self, unsigned int nIMODT):
        '''Accessor function
        '''
        self.IMODT = nIMODT
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setTMF1(self, double [:] nTMF1_arr):
        '''Accessor function
        '''
        self.TMF1 = np.asarray(nTMF1_arr)
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setTMF2(self, double [:] nTMF2_arr):
        '''Accessor function
        '''
        self.TMF2 = np.asarray(nTMF2_arr)
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void setTNT(self, double [:] nTNT_arr):
        '''Accessor function
        '''
        self.TNT = np.asarray(nTNT_arr)
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void readParamsFile(self, str paramsfile='nafaut.par'):
        '''Reads an input file to set the parameters needed by the constructor
        Input: paramsfile : String with the file name
        Returns : None
        '''
        cdef list newParams

        f = open(paramsfile, 'r')
        newParams = []
        for line in f.readlines():
            if r'//' in line:
                continue
            if r'&NAMSTD' in line:
                continue
            #newParams.append(line.split('=')[1].replace(',','').replace('D0',''))
            newParams.append(line.split('=')[1].replace(',','').replace('D0','').replace(' ','').replace('\n','').replace('\'',''))
            self.setParametersFilename(newParams[0])
            self.setOutputFilename(newParams[1])
            self.setDataFilename(newParams[2])
            self.setSolutionFilename(newParams[3])
            self.setTabFilename(newParams[4])
            self.setDTOUR(np.float64(newParams[5]))
            self.setT0(np.float64(newParams[6]))
            self.setXH(np.float64(newParams[7]))
            self.setKTABS(int(newParams[8]))
            self.setNEPS(int(newParams[9]))
            self.setNTERM(int(newParams[10]))
            self.setICPLX(int(newParams[11]))
            self.setIW(int(newParams[12]))
            self.setNULIN(int(newParams[13]))
            self.setICOL(int(newParams[14]))
            self.setICX(int(newParams[15]))
            self.setICY(int(newParams[16]))
            self.setIPRNAF(int(newParams[17]))
            self.setIFTES(int(newParams[18]))
            self.setIFSAV(int(newParams[19]))
            self.setIFCOMP(int(newParams[20]))
            self.setIFRES(int(newParams[21]))
            self.setIFSTAB(int(newParams[22]))
            self.setIMODT(int(newParams[23]))
            nTMF1 = np.array(np.float64(newParams[24]), np.float64(newParams[27]), np.float64(newParams[30]), np.float64(newParams[33]))
            nTMF2 = np.array(np.float64(newParams[25]), np.float64(newParams[28]), np.float64(newParams[31]), np.float64(newParams[34]))
            nTNT  = np.array(np.float64(newParams[26]), np.float64(newParams[29]), np.float64(newParams[32]), np.float64(newParams[35]))
            self.setTMF1(nTMF1)
            self.setTMF2(nTMF2)
            self.setTNT(nTNT)
            info('Input parameters have changed to new configuration.')
            self.printConfig()
        f.close()
# # - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef printConfig(self, bool retString=False):
        '''Prints the current configuration of the instance of PyNAFF class.
        Inputs : retString: (bool) To also return the multiline string created.
        Returns : None or the multiline string.
        '''
        cdef str c

        c = '''
        Numerical Analysis of Fundamental Frequencies
        Adaptation of J. Lashkar's method
        -----------------------------------------------



        parametersFilename      = {}
        outputFilename          = {}
        dataFilename            = {}
        solutionFilename        = {}
        tabFilename             = {}

        DTOUR                   = {}
        T0                      = {}
        XH                      = {}
        KTABS                   = {}
        NEPS                    = {}
        NTERM                   = {}
        ICPLX                   = {}
        IW                      = {}
        NULIN                   = {}
        ICOL                    = {}
        ICX                     = {}
        ICY                     = {}
        IPRNAF                  = {}
        IFTES                   = {}
        IFSAV                   = {}
        IFCOMP                  = {}
        IFRES                   = {}
        IFSTAB                  = {}
        IMODT                   = {}
        TMF1                    = {}
        TMF2                    = {}
        TNT                     = {}'''.format(self.parametersFilename, self.outputFilename, self.dataFilename, self.solutionFilename, self.tabFilename, self.DTOUR, self.T0, self.XH, self.KTABS, self.NEPS, self.NTERM, self.ICPLX, self.IW, self.NULIN, self.ICOL, self.ICX, self.ICY, self.IPRNAF, self.IFTES, self.IFSAV, self.IFCOMP, self.IFRES, self.IFSTAB, self.IMODT, np.asarray(self.TMF1), np.asarray(self.TMF2), np.asarray(self.TNT))
        info(c)
        if retString:
            return c
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void inipar(self, str fromFile=None):
        '''Initializes the parameters either from file or using the default constructor
        Input: fromFile None or String
        Returns: None
        '''
        if fromFile is None:
            fromFile = self.inputParamFilename

            if fromFile is not None:
                info("# inipar : Initialization via parameter file [{}]".format(fromFile))
                self.readParamsFile(fromFile)
        else:
            info("# inipar : Default parametrization.")
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void prtpar(self, str outfile=None):
        '''Print parametrisation of the class in <outfile>
        Input: outfile : name of the file to be created.
        Returns: None
        '''
        if outfile is None:
            outfile = self.parametersFilename

            if outfile is not None:
                fout = open(outfile, 'w')
                info("# prtpar : Saving initial parameters configuration to file [{}]".format(outfile))
                outparam = self.printConfig(retString=True)
                fout.write(outparam)
                fout.close()
            else:
                info("# prtpar : Skipping saving initial parameters (empty parameter file field)")
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void initAngleStep(self):
        '''Additional initialization
        Input: None
        Returns: None
        '''
        self.UNIANG = self.DTOUR/(2.0*np.pi)
        self.FREFON = self.DTOUR/(self.KTABS*self.XH)
        self.EPS    = np.abs(self.FREFON)/self.NEPS
        info("# initAngleStep : Initialized fundamental frequency to FREFON={} and EPS={}".format(self.FREFON, self.EPS))
        self.IKT    = int(self.KTABS/10)
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void iniwin(self, unsigned int KTABS, unsigned int window):
        '''Initialize the window array (self.TWIN). Elements of the window array are calculated depending on the
        integer passed in the function.
            window = 0   : No window is implemented. self.TWIN is an array of ones.
            window = n>0 : Trigonometric Window of power n :
                                                phi(T) = CN*(1+cos(np.pi*t))**n,
                                                where CN = 2**n *(n!)**2 / (2n)!
            window = -1  : Exponential window :
                                                phi(T) = 1.0/CE*np.exp(-1/(1-t**2))
                                                CE= 0.22199690808403971891
        Input:  KTABS  : Number of turns required
                window : (int) define the window
        Returns: None
        '''
        cdef double CE, T1, T2, TM, PIST, XH, CN
        cdef unsigned int i, IT
        CE = np.float64(0.22199690808403971891)
        XH = self.XH
        T1 = self.T0
        T2 = self.T0 + KTABS*XH
        TM = (T2-T1)/2
        PIST = np.pi/TM

        if window == 0 :
            self.TWIN = np.ones(KTABS+1)
        elif window > 0 :
            CN = 1.0

            for i in range(1,window+1):
                CN = CN*2.0*i/(window+1)
            for IT in range(KTABS+1):
                T = (IT)*self.XH-TM
                self.TWIN[IT] = CN*(1.0+np.cos(T*PIST))**window
        elif window == -1:
            self.TWIN[0] = 0.0
            self.TWIN[KTABS] = 0.0

            for IT in range(1,KTABS):
                T=((IT)*XH-TM)/TM
                self.TWIN[IT] = np.exp(-1.0/(1.0-T**2))/CE

        debug("# iniwin :  KTABS = {}, window = {}".format(KTABS, window))
# # - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef loadArray(self, unsigned int KTABS, unsigned int ICX, signed int ICY, unsigned int ICOL, unsigned int NULIN, double complex [:] data):
    # cpdef void loadArray(self, unsigned int KTABS, unsigned int ICX, signed int ICY, unsigned int ICOL, unsigned int NULIN, np.ndarray[np.float64_t] data):
        '''Loads the data from a numpy array. A conversion to dtype('complex128') is done in case
        the input array is not complex
        Input:  KTABS : Number of turns
                ICX   : Column of the X data
                ICY   : Column of the Y data
                ICOL  : Total number of columns
                NULIN : Number of lines to skip
                filename: This is the actual array
        Returns: None
        '''

        ICX = ICX-1
        ICY = ICY-1

        # if len(data.shape) == 1:
        np.asarray(self.ZTABS)[:KTABS+1] = np.asarray(data)[NULIN:NULIN+KTABS+1]
        # else:
        #     raise IndexError("# loadArray : ndarray input not yet implemented.")
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void loadsol(self, unsigned int KTABS, unsigned int ICX, signed int ICY, unsigned int ICOL, unsigned int NULIN, str filename):
        '''Loads the data from a file. If ICY>0 then the data are X+iY
        Input:  KTABS : Number of turns
                ICX   : Column of the X data
                ICY   : Column of the Y data
                ICOL  : Total number of columns
                NULIN : Number of lines to skip
                filename: String of the filename
        Returns: None
        '''
        cdef object data
        cdef np.ndarray[np.float64_t, ndim=1] x, y
        cdef unsigned int i

        if filename is None:
            filename = self.dataFilename

        info("# loadsol : Skipping row option is set to [{}]".format(NULIN))
        data = np.loadtxt(filename, comments=["#", "@", "*", "$"], skiprows=int(NULIN))
        debug("# loadsol : Data Shape = {}".format(data.shape))

        ICX = ICX -1
        ICY = ICY -1
        debug("# loadsol : ICX={}, ICY={}".format(ICX, ICY))
        if ICY < 0:
            debug("# loadsol : Setting y to zeros")
            y = np.zeros(KTABS+1)
        else:
            if len(data.shape)>1:
                y = data[:KTABS+1, ICY]
            else:
                y = data[:KTABS+1]

        if len(data.shape)>1:
            x = data[:KTABS+1, ICX]
        else:
            x = data[:KTABS+1]

        for i in range(KTABS+1):
            self.ZTABS[i] = np.complex(x[i],y[i])
        info("# loadsol : File read successfully | KTABS = len(X) = len(Y) = len(ZTABS) = {} , {}, {}, {}".format(KTABS, len(x), len(y), len(self.ZTABS)))
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void prtabs(self, unsigned int KTABS, object ZTABS, unsigned int IPAS, object where=0):
        '''Prints or saves into a file the TABs
        Input: KTABS : number of turns
               ZTABS : The complex array with tabs
               IPAS  : Step
               where : Flag to either print to STDOUT (where=0) or to a file (where = string)
        Returns : None
        '''
        cdef unsigned int i
        if IPAS == None:
            IPAS = int(self.IKT)

        if where == 0:
            for i in range(0, KTABS, IPAS):
                info("# prtabs : i, Re(ZTABS), Im(ZTABS) = {}  ,  {}  ,  {} ".format(i, np.real(ZTABS[i]), np.imag(ZTABS[i]) ))
        else:
            fout = open(where, 'aw')
            for i in range(0, KTABS, IPAS):
                fout.write("{}\t{}\t{}".format(i, np.real(ZTABS[i]), np.imag(ZTABS[i])))
            fout.close()
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void inifre(self):
        '''Initialize number of frequencies and aplitudes to zero.
        Input: None
        Returns: None
        '''
        self.NFR = 100
        self.NFS = 0
        self.TFS = np.zeros(self.NFR).astype('float64')
        self.ZAMP = np.zeros(self.NFR).astype('complex128') #np.array([np.complex(0.,0.) for i in xrange(self.NFR)])
        self.ZALP = np.zeros((self.NFR,self.NFR)).astype('complex128')
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef unsigned int puiss2(self, unsigned int NT):
        '''Return the largest number that is a power of 2, that fits into the NT
        Input : NT : Number for which the largest power of two is wanted.
        Returns: The largest power of 2
        Example : puiss2(300) = 256
        '''
        cdef unsigned int N, N2
        N = NT
        if N == 0:
            N2 = 0
            return N2
        N2 = 1
        while True:
            if N >= 2:
                N2 = N2*2
                N = N/2
            else:
                break
        return N2
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef maxx(self, unsigned int KTABS2, np.ndarray[np.float64_t, ndim=1] RTAB):
        '''Calculate the maximum and the index of the maximum.
        Input: KTABS2 : Not used
               RTAB   : Real tabs for which the max is wanted.
        Returns: INDX :  The index of the maximum
                 VMAX :  The maximum amplitude.
        '''
        cdef unsigned int INDX
        cdef double VMAX

        INDX = np.argmax(RTAB)  # only the first
        VMAX = np.max(RTAB)
        info("# maxx : Found VMAX={}  at INDX={}".format(VMAX, INDX))
        return INDX, VMAX
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef double fftmax(self, unsigned int KTABS):
        '''Estimate the fundamental frequency
        Input : KTABS : Number of turns
        Returns : FR  : fundamental frequency
        '''
        cdef unsigned int KTABS2, i, IDV, IPAS, INDX
        cdef signed int ISG, IFR
        cdef double FREFO2, VMAX
        cdef np.ndarray[np.float64_t, ndim=2] TAB
        cdef np.ndarray[np.complex128_t, ndim=1] data_to_fft, y
        cdef np.ndarray[np.float64_t, ndim=1] RTAB


        KTABS2 = self.puiss2(KTABS)
        FREFO2 = float(self.FREFON*self.KTABS)/float(KTABS2)
        debug("# fftmax : KTABS2 = {}  &  FREFO2 = {}".format(KTABS2, FREFO2))

        TAB = np.zeros((2,KTABS2)).astype('float64')

        ISG = -1
        IDV = KTABS2
        IPAS = 1

        for i in range(KTABS2):
            TAB[0, i] = np.real(self.ZTABS[i])*self.TWIN[i]
            TAB[1, i] = np.imag(self.ZTABS[i])*self.TWIN[i]

        data_to_fft = np.zeros(KTABS2).astype('complex128')
        for i in range(KTABS2):
            data_to_fft[i] = np.complex(TAB[0,i], TAB[1,i])

        y  = np.fft.fft(data_to_fft)

        RTAB = np.zeros(KTABS2)
        for i in range(KTABS2):
            RTAB[i] = np.sqrt( np.real(y[i])**2 + np.imag(y[i])**2)/IDV

        INDX, VMAX = self.maxx(KTABS2, RTAB)

        if INDX <= KTABS2/2.0:
            IFR = INDX-1
        else:
            IFR = INDX-1-KTABS2
            info("# fftmax | This would give negative index?")

        FR = (IFR+1)*FREFO2*IPAS
        info('IFR = {} , FR = {} , RTAB[INDX] = {}'.format(IFR, FR, RTAB[INDX]))

        return FR
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef frefin(self, unsigned int KTABS, double FR, double RPASS0, double RPREC):
        '''Refined search for the frequency

        Input: KTABS : Number of turns
               FR    : Frequency already found
               RPASS0: Step
               RPREC : Precision
        Returns: FR : Frequency
                 A  : Real part of the amplitude
                 B  : Imaginary part of the amplitude
                 RM : Amplitude
        '''

        cdef double X, PASS, EPS, nFR, A, B, RM, XM, YM
        cdef unsigned int KTABSF, IPRT



        X      = FR           # this is the FR from fftmax()
        PASS   = RPASS0       # this is the STAREP passed by MFTNAF
        EPS    = RPREC        # this is the EPS passed by MFTNAF
        KTABSF = KTABS

        IPRT = 1
        XM, YM = self.maxiqua(X, PASS, EPS, IPRT)

        debug("# frefin : Maxiqua returned XM = {}, YM = {}".format(XM, YM))
        nFR = XM
        A  = self.AF
        B  = self.BF
        RM = YM

        debug("# frefin returns to MFTNAF : FR = {}, A = {}, B = {}, RM = {}".format(nFR, A, B, RM))

        return nFR, A, B, RM
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef maxiqua(self, double X, double PASS, double EPS, unsigned int IPRT):#,object NFPRT):
        '''Calculation by quadric interpolation of the maximum of a function FUNC given by the user
        Inputs: X, PASS : the max is supposed to be within X-PASS, X+PASS
                EPS     : Precision with which we will calculate the maximum
                XM      : Absica of the maximum
                YM      : YM = FUNC(XM)
                IPRT    : 1/0 Print do not print in file...
                NFPRT   : File to print

        Returns: XM, YM
        '''
        cdef double PAS, EPSI, X2, Y2, X1, X3, Y1, Y3, R2, R3, A, B, XX2


        EPSI = np.float64(1.0e-15)
        PAS = PASS
        X2 = X
        Y2 = self.func(X2)
        X1 = X2 - PAS
        X3 = X2 + PAS
        Y1 = self.func(X1)
        Y3 = self.func(X3)

        debug("# MAXIQUA : X2 = {}  | Y2 = {}, X1 = {} , X3 = {}, Y1 = {}, Y3 = {}".format(X2,Y2,X1,X3,Y1,Y3))

        while True:
            if PAS >= EPSI:
            #    if IPRT == 1:
                debug("# maxiqua : {} {} {} {} {} ".format(PAS, X2, Y1, Y2, Y3))
                if np.abs(Y3-Y1) < EPSI:
                    debug("# MAXIQUA : ERROR Y3-Y1 < EPSI")
                    break
                if (Y1<Y2) and (Y3<Y2):
                    debug("# maxiqua :  maxiif (Y1<Y2) and (Y3<Y2)")
                    R2  = (Y1-Y2)/(X1-X2)
                    R3  = (Y1-Y3)/(X1-X3)
                    A   = (R2 - R3)/(X2-X3)
                    B   = R2 - A*(X1+X2)
                    XX2 = -B/(2.0*A)
                    PAS = np.abs(XX2-X2)

                    if XX2 > X2:
                        X1 = X2
                        Y1 = Y2
                        X2 = XX2
                        Y2 = self.func(X2)
                        X3 = X2 + PAS
                        Y3 = self.func(X3)
                    else:
                        X3 = X2
                        Y3 = Y2
                        X2 = XX2
                        Y2 = self.func(X2)
                        X1 = X2 - PAS
                        Y1 = self.func(X1)
                else:
                    if Y1>Y3:
                        X2 = X1
                        Y2 = Y1
                    else:
                        X2 = X3
                        Y2 = Y3

                    X1 = X2 - PAS
                    X3 = X2 + PAS
                    Y1 = self.func(X1)
                    Y3 = self.func(X3)
            else:
                break
        XM = X2
        YM = Y2

        info("# maxiqua : PAS = {} | XM = {}, YM = {}".format(PAS, XM, YM))

        return XM, YM
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef double func(self, double X):
        '''A function used for the interpolation
        Input   :   X : Frequency
        Returns : RMD : Amplitude magnitude
        '''
        cdef double RMD, A, B

        RMD, A, B = self.profre(self.KTABS, X)
        self.AF = A
        self.BF = B
        debug("# func X = {} | A = {}, B = {}, RMD = {}".format(X,A,B,RMD))
        return RMD
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef profre(self, unsigned int KTABS, double FS):
        '''Calculates the exponential scalar product of (-I * FS)*t by TAB(0:7)
        using the Hardy Integration method
        Input  : KTABS : Number of turns
                 FS    : Frequency
        Returns: RMD : Magnitude of amplitude
                 A   : Real part of amplitude
                 B   : Imaginary part of amplitude
        @NB : FS is given in seconds per year
        '''
        cdef double OM, ANG0, ANGI, H, A, B, RMD
        cdef np.complex128_t ZI, ZAC, ZINC, ZEX
        cdef unsigned int LTF
        cdef np.ndarray[np.complex128_t, ndim=1] ZT, ZTF
        cdef np.complex128_t ZA

        OM = np.float64(FS)/np.float64(self.UNIANG)
        ZI = np.complex128(np.complex(0.0, 1.0))

        LTF = KTABS
        ANG0 = OM*self.T0
        ANGI = OM*self.XH
        ZAC  = np.exp(-ZI*ANG0)
        ZINC = np.exp(-ZI*ANGI)
        ZEX  = ZAC/ZINC

        ZT, ZTF = self.ztpow2(int(self.KTABS+1), 64, np.asarray(self.ZTABS), np.asarray(self.TWIN), ZINC, ZEX)

        # length of the step
        H = np.float64(1.0)/LTF
        ZA = self.zardyd(ZTF, LTF+1, H)
        A = np.real(ZA)
        B = np.imag(ZA)
        RMD = np.abs(ZA)
        return RMD, A, B
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef ztpow2(self, unsigned int N, unsigned int N1, np.ndarray[np.complex128_t, ndim=1] ZTA, np.ndarray[np.float64_t, ndim=1] TW, np.complex128_t ZA, np.complex128_t ZAST):
        '''Calculate : ZTF(i) = ZTA(i)* TW(i)*ZAST*ZA**i in vector form.
        Inputs : N  : KTABS + 1
                 N1 : 64 (hardcoded)
                 ZTA: array of complex data
                 TW : array of the window
                 ZA : exponential
                 ZAST: exponential
        Returns: ZT : complex number
                 ZTF: complex array
        '''
        cdef np.ndarray[np.complex128_t, ndim=1] ZT, ZTF
        cdef unsigned int i, I, NT, INC, NX

        if N<1:
            raise ValueError("# PyNAFF:PyNAFF:ztpow2 : Error the input size N [{}] is less than minimum required. [1]".format(N))

        ZT = np.zeros(N1).astype('complex128')
        ZTF = np.zeros(N).astype('complex128')

        ZT[0] = ZAST*ZA
        for i in xrange(1,N1):
            ZT[i] = ZT[i-1]*ZA

        for i in xrange(0,N1):
            ZTF[i] = ZTA[i]*TW[i]*ZT[i]

        ZT1 = ZT[N1-1]/ZAST
        ZINC = 1
        INC = 0
        NT = int(N)/int(N1)
        for IT in xrange(1, NT):
            ZINC = ZINC*ZT1
            INC  = INC+N1

            for I in xrange(0,N1):
                ZTF[INC+I] = ZTA[INC+I]*TW[INC+I]*ZT[I]*ZINC

        ZINC = ZINC*ZT1
        INC  = INC + N1
        NX   = N-NT*N1

        for I in xrange(0,NX):
            ZTF[INC+I] = ZTA[INC+I]*TW[INC+I]*ZT[I]*ZINC

        return ZT, ZTF
# # - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cpdef np.ndarray[np.complex128_t, ndim=1] ztpow(self, unsigned int N, unsigned int N1, np.complex128_t ZA, np.complex128_t ZAST):
        '''Calculate the ZT[i] = ZAST * ZA^{i} in the vector ZT[N]
        Inputs : N  : KTABS + 1
                 N1 : 64 (hardcoded)
                 ZA : exponential
                 ZAST: exponential
        Returns: ZT : complex number
        '''

        cdef np.ndarray[np.complex128_t, ndim=1] ZT
        cdef unsigned int i, I, NT, INC, NX

        ZT = np.zeros(N).astype('complex128')

        if N<1:
            raise ValueError("# PyNAFF:PyNAFF:ztpow : The input size N [{}] is less than minimum required. [1]".format(N))

        ZT[0] = ZAST*ZA
        for i in xrange(1,N1):
            ZT[i] = ZT[i-1]*ZA

        ZT1 = ZT[N1-1]/ZAST
        ZINC = 1
        INC  = 0
        NT = int(N)/int(N1)
        for i in xrange(1,NT):
            ZINC = ZINC*ZT1
            INC  = INC + N1
            for i in xrange(0,N1):
                ZT[INC+i] = ZT[i]*ZINC

        ZINC = ZINC*ZT1
        INC  = INC + N1
        NX   = int(N)-int(NT)*int(N1)
        for I in xrange(0, NX):
            ZT[INC+I] = ZT[I]*ZINC
        return ZT
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef ztpow2a(self, unsigned int N, unsigned int N1, np.ndarray[np.float64_t, ndim=1] TW, np.complex128_t ZA, np.complex128_t ZAST):
        '''Calculate the ZTF(i) = TW(i)*ZAST*ZA**i in the vector ZT[N]
        Inputs : N  : KTABS + 1
                 N1 : 64 (hardcoded)
                 TW : array of the window
                 ZA : exponential
                 ZAST: exponential
        Returns: ZT : complex number
                 ZTF: complex array
        '''
        cdef np.ndarray[np.complex128_t, ndim=1] ZT, ZTF
        cdef unsigned int i, I, NT, INC, NX


        ZT = np.zeros(N1).astype('complex128')
        ZTF = np.zeros(N).astype('complex128')
        if N < 1 :
            raise ValueError("# PyNAFF:PyNAFF:ztpow2a : The input size N [{}] is less than minimum required. [1]".format(N))

        ZT[0] = ZAST*ZA
        debug('# ztpow2a : ZT[0] = {}'.format(ZT[0]))
        for i in range(1,N1):
            ZT[i] = ZT[i-1]*ZA

        for i in range(0,N1):
            ZTF[i] = TW[i]*ZT[i]

        ZT1 = ZT[N1-1]/ZAST
        ZINC = 1
        INC = 0
        NT = int(N)/int(N1)

        debug('# ztpow2a : ZT1 = {} | NT = {}'.format(ZT1, NT))

        for IT in xrange(1,NT):
            ZINC = ZINC*ZT1
            INC  = INC + N1
            for I in xrange(0,N1):
                ZTF[INC+I] = TW[INC+I]*ZT[I]*ZINC

        ZINC = ZINC*ZT1
        INC = INC + N1
        NX = N-NT*N1
        for I in xrange(0,NX):
            ZTF[INC+I] = TW[INC+I]*ZT[I]*ZINC

        return ZT, ZTF
# # - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef np.complex128_t zardyd(self, np.ndarray[np.complex128_t, ndim=1] ZT, unsigned int N, double H):
        '''Calculate the integral of a function tatulated by Hardy's method
        T[N] is the table of the values of the function

        Inputs : ZT  : complex array of the exponentials
                 N   : 6*K+1
                 H   : not between two values
        Returns: ZOM : value of the integral in the interval [X1, XN]
        '''

        cdef unsigned int ITEST, K
        cdef np.complex128_t ZOM

        ITEST = np.mod(N,6)
        if ITEST != 1 :
            raise ValueError("# PyNAFF:PyNAFF:zardyd : The number of turns required [{}] is not a multiple of 6.".format(N-1))

        K = (int(N)-1)/6
        ZOM = 41.*ZT[0]+216.*ZT[1]+27.*ZT[2]+272.*ZT[3]+27.*ZT[4]+216.*ZT[5]+41.*ZT[int(N)-1]

        for I in xrange(1, K):
            ZOM=ZOM+82.0*ZT[6*I+1-1]+216.0*ZT[6*I+2-1]+27.0*ZT[6*I+3-1]+272.0*ZT[6*I+4-1]+27.0*ZT[6*I+5-1]+216.0*ZT[6*I+6-1]

        ZOM=ZOM*H*6.0/840.0

        return ZOM
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef fretes(self, double FR, double TOL):
        '''Tests the new frequency found compared to the old ones.
        The distance between frequencies must be FREFON

        Input   : FR  : Frequency to be tested
                  TOL : tolerance (i.e. 1.0e-7 is ok)

        Returns : IFLAG = 1  : if the test succeeds (you can continue)
                  IFLAG = 0  : if the test fails (it is better to stop)
                  IFLAG = -1 : if TEST<START but TEST/START < TOL
                                (practically we find the same frequency index NFR)
                  NUMFR      : Frequency index found.
        '''
        cdef signed int NUMFR, IFLAG
        cdef double ECART, TEST
        cdef unsigned int i

        NUMFR = 0
        IFLAG = 1
        ECART = np.abs(self.FREFON)

        debug("# fretes : ECART =  {} , NFS= {}, FR = {}".format(ECART, self.NFS, FR))

        for i in range(self.NFS):
            TEST = np.abs(self.TFS[i] - FR)
            if TEST < ECART:
                debug("# fretes : i = {} | TEST = {} | TOL = {}".format(i, TEST, TOL))
                if TEST/ECART < TOL :
                    IFLAG = - 1
                    NUMFR = i
                    debug("# fretes : TEST/ECART = {} --> We continue.".format(TEST/ECART))
                    continue
                else:
                    IFLAG = 0
                    debug("# fretes : TEST = {} , ECART = {}".format(TEST, ECART))
                    debug("# fretes : Frequency FR = {}  is to close to {}".format(FR, self.TFS[i]))
                    continue

        debug("# fretes : Returning IFLAG = {} | NUMFR = {}".format(IFLAG, NUMFR))
        return IFLAG, NUMFR
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef np.complex128_t proscaa(self, unsigned int KTABS, double F1, double F2): # ZP is the return value
        '''Calculate the scalar product of exp(I*F1*t) with exp(-I*F... )
        on the interval [0:KTABS],  T = T0+XH*IT
        Analytical calculateion

        Input:   KTABS : Number of turns
                 F1    : 1st frequency
                 F2    : 2nd frequency
        Returns: ZP    : complex scalar product

        '''
        cdef np.complex128_t ZI, ZAC, ZINC, ZEX, ZP
        cdef np.ndarray[np.complex128_t, ndim=1] ZT, ZTF
        cdef double OM, ANG0, ANGI, H
        cdef unsigned int LTF


        ZI = np.complex128(np.complex(0.0, 1.0))
        debug("# proscaa : Inputs KTABS={}, F1={}, F2={}".format(KTABS, F1,F2))

        OM = (F1-F2)/self.UNIANG
        LTF = KTABS
        ANG0 = OM*self.T0
        ANGI = OM*self.XH
        debug("# proscaa : ANG0 = {}, ANGI = {}".format(ANG0,ANGI))

        ZAC   = np.exp(-ZI*ANG0)
        ZINC  = np.exp(-ZI*ANGI)
        ZEX = ZAC/ZINC
        debug("# proscaa : ZEX = {}".format(ZEX))
        ZT, ZTF = self.ztpow2a(KTABS+1, 64,  np.asarray(self.TWIN), ZINC, ZEX)
        ## STEP SIZE
        H = np.float64(1.0)/LTF

        ZP = self.zardyd(ZTF, LTF+1, H)
        debug("# proscaa: Zardyd returned ZP = {}".format(ZP))
        return ZP
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void gramsc(self, unsigned int KTABS, double FS, double A, double B):
        '''Gram-Schmidt : Calculate the ZTEE[I] = <EN, EI>
        Input: KTABS : Number of turns
                FS   : Frequency
                A, B : real and imaginary part of the amplitude

        Returns: None (it modifies the input data)
        '''

        cdef np.ndarray[np.complex128_t, ndim=1] ZTEE, ZT
        # cdef np.ndarray[np.complex128_t, ndim=2] ZA
        cdef np.complex128_t ZALP
        cdef unsigned int i, NF, k, j, IT
        cdef double DIV
        cdef np.complex128_t ZDIV, ZMUL, ZI, ZOM, ZEX, ZINC


        ZTEE = np.zeros(self.NFR).astype('complex128')

        debug("# gramsc : The inputs are : KTABS = {}, FS = {}, A={}, B={}, NFS={}".format(KTABS, FS, A, B, self.NFS))

        for i in range(0, self.NFS):
            debug('#gramsc : running proscaa {}'.format(i))
            ZTEE[i] = self.proscaa(KTABS, FS, self.TFS[i])

        NF = self.NFS + 1

        ZTEE[NF-1] = np.complex(1.,0.)
        debug("# gramsc : Setting NF=NFS+1 = {}, NFS = {}".format(NF, self.NFS))

        self.TFS[NF-1] = FS

        for k in xrange(1,self.NFS+1):
            for i in xrange(1, self.NFS+1):
                for j in xrange(1,i+1):
                    self.ZALP[NF-1, k-1] = self.ZALP[NF-1, k-1] - np.conj(self.ZALP[i-1,j-1])*self.ZALP[i-1,k-1]*ZTEE[j-1]


        self.ZALP[NF-1,NF-1] = np.complex(1.0, 0.0)

        # we make the norm of FN = 1
        DIV = np.float64(1.0)
        ZDIV = np.complex128(np.complex(0., 0.))
        for i in xrange(0,NF):
            ZDIV = ZDIV + np.conj(self.ZALP[NF-1,i])*ZTEE[i]

        DIV = np.sqrt(np.abs(ZDIV))

        debug("# gramsc : ZDIV = {} , DIV = {}".format(ZDIV, DIV))
        for i in xrange(0,NF):
            self.ZALP[NF-1, i] = self.ZALP[NF-1, i]/DIV

        # F1, F2, ... FN is the orhonormal basis
        # We remove now a F <F, FN>FN (<F,FN> = <F, EN>)
        ZMUL = np.complex128(np.complex(A,B))/DIV
        debug("# gramsc : ZMUL = {}".format(ZMUL))
        ZI = np.complex128(np.complex(0.0, 1.0))

        debug("# gramsc : NF = {}, len(TFS) = {}, len(ZALP) = {}, len(ZAMP)={}, ".format(NF, len(self.TFS), len(self.ZALP), len(self.ZAMP)))
        for i in xrange(0,NF):
            
            ZOM = self.TFS[i]/self.UNIANG*ZI
            ZA = self.ZALP[NF-1,i]*ZMUL

            # The amplitudes of the terms are corrected
            # Attention here (real case) we will also have the conjugate term which we do not calculate
            # the total term is #  2* Re(ZAMP[I] * Exp(ZI*TFS[I]*T))

            self.ZAMP[i] = self.ZAMP[i]+ZA

            info("# GRAMSC : Should print in file : {} {} {} {} {} {}".format(i, self.TFS[i], np.abs(self.ZAMP[i]), np.real(self.ZAMP[i]), np.imag(self.ZAMP[i]), np.arctan2(np.imag(self.ZAMP[i]), np.real(self.ZAMP[i]))   ) )

            # we retire the contribution of the term TFS[I] in the TABS
            if self.ICPLX == 1:

                ZEX = ZA * np.exp(ZOM*(self.T0-self.XH))
                ZINC = np.exp(ZOM*self.XH)
                ZT = self.ztpow(KTABS+1, 64, ZINC, ZEX)

                debug("# GRAMSC : inside ICPLX, ZOM = {}, ZA = {} ".format(ZOM, ZA))
                debug("# GRAMSC : inside ICPLX, ZINC = {}, ZEX = {} ".format(ZINC, ZEX))

                for IT in xrange(0,int(self.KTABS)+1):
                    self.ZTABS[IT] = self.ZTABS[IT]-ZT[IT]

            else:
                ZEX  = ZA*np.exp(ZOM*(self.T0-self.XH))
                ZINC = np.exp(ZOM*self.XH)
                ZT   = self.ztpow(KTABS+1, 64, ZINC, ZEX)
                for IT in xrange(0, int(self.KTABS)+1):
                    self.ZTABS[IT] = self.ZTABS[IT]-ZT[IT]
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void modfre(self, unsigned int KTABS, unsigned int NUMFR, double A, double B ): #(KTABS,NUMFR,A,B)
        '''Allows you to modify an amplitude already calculated when you find the same term. A TOL is present

        Input:  KTABS  : Number of turns
                NUMFR  : number of the frequency already existent
                A,B    : real and imaginary parts of the amplitude of the modification,
                            to put to the amplitude of the frequency NUMFR
        Returns: None
        '''

        cdef np.complex128_t ZI, ZOM, ZA, ZEX, ZINC
        cdef np.ndarray[np.complex128_t, ndim=1] ZT
        cdef int IT

        ZI = np.complex128(np.complex(0.0, 1.0))
        ZOM = self.TFS[NUMFR]/self.UNIANG*ZI
        ZA = np.complex128(np.complex(A,B))

        debug("# modfre : Recieved input for KTABS = {} | NUMFR = {} | A = {} | B = {} ".format(KTABS, NUMFR, A, B))()
        info("# modfre : Correction of IFR = {} with Amplitude = {} ".format(NUMFR, np.abs(ZA)))


        # The amplitudes of the terms are corrected.
        # ATTENTION!! Here (real case) we should also have the conjugate term which we do not calculate
        # The total term is :  2.0 * np.real(ZAMP[I]*np.exp(ZI*TFS[I]*T))
        self.ZAMP[NUMFR] = self.ZAMP[NUMFR] + ZA

        info("# modfre : This have to be written in file | TFS[NUMFR] = {}, ABS(ZAMP[NUMFR])={}, Re(ZAMP[NUMFR])={}, Im(ZAMP[NUMFR])={}, Arctan2(Im,Re) ={}".format(self.TFS[NUMFR], np.abs(self.ZAMP[NUMFR]),np.real(self.ZAMP[NUMFR]), np.imag(self.ZAMP[NUMFR]) ,np.arctan2(np.imag(self.ZAMP[NUMFR]), np.real(self.ZAMP[NUMFR]))))

        # REMOVE THE CONTRIBUTION OF THE TERM TFS[NUMFR] FROM TABS
        if self.ICPLX == 1 :
            ZEX  = ZA*np.exp(ZOM*(self.T0-self.XH))
            ZINC = np.exp(ZOM*self.XH)
            ZT  = self.ztpow(KTABS+1, 64, ZINC, ZEX)
            debug("# modfre : in icplx : ZEX = {}, ZINC = {}, ZT = {}".format(ZEX, ZINC, ZT))
            for IT in xrange(KTABS+1):
                self.ZTABS[IT] = self.ZTABS[IT]-ZT[IT]

        else:
            ZEX  = ZA*np.exp(ZOM*(self.T0-self.XH))
            ZINC = np.exp(ZOM*XH)
            ZT = self.ztpow(KTABS+1,64, ZINC, ZEX)
            for IT in xrange(KTABS+1):
                self.ZTABS[IT] = self.ZTABS[IT]-np.real(ZT[IT])

# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void mftnaf(self, unsigned int KTABS, unsigned int NTERM, double EPS):
        '''Main driving routine for NAFF
        Inputs : KTABS : Number of turns
                 NTERM : Number of frequencies wanted
                 EPS   : Precision
        Returns: None
        '''

        cdef double TOL, STAREP, fr, A, B, RM
        cdef signed int IFLAG
        cdef unsigned int term, NUMFR

        TOL = np.float64(1.0e-4)
        STAREP = np.abs(self.FREFON)/3.0
        self.inifre()
        if self.debug:
            self.printConfig()

        IFLAG = 90
        ##--- now perform the loop:
        for term in xrange(int(NTERM)):
            info('# mftnaf =============== RUNNING MFTNAF FOR TERM {} ================ '.format(NTERM))
            fr = self.fftmax(KTABS) # Fast Fourier Analysis - Search for a maximum of the spectrum from a frequency F
            debug("# mftnaf : FFTMAX just finished and found freq = {}".format(fr))

            fr, A, B, RM  = self.frefin(KTABS, fr, STAREP, EPS)
            debug("# mftnaf : frefin just finished and returned : fr = {}, A={}, B={}, RM ={}".format(fr, A, B, RM))


            IFLAG, NUMFR = self.fretes(fr, TOL)
            debug("# mftnaf : fretes just finished returned : IFLAG = {}, NUMFR = {}".format(IFLAG, NUMFR))


            if IFLAG == 0:
                continue
            elif IFLAG == 1:
                debug('# mftnaf : Because IFLAG=0: NFS before Gramsc = {}'.format(self.NFS))

                info('# mftnaf : Running Gram-Schmidt')
                self.gramsc(KTABS, fr, A, B)

                debug('# mftnaf : Because IFLAG=0: NFS after Gramsc & before Increment = {}'.format(self.NFS))
                self.NFS = self.NFS+1
                debug('# mftnaf : Because IFLAG=0: NFS after Increment = {}'.format(self.NFS))
            elif IFLAG == -1:
                self.modfre(KTABS, NUMFR, A, B)
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void savsol(self, str NFSOLFILENAME):
        '''Save the solution found by NAFF. According to the initialization a file may not be written
        but rather print the result to the STDOUT. Or in case of numpy array data the results
        are appended to self.result

        Input : NFSOLFILENAME : The name of the file to be saved.
        Returns : None
        '''
        cdef unsigned int i 
        cdef double AMP

        if self.isFile:
            if self.saveFiles:
                fsol = open(NFSOLFILENAME, 'aw')
                fsol.write("%s\n" % self.UNIANG)
                fsol.write("%s\n" % self.NFS)
                for i in xrange(self.NFS):
                    AMP = np.abs(self.ZAMP[i])
                    fsol.write("%s\t%s\t%s\t%s\t%s\n" % (i, self.TFS[i], AMP, np.real(self.ZAMP[i]), np.imag(self.ZAMP[i])))
                fsol.close()
            else:
                for i in xrange(self.NFS):
                    AMP = np.abs(self.ZAMP[i])
                    warn("Solution ICX={} : {} {} {} {} {} ".format(self.ICX, str(i), str(self.TFS[i]), str(AMP), str(np.real(self.ZAMP[i])), str(np.imag(self.ZAMP[i]))))
        else:
                for i in xrange(self.NFS):
                    AMP = np.abs(self.ZAMP[i])
                    self.result.append((int(self.ICX), int(i), self.TFS[i], AMP, np.real(self.ZAMP[i]), np.imag(self.ZAMP[i])) )
                    #np.append(self.result,  np.array([int(self.ICX), int(i), self.TFS[i], AMP, np.real(self.ZAMP[i]), np.imag(self.ZAMP[i])]))
                    warn("Solution ICX = {}: {} {} {} {} {} ".format(int(self.ICX), int(i), self.TFS[i], AMP, np.real(self.ZAMP[i]), np.imag(self.ZAMP[i])))
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void appendToSol(self, str NFSOLFILENAME, str mystr):
        '''Function to append a string to the solution file.
        Inputs : NFSOLFILENAME : filename
                 mystr         : string to be appended.
        Returns: None
        '''
        fsol = open(NFSOLFILENAME, 'aw')
        fsol.write(mystr+"\n")
        fsol.close()
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef tessol(self, unsigned int KTABS, double EPS):  # no idea what this one does
        '''Subprogram to verify the accuracy of the solutions obtained by fourrier analysis.
        We analyse again the solution and we compare the terms.

        Inputs: KTABS : Number of turns
                EPS   : Precision
        Returns: TFSR : Frequencies
                 ZAMPR: Complex amplitudes
        '''

        cdef np.ndarray[np.float64_t, ndim=1] TFST, TFSR
        cdef np.ndarray[np.complex128_t, ndim=1] ZAMPT, ZAMPR, ZT
        cdef np.ndarray[np.complex128_t, ndim=2] ZALPT 
        cdef double OFFSET
        cdef np.complex128_t ZI, ZOM, ZA, ZEX, ZINC
        cdef unsigned int IFR, IT


        TFST = np.zeros(self.NFR).astype('float64')
        ZAMPT = np.zeros(self.NFR).astype('complex128')
        ZALPT = np.zeros((self.NFR, self.NFR)).astype('complex128')
        TFSR  = np.zeros(self.NFR).astype('float64')
        ZAMPR = np.zeros(self.NFR).astype('complex128')

        OFFSET = KTABS*self.XH

        # initialize tabs
        debug("# tessol : Initialized self.ZTABS to 0+0j")
        self.ZTABS = np.zeros(KTABS+1).astype('complex128')

        # calculate the new solution
        ZI = np.complex128(np.complex(0.0, 1.0))
        for IFR in range(self.NFS):
            ZOM = self.TFS[IFR]/self.UNIANG*ZI
            ZA  = self.ZAMP[IFR]

            if self.ICPLX == 1 :
                ZEX  = ZA * np.exp((self.T0+OFFSET-self.XH)*ZOM)
                ZINC = np.exp(self.XH*ZOM)
                ZT   = self.ztpow(KTABS+1, 64, ZINC, ZEX)

                for IT in range(0,KTABS+1):
                    self.ZTABS[IT] = self.ZTABS[IT]+ZT[IT]
            else:
                ZEX  = ZA * np.exp((self.T0+OFFSET-self.XH)*ZOM)
                ZINC = np.exp(XH*ZOM)
                ZT   = self.ztpow(KTABS+1, 64, ZINC, ZEX)
                for IT in range(0, KTABS+1):
                    self.ZTABS[IT] = self.ZTABS[IT]+np.real(ZT[IT])

        TFST[:self.NFS] = self.TFS[:self.NFS]
        ZAMPT[:self.NFS] = self.ZAMP[:self.NFS]
        ZALPT[:self.NFS,:self.NFS] = self.ZALP[:self.NFS, :self.NFS]

        # new caclulation of the solutions
        self.NTERM = self.NFS
        self.mftnaf(self.KTABS, self.NTERM, self.EPS)

        # results
        TFSR[:self.NFS] = self.TFS[:self.NFS]
        ZAMPR[:self.NFS] = self.ZAMP[:self.NFS]

        # removal of the solution
        self.TFS[:self.NFS] = TFST[:self.NFS]
        self.ZAMP[:self.NFS] = ZAMPT[:self.NFS]

        self.ZALP[:self.NFS, :self.NFS] = ZALPT[:self.NFS, :self.NFS]

        return TFSR, ZAMPR
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void compso(self, str NFOUTFILE, unsigned int NFS, np.ndarray[np.float64_t, ndim=1] TFSA, np.ndarray[np.complex128_t, ndim=1] ZAMPA, np.ndarray[np.float64_t, ndim=1] TFSB, np.ndarray[np.complex128_t, ndim=1] ZAMPB):
        '''This function compares the two solutions
        Input:  NFOUTFILE   :   File to write the result
                NFS         :   Number of frequencies
                TFSA        :   Table of frequencies of the first solution
                ZAMPA       :   Table of complex amplitudes of the first solution
                TFSB        :   Table of frequencies of the second solution
                ZAMPB       :   Table of complex amplitudes of the second solution
        Returns: None
        '''
        cdef double RO
        cdef unsigned int IFR

        if NFOUTFILE == "STDOUT":
            info('# compso : TESSOL: Comparison , NFS = {}'.format(NFS))
            for IFR in xrange(NFS):
                RO = np.sqrt(np.real(ZAMPA[IFR])**2 + np.imag(ZAMPA[IFR])**2)
                info("# compso : {} {} {} {} {}".format(IFR, TFSA[IFR], TFSB[IFR], TFSA[IFR]-TFSB[IFR], RO) )
        else:
            fout = open(NFOUTFILE, 'aw')
            fout.write('TESSOL: Comparison , NFS = %s\n'% NFS)
            for IFR in xrange(NFS):
                RO = np.sqrt(np.real(ZAMPA[IFR])**2 + np.imag(ZAMPA[IFR])**2)
                fout.write("%s\t%s\t%s\t%s\t%s\n" % (IFR, TFSA[IFR], TFSB[IFR], TFSA[IFR]-TFSB[IFR], RO) )
            fout.close()
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cdef void savtab(self, unsigned int KTABS, np.ndarray[np.complex128_t, ndim=1] ZTABS, str nomftab):
        '''Save the tabs in a file.
        Inputs : KTABS   : Number of turns
                 ZTABS   : Complex data
                 nomftab : String filename of the tab file
        Returns: None
        '''
        cdef unsigned int i
        ftab = open(nomftab, 'aw')
        for i in xrange(KTABS+1):
            ftab.write("%s\t%s\n" % (np.real(ZTABS[i]), np.imag(ZTABS[i])))
        ftab.close()
# - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - - * - - -
    cpdef run(self):
        '''Driving routine for the full PyNAFF class
        '''
        #STEP 1) Load the initial parametrisation & print it
        self.inipar()
        if self.debug:
            self.printConfig()

        # print the parametrisation to a file
        if self.saveFiles:
            self.prtpar()

        # STEP 2) Additional parametrisation of UNIANG, EPS , IKT and FREFON
        self.initAngleStep()

        # STEP 3) Initialize the Window
        self.iniwin(self.KTABS, window=self.IW)

        # STEP 4) Load the data
        debug("# run : Loading data from file [{}]".format(self.dataFilename))
        if self.isFile:
            self.loadsol(self.KTABS, self.ICX, self.ICY, self.ICOL, self.NULIN, filename=self.dataFilename)
        else:
            # @TODO clean up this input
            self.loadArray(self.KTABS, self.ICX, self.ICY, self.ICOL, self.NULIN, data=np.asarray(self.dataArray))

        # Print the tables
        if self.saveFiles:
            self.prtabs(self.KTABS, self.ZTABS, self.IKT, where=self.tabFilename)
        else:
            self.prtabs(self.KTABS, self.ZTABS, self.IKT, where=0)

        # STEP 5) Run the NAFF routine
        self.IMODT = 0
        self.mftnaf(self.KTABS, self.NTERM, self.EPS)

        # Print the tables
        if self.debug:
            self.prtabs(self.KTABS, self.ZTABS, self.IKT)

        # STEP 6) Save the solutions
        self.savsol(self.solutionFilename)

        # STEP 7) Do a comparison
        if self.IFCOMP == 1:
            info('# run : Running Comparison...')
            # Save the current table
            self.ZRTABS = self.ZTABS
            self.IMODT = 0
            TFSR , ZAMPR = self.tessol(self.KTABS, self.EPS)
            if self.saveFiles:
                self.compso(self.outputFilename, self.NFS, self.TFS, self.ZAMP, TFSR, ZAMPR)
            else:
                self.compso("STDOUT", self.NFS, self.TFS, self.ZAMP, TFSR, ZAMPR)
            self.ZTABS[0:KTABS+1] = ZRTABS[0:KTABS+1]

        # STEP 8) Do stuff for the residuals
        if self.IFRES == 1:
            self.prtabs(self.KTABS, self.ZTABS, self.IKT)
            self.mftnaf(self.KTABS, self.NTERM, self.EPS)
            self.prtabs(self.KTABS, self.ZTABS, self.IKT)
            if saveFiles:
                self.appendToSol(self.solutionFilename, "*************")
                self.savsol(self.solutionFilename)

        # STEP 9) Save the data that remain
        if (self.IFSTAB == 1) and (self.saveFiles):
            self.savtab(self.KTABS, np.asarray(self.ZTABS), self.tabFilename)

        if not self.isFile:
            return np.array(self.result)

        info("---- DONE ---- ")


# ###################################     END OF CLASS PYNAFF      ###################################
#
#
#
##############################    D R I V I N G   R O U T I N E S    ##############################

cdef void naff_file(str data='track.obs0001.p0001', unsigned int turns=300, unsigned int nterm=1, unsigned int icol=1, unsigned int icx=1, signed int icy=0, bool saveFiles=False, 
        bool doAll=False, double neps=100000000., unsigned int loglevel=20, str logfile=None, bool debug=False, bool batch=True, unsigned int icplx=1, signed int iw=+1,
        unsigned int nulin=0, unsigned int iprnaf=0, unsigned int iftes=0, unsigned int ifsav=0, unsigned int ifcomp=0, unsigned int ifres=0, unsigned int ifstab=1,
        unsigned int imodt=0, str inputParamFile=None, str solutionFile='track.sol', str parametersOutLog='nafpar.par', str outBuffFile='naf.out', str tabFile='naf.tab'):
    ''' Function to run the naff on data taken from file.
    Made modular to allow higher dimensional data
    '''

    if batch:
        loglevel  = 50
        saveFiles = False
        debug     = False

    n = PyNAFF(KTABS=turns, filename=data, data_array=None, saveFiles=saveFiles, isFile=True, inputParamFile=inputParamFile, solutionFile=solutionFile, parametersOutLog=parametersOutLog,
        outBuffFile=outBuffFile, tabFile=tabFile, batch=batch, debug=debug, logfile=logfile, loglevel=loglevel, neps=neps,
        nterm=nterm, icplx=icplx, iw=iw, nulin=nulin, icol=icol, icx=icx ,icy=icy, iprnaf=iprnaf, iftes=iftes, ifsav=ifsav, ifcomp=ifcomp, ifres=ifres, ifstab=ifstab,imodt=imodt)
    n.run()

# ################################
# not sure about this return type
# cpdef double[:] naff_array(double [:] data, unsigned int turns=300, unsigned int nterm=1, unsigned int icol=1, unsigned int icx=1, 
#          signed int icy=0, bool saveFiles=False, bool doAll=False, double neps=100000000., unsigned int loglevel=20, str logfile=None, bool debug=False, bool batch=True,
#          unsigned int icplx=1, signed int iw=+1, unsigned int nulin=0,  unsigned int iprnaf=0, unsigned int iftes=0, unsigned int ifsav=0, unsigned int ifcomp=0, 
#          unsigned int ifres=0, unsigned int ifstab=1, unsigned int imodt=0, str inputParamFile=None, str solutionFile='track.sol', str parametersOutLog='nafpar.par',
#          str outBuffFile='naf.out', str tabFile='naf.tab'):
def naff_array(data, turns=300, nterm=1, icol=1, icx=1, icy=0, saveFiles=False, doAll=False, neps=100000000., loglevel=20, logfile=None, debug=False, batch=True,
         icplx=1, iw=+1, nulin=0, iprnaf=0, iftes=0, ifsav=0, ifcomp=0, ifres=0, ifstab=1, imodt=0, inputParamFile=None, solutionFile='track.sol', 
         parametersOutLog='nafpar.par', outBuffFile='naf.out', tabFile='naf.tab'):
    ''' Function to run the naff on data taken from a numpy array.
    Made modular to allow higher dimensional data
    '''

    data = data.astype('complex128')

    n = PyNAFF(KTABS=turns, data_array=data, filename=None, saveFiles=saveFiles, isFile=False, inputParamFile=inputParamFile, solutionFile=solutionFile, parametersOutLog=parametersOutLog,
        outBuffFile=outBuffFile, tabFile=tabFile, batch=batch, debug=debug, logfile=logfile, loglevel=loglevel, neps=neps,
        nterm=nterm, icplx=icplx, iw=iw, nulin=nulin, icol=icol, icx=icx ,icy=icy, iprnaf=iprnaf, iftes=iftes, ifsav=ifsav, ifcomp=ifcomp, ifres=ifres, ifstab=ifstab,imodt=imodt)

    return n.run()


# ################################


def naff(data, turns=300, nterm=1, icol=1, icx=1, icy=0, saveFiles=False, doAll=False, neps=100000000., loglevel=20, logfile=None, 
        debug=False, batch=True, icplx=1, iw=+1, nulin=0,  iprnaf=0, iftes=0, ifsav=0, ifcomp=0, ifres=0, ifstab=1, imodt=0, inputParamFile=None, 
        solutionFile='track.sol', parametersOutLog='nafpar.par', outBuffFile='naf.out', tabFile='naf.tab'):
    '''Driving routine for the PyNAFF module. Works with data from text files or numpy arrays.
    @TODO will work for multi-BPM analysis
    '''

    #############################
    #   FILE DATA
    #############################
    if type(data)==str:
        #print("# naff : Working with file [{} : icol={}, icx={}, icy={}] for turns={}".format(data, icol, icx, icy, turns))

        ret = np.ones(1)
        # @TODO remove this restriction:
        # This is a check required for Hardy's
        if np.mod(turns,6) != 0:
            raise ValueError("# PyNAFF.naff() : Number of turns [{}] not a mupliple of 6.".format(turns))

        # @TODO I pass it to a function for future doAll implementation :)
        naff_file(turns=turns, data=data, saveFiles=saveFiles, inputParamFile=inputParamFile, solutionFile=solutionFile, parametersOutLog=parametersOutLog,
                outBuffFile=outBuffFile, tabFile=tabFile, batch=batch, debug=debug, logfile=logfile, loglevel=loglevel, neps=neps, doAll=doAll,
                nterm=nterm, icplx=icplx, iw=iw, nulin=nulin, icol=icol, icx=icx ,icy=icy, iprnaf=iprnaf, iftes=iftes, ifsav=ifsav, ifcomp=ifcomp, ifres=ifres, ifstab=ifstab,imodt=imodt)
        return 1



    # @TODO how bout working with pandas series input?
    #############################
    #   PANDA SERIES
    #############################
    # if type(data) == pandas.core.series.Series:
    #     data = np.array(data)
    #     print("# naff : Working with Pandas Series.. Converting it to numpy array.") #"#[shape= {}, icol={}, icx={}, icy={}, doAllCols={}] for turns={}".format(data.shape, icol, icx, icy, doAll, turns))
    #     pass


    #############################
    #   NUMPY ARRAY
    # #############################
    elif type(data) == np.ndarray:
        # @TODO FIX THIS MESSAGE!!!! ICOL for 1d arrays is wrong!
        #print("# naff : Working with numpy array [shape= {}, icol={}, icx={}, icy={}, doAllCols={}] for turns={}".format(data.shape, icol, icx, icy, doAll, turns))
        isFile = False

        if batch:
            loglevel  = 50
            saveFiles = False
            debug     = False

        # @TODO work with larger shape arrays
        if len(data.shape)  == 1:
            icol = 1
            icx  = 1
            icy  = 0

            # Check if I have enough turns in my data
            if turns+1 > len(data):
                raise ValueError("# PyNAFF.naff() : Number of turns [{}] exceeds the minimum required data length [{}]".format(turns, len(data+1)))

            # @TODO remove this restriction:
            # This is a check required for Hardy's
            if np.mod(turns,6) != 0:
                raise ValueError("# PyNAFF.naff() : Number of turns [{}] not a mupliple of 6.".format(turns))


            # @TODO : make it work for doAll :)
            myresult = naff_array(turns=turns, data=data, saveFiles=saveFiles, inputParamFile=inputParamFile, solutionFile=solutionFile, parametersOutLog=parametersOutLog,
                    outBuffFile=outBuffFile, tabFile=tabFile, batch=batch, debug=debug, logfile=logfile, loglevel=loglevel, neps=neps,
                    nterm=nterm, icplx=icplx, iw=iw, nulin=nulin, icol=icol, icx=icx ,icy=icy, iprnaf=iprnaf, iftes=iftes, ifsav=ifsav, ifcomp=ifcomp, ifres=ifres, ifstab=ifstab,imodt=imodt)


        else:
            raise ValueError("len(data.shape)")

        # TODO : doAll stuff...
        return myresult


# ==========
    # Unknown data type
    else:
        raise TypeError("Unrecognised input data type. Supported are filenames (string) & numpy arrays.")


########################################################################################################

cdef void main():
    """ Main function for file to be ran as script :
    If the list of arguments is larger than 1 then the arguments are needed for initialization.
    Else the default (track.obs) debug file is used.

    Argument list :
    >> python PyNAFF.py filename icol icx icy nulin KTABS nterm iw

    """
    if len(sys.argv) > 1:
        n = PyNAFF(KTABS=int(sys.argv[6]), filename=str(sys.argv[1]),saveFiles=True, isFile=True, inputParamFile=None, solutionFile='track.sol', parametersOutLog='nafpar.par',
            outBuffFile='naf.out', tabFile='naf.tab', batch=False, debug=False, logfile=None, loglevel=10, neps=100000000.,
            nterm=int(sys.argv[7]), icplx=1, iw=int(sys.argv[8]), nulin=int(sys.argv[5]), icol=int(sys.argv[2]), icx=int(sys.argv[3]) ,icy=int(sys.argv[4]), iprnaf=0, iftes=0, ifsav=0, ifcomp=0,
            ifres=0, ifstab=1, imodt=0)
        n.run()
    else:
        n = PyNAFF(KTABS=300, filename='track.obs0001.p0001',saveFiles=True, isFile=True, inputParamFile=None, solutionFile='track.sol', parametersOutLog='nafpar.par',
            outBuffFile='naf.out', tabFile='naf.tab', batch=False, debug=False, logfile=None, loglevel=10, neps=100000000.,
            nterm=1, icplx=1, iw=+1, nulin=8, icol=10, icx=3 ,icy=5, iprnaf=0, iftes=0, ifsav=0, ifcomp=0,
            ifres=0, ifstab=1, imodt=0)
        n.run()


########################################################################################################
########################################################################################################
########################################################################################################

if __name__ == '__main__':
    main()
