import numpy as np
import math
import sys
import pandas as pd
from typing import Tuple
import logging
logging.basicConfig(stream=sys.stdout, format="[%(name)s::%(funcName)s] - %(levelname)s: %(message)s")

class FundamentalFrequencies:
    """NAFF - Numerical Analysis of Fundamental Frequencies
    An instance of the class defines an analysis. 
    """
    def __init__(self, data: np.array, turns: int = 300, nterms: int = 1, skipTurns: int = 0, getFullSpectrum: bool = False, window: int = 1, tol: float = 1e-4, dcWarn: bool = True, autoRun: bool = False) -> None:
        """Constructor of the class. The initial setup of the data to be analyzed are provided here.

        Args:
            data (np.array): The array with the turn-by-turn data
            turns (int, optional): Number of points to consider from the input data. Defaults to 300.
            nterms (int, optional): Maximum number of harmonics to search for in the data sample. Defaults to 1.
            skipTurns (int, optional): Number of observations (data points) to skip from the start of the input iterable. Defaults to 0.
            getFullSpectrum (bool, optional): Defines if a full FFT (True) or an rFFT (False) will be performed. Defaults to False.
            window (int, optional): The order of window to be applied on the input data. Defaults to 1.
            tol (float, optional): Expert setting to increase the acceptance window for the harmonics, as long as `getFullSpectrum=True`.
                                   Higher values should let NAFF recover more frequencies, but the maximum number will always be `nterms`. Defaults to 1e-4.
            dcWarn (bool, optional): Defines whether a warning for removing the DC component will be shown. Defaults to True.
        """
        self.data = data
        self.turns = turns
        self.nterms = nterms
        self.skipTurns = skipTurns
        self.getFullSpectrum = getFullSpectrum 
        self.window = window
        self.tol = tol
        self.dcWarn = dcWarn

        # Initialize Logging
        self.logger = logging.getLogger('PyNAFF')
        self.logger.setLevel('INFO')

        # validate input
        self._validate_input()

        # Initialize Analysis Parameters
        self.NFR  = 100
        self.vars = {
            'NFS' 		: 0,
            'TFS' 		: np.zeros(self.NFR).astype('float64'),
            'ZAMP' 		: np.zeros(self.NFR).astype('complex128'),
            'ZALP'	 	: np.zeros((self.NFR,self.NFR)).astype('complex128'),
            'ZTABS' 	: np.array([]).astype('complex128'),
            'TWIN'  	: np.array([]).astype('float64'),
            }
        
        self.results = None

    def _validate_input(self):
        """Validate the input data and reformats if needed.

        Raises:
            ValueError: Input data must be longer than the turns requested.
            ValueError: Minimum number of turns is 6.
        """
        if self.turns >= len(self.data)+1:
            err_msg = f"Input data must be at least of length turns+1. [Provided : turns = {self.turns} | len(data) = {len(self.data)}]"
            self.logger.fatal(err_msg, exc_info=True)
            raise ValueError(err_msg)
        if self.turns < 6:
            err_msg = "Minimum number of turns is 6. [Provided : turns = {self.turns}]"
            self.logger.fatal(err_msg, exc_info=True)
            raise ValueError(err_msg)
        
        if np.mod(self.turns,6)!=0:
            temp_turns = self.turns
            a,b = divmod(self.turns,6)
            self.turns = int(6*a)
            self.logger.info(f"Number of turns has been modified as a multiple of 6 [Provided : {temp_turns} | Setting to {self.turns}]")

    def getIntegral(self, FR: float, turns: int) -> Tuple[float, float, float]:
        """Calculate the integral using Hardy's method

        Args:
            FR (float): Frequency
            turns (int): Number of turns

        Raises:
            ValueError: The number of turns must be multiple of 6.

        Returns:
            Tuple[float, float, float]: Absolute, real and imaginary components of the result.
        """
        if np.mod(turns, 6)!= 0:
            err_msg = "Turns need to be *6."
            self.logger.fatal(err_msg)
            raise ValueError(err_msg)
        K = int(turns/6)

        i_line = np.linspace(1, turns, num=turns, endpoint=True)
        ZTF_tmp = self.vars['ZTABS'][1:]*self.vars['TWIN'][1:]*np.exp(-2.0*(i_line)*np.pi*1.0j*FR)
        ZTF = np.array(self.vars['ZTABS'][0]*self.vars['TWIN'][0])
        ZTF = np.append(ZTF, ZTF_tmp).ravel()
        N = turns + 1
        ZOM = 41.*ZTF[0]+216.*ZTF[1]+27.*ZTF[2]+272.*ZTF[3]+27.*ZTF[4]+216.*ZTF[5]+41.*ZTF[int(N)-1]
        for I in range(1, K):
            ZOM=ZOM+82.0*ZTF[6*I+1-1]+216.0*ZTF[6*I+2-1]+27.0*ZTF[6*I+3-1]+272.0*ZTF[6*I+4-1]+27.0*ZTF[6*I+5-1]+216.0*ZTF[6*I+6-1]
        ZOM=ZOM*(1.0/turns)*(6.0/840.0)
        A = np.real(ZOM)
        B = np.imag(ZOM)
        RMD = np.abs(ZOM)
        return RMD, A, B

    def frefin(self, turns: int, FR: float, STAREP: float, EPS: float) -> Tuple[float, float, float, float]:
        """Tries to refine the frequency found using slopes & root finding methods

        Args:
            turns (int): Number of turns
            FR (float): Frequncy
            STAREP (float): Normalized factor
            EPS (float): Precision factor

        Returns:
            Tuple[float, float, float, float]: Returns the refined frequency components
        """
        EPSI = 1.0e-15
        X2  = FR
        PAS = STAREP
        Y2, A2, B2  = self.getIntegral(X2, turns)
        X1  = X2 - PAS
        X3  = X2 + PAS
        Y1, A1, B1  = self.getIntegral(X1, turns)
        Y3, A3, B3  = self.getIntegral(X3, turns)
        while True:
            if PAS >=EPS:
                if np.abs(Y3-Y1) < EPSI:
                    break
                if (Y1<Y2) and (Y3<Y2):
                    R2  = (Y1-Y2)/(X1-X2)
                    R3  = (Y1-Y3)/(X1-X3)
                    A   = (R2 - R3)/(X2-X3)
                    B   = R2 - A*(X1+X2)
                    XX2 = -B/(2.0*A)
                    PAS = np.abs(XX2-X2)
                    if XX2 > X2:
                        X1 = X2
                        Y1, A1, B1 = Y2, A2, B2
                        X2 = XX2
                        Y2, A2, B2 = self.getIntegral(X2, turns)
                        X3 = X2 + PAS
                        Y3, A3, B3 = self.getIntegral(X3, turns)
                    else:
                        X3 = X2
                        Y3, A3, B3 = Y2, A2, B2
                        X2 = XX2
                        Y2, A2, B2 = self.getIntegral(X2, turns)
                        X1 = X2 - PAS
                        Y1, A1, B1 = self.getIntegral(X1, turns)
                else:
                    if Y1>Y3:
                        X2 = X1
                        Y2, A2, B2 = Y1, A1, B1
                    else:
                        X2 = X3
                        Y2, A2, B2 = Y3, A3, B3

                    X1 = X2 - PAS
                    X3 = X2 + PAS
                    Y1, A1, B1 = self.getIntegral(X1, turns)
                    Y3, A2, B2 = self.getIntegral(X3, turns)
                    if (Y3-Y1)-(Y3-Y2)==0.0:
                        PAS=PAS+EPS

            else:
                break
        return X2, Y2, A2, B2
    
    def fretes(self, FR: float, FREFON: float) -> Tuple[int, int]:
        """If more than one term found, checks how different they are.

        Args:
            FR (float): Frequency
            FREFON (float): Normalized factor

        Returns:
            Tuple[int, int]: Similarity flags
        """
            
        IFLAG = 1
        NUMFR = 0
        ECART = np.abs(FREFON)
        for i in range(len(self.vars['TFS'])):
            TEST = np.abs(self.vars['TFS'][i] - FR)
            if TEST < ECART:
                if float(TEST)/float(ECART) < self.tol: # tolerance value was 1e-4 in Laskar's original work.
                    IFLAG = -1
                    NUMFR = i
                    break
                else:
                    IFLAG = 0
                    continue
        return IFLAG, NUMFR

    def modfre(self, turns: int, FR: float, NUMFR: int, A: float, B: float) -> None:
        """Modifies found frequencies

        Args:
            turns (int): Number of turns
            FR (float): Frequency
            NUMFR (int): Frequency index
            A (float): Real part of found term
            B (float): Imaginary part of found term
        """

        ZI  = 0. + 1.0j
        ZOM = 1.0j*FR
        ZA  = 1.0*A + 1.0j*B
        if len(self.vars['ZAMP'])<= NUMFR:
            self.vars['ZAMP'][NUMFR] = 0
        self.vars['ZAMP'][NUMFR] = self.vars['ZAMP'][NUMFR] + ZA
        i_line = np.linspace(1, turns, num=turns, endpoint=True)
        ZT_tmp = ZA*np.exp(2.0*(i_line)*np.pi*ZOM)
        ZT     = np.array([ZA])
        ZT     = np.append(ZT, ZT_tmp).ravel()
        ZTABS_tmp = self.vars['ZTABS'] - ZT
        self.vars['ZTABS'] = ZTABS_tmp
    
    def proscaa(self, turns: int, FS: float, FS_OLD: float) -> float:
        """

        Args:
            turns (int): Number of turns
            FS (float): Frequency
            FS_OLD (float): Old Frequency

        Returns:
            float: Updated term
        """
        ZI = 0.0+1.0j
        OM = FS-FS_OLD
        ANGI = 2.0*np.pi*OM
        i_line = np.linspace(1, turns, num=turns, endpoint=True)
        ZT_tmp = np.exp(-2.0*(i_line)*1.0j*np.pi*OM)
        ZT_zero = np.array([1])
        ZT = np.append(ZT_zero, ZT_tmp).ravel()
        ZTF = np.multiply(self.vars['TWIN'],ZT)
        N = turns + 1
        ZOM = 41.*ZTF[0]+216.*ZTF[1]+27.*ZTF[2]+272.*ZTF[3]+27.*ZTF[4]+216.*ZTF[5]+41.*ZTF[int(N)-1]
        for I in range(1, int(turns/6)):
            ZOM=ZOM+82.0*ZTF[6*I+1-1]+216.0*ZTF[6*I+2-1]+27.0*ZTF[6*I+3-1]+272.0*ZTF[6*I+4-1]+27.0*ZTF[6*I+5-1]+216.0*ZTF[6*I+6-1]
        ZOM=ZOM*(1.0/turns)*(6.0/840.0)
        return ZOM
    
    def gramsc(self, turns: int, FR: float, A: float, B: float) -> None:
        """Remove the contribution of the frequency found from the Data and orthonormalize

        Args:
            turns (int): Number of turns
            FR (float): Frequency
            A (float): Real part of found term
            B (float): Imaginary part of found term
        """
        ZTEE = np.zeros(self.vars['NFS']+1).astype('complex128')
        for i in range(0, self.vars['NFS']):
            ZTEE[i] = self.proscaa(turns, FR, self.vars['TFS'][i])
        NF = self.vars['NFS']+1
        ZTEE[NF-1] = 1.0+0.0j
        self.vars['TFS'][NF-1] = FR
        for k in range(1, self.vars['NFS']+1):
            for i in range(1, self.vars['NFS']+1):
                for j in range(1,i+1):
                    self.vars['ZALP'][NF-1, k-1] = self.vars['ZALP'][NF-1, k-1] - np.conj(self.vars['ZALP'][i-1,j-1])*self.vars['ZALP'][i-1,k-1]*ZTEE[j-1]

        self.vars['ZALP'][NF-1, NF-1] = 1.0+0.0j
        DIV  = 1.0
        ZDIV = 0.0+0.0j
        for i in range(0, NF):
            ZDIV = ZDIV + np.conj(self.vars['ZALP'][NF-1, i])*ZTEE[i]
        DIV = np.sqrt(np.abs(ZDIV))
        self.vars['ZALP'][NF-1,:] = self.vars['ZALP'][NF-1,:]/DIV
        ZMUL = complex(A,B)/DIV
        ZI = 0.0+1.0j

        for i in range(0, NF):
            ZOM = 1.0j*self.vars['TFS'][i]
            ZA  = self.vars['ZALP'][NF-1,i]*ZMUL
            self.vars['ZAMP'][i] = self.vars['ZAMP'][i]+ZA
            ZT_zero = np.array([ZA])
            i_line = np.linspace(1, turns, num=turns, endpoint=True)
            ZT_tmp = ZA*np.exp(2.0*(i_line)*1.0j*np.pi*self.vars['TFS'][i])
            ZT = np.append(ZT_zero, ZT_tmp).ravel()
            self.vars['ZTABS'] = self.vars['ZTABS'] - ZT
    
    def run(self) -> np.array:
        """Using the provided input during initialization run the NAFF algorithm.

        Returns:
            np.array: The array with the results 
                      [order of harmonic, frequency, Amplitude, Re{Amplitude}, Im{Amplitude}]
        """
        FREFON = 1.0/self.turns
        NEPS   = 100000000
        EPS    = FREFON/NEPS

        T    = np.linspace(0, self.turns, num=self.turns+1, endpoint=True)*2.0*np.pi - np.pi*self.turns
        self.vars['TWIN'] = 1.0+np.cos(T/self.turns)
        self.vars['TWIN'] = ((2.0**self.window*math.factorial(self.window)**2)/float(math.factorial(2*self.window)))*(1.0+np.cos(T/self.turns))**self.window
        self.vars['ZTABS'] = self.data[self.skipTurns:self.skipTurns+self.turns+1]

        STAREP = FREFON/3.0
        for term in range(self.nterms):
            data_for_fft = np.multiply(self.vars['ZTABS'], self.vars['TWIN'])[:-1] # .astype('complex128')
            if self.getFullSpectrum:
                y = np.fft.fft(data_for_fft)
            else:
                y = np.fft.rfft(data_for_fft)

            RTAB = np.sqrt(np.real(y)**2 + np.imag(y)**2)/self.turns  # normalized
            INDX = np.argmax(RTAB)
            VMAX = np.max(RTAB)

            if INDX == 0 and self.dcWarn:
                self.logger.warn('Remove the DC component from the data (i.e. the mean).')
            if INDX <= self.turns/2.0:
                IFR = INDX - 1
            else:
                IFR = INDX-1-self.turns

            FR = (IFR+1)*FREFON
            FR, RMD, A, B = self.frefin(self.turns, FR, STAREP, EPS)
            IFLAG, NUMFR = self.fretes(FR, FREFON)
            if IFLAG ==1:
                self.gramsc(self.turns, FR, A, B)
                self.vars['NFS'] = self.vars['NFS'] + 1
            elif IFLAG == 0:
                # continue
                break  # if I put continue it will find again and again the same freq/ with break it stops repeating
            elif IFLAG == -1:
                self.modfre(self.turns, FR, NUMFR, A, B)

        result = []
        for i in range(self.vars['NFS']):
            AMP = np.abs(self.vars['ZAMP'][i])
            result.append(np.array([int(i), self.vars['TFS'][i], AMP, np.real(self.vars['ZAMP'][i]), np.imag(self.vars['ZAMP'][i])]))
        result = np.array(result)
        # set the results as class parameter
        self.results = result
        return result
    
    def to_pandas(self) -> pd.DataFrame:
        """Converts the results cache to pandas dataframe

        Returns:
            pd.DataFrame: A Pandas DataFrame with the results
        """
        column_names = ["harmonic_order", "frequency", "amplitude", "real_amplitude", "imag_amplitude"]
        return pd.DataFrame(data=self.results, columns=column_names)
    
    def to_dict(self) -> dict:
        """Converts the results cache to python dictionary

        Returns:
            dict: A python dict with the results
        """
        
        return {
            "harmonic_order" : self.results[:,0],
            "frequency" : self.results[:,1],
            "amplitude" : self.results[:,2],
            "real_amplitude" : self.results[:,3],
            "imag_amplitude" : self.results[:,4]
        }
        

# Example
# if __name__ == '__main__':
#     Generate sample signal
#     x = np.linspace(1, 500, num=500, endpoint=True)
#     f0, a0 = [0.31, 0.32, 0.33, 0.34, 0.34 + 0.016, 0.34 - 0.016], [1, 0.5, 0.25, 0.12, 0.06, 0.03]
#     data = np.array(sum([a * np.sin(2.0 * np.pi * (q * x)) for q, a in zip(f0, a0)]))
#     
#     Initialize the FundamentalFrequencies class
#     my_naff = FundamentalFrequencies(data, turns=300, nterms=20, skipTurns=0, getFullSpectrum=True)
#     results_arr = my_naff.run() # run also returns the array, but results can also be accessed through the object
#     print(f"Frequencies:\n{my_naff.results[:,1]}")
#     print(f"Amplitudes:\n{my_naff.results[:, 2]}")
#
#     Convert results to pandas
#     print(my_naff.to_pandas())
#
#     Convert results to dict
#     print(my_naff.to_dict())
    


