import numpy as np
import math
from warnings import warn, simplefilter
simplefilter("ignore", np.ComplexWarning)  # suppress persisting cast to complex warnings from numpy
"""
# NAFF - Numerical Analysis of Fundamental Frequencies
# Version : 1.1.5
# Authors : F. Asvesta, N. Karastathis, P.Zisopoulos
# Contact : fasvesta@cern.ch
#
"""

__version = '1.1.5'
__PyVersion = [2.7, 3.7]
__authors = {'F. Asvesta': 'fasvesta@cern.ch',
			  'N. Karastathis': 'nkarast@gmail.com',
			  'P. Zisopoulos': 'pzisopou@cern.ch'
			}

def naff(data, turns=300, nterms=1, skipTurns=0, getFullSpectrum=False, window=1, tol=1e-4, warnings=True):
	'''
	The driving function for the NAFF algorithm.
	Inputs :
	*  data : NumPy array with TbT data
	*  turns : number of points to consider from the input data
	*  nterms : maximum number of harmonics to search for in the data sample
	*  skipTurns : number of observations (data points) to skip from the start of the input iterable
	*  getFullSpectrum : [True | False]
					  If True, a normal FFT is used (both negative and positive frequencies)
					  If False, an rFFT is used (only positive frequencies)
	*  window : the order of window to be applied on the input data (default =1)
	*  tol : Expert setting to increase the acceptance window for the harmonics, as long as `getFullSpectrum=True`.
	         Higher values should let NAFF recover more frequencies, but the maximum number will always be `nterms`.
	         Default value should be 1e-4.

	Returns : Array with frequencies and amplitudes in the format:
		  [order of harmonic, frequency, Amplitude, Re{Amplitude}, Im{Amplitude}]
	'''
	if turns >= len(data)+1:
		raise ValueError('#naff : Input data must be at least of length turns+1.')
	if turns < 6:
		raise ValueError('#naff : Minimum number of turns is 6.')

	if np.mod(turns,6)!=0:
		a,b=divmod(turns,6)
		turns = int(6*a)

	NFR  = 100
	vars = {
	'NFS' 		: 0,
	'TFS' 		: np.zeros(NFR).astype('float64'),
	'ZAMP' 		: np.zeros(NFR).astype('complex128'),
	'ZALP'	 	: np.zeros((NFR,NFR)).astype('complex128'),
	'ZTABS' 	: np.array([]).astype('complex128'),
	'TWIN'  	: np.array([]).astype('float64'),
	}
	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
	def getIntegral(FR, turns):
		'''
		Calculate the integral using Hardy's method'
		'''
		if np.mod(turns, 6)!= 0:
			raise ValueError("Turns need to be *6")
		K = int(turns/6)

		i_line = np.linspace(1, turns, num=turns, endpoint=True)
		ZTF_tmp = vars['ZTABS'][1:]*vars['TWIN'][1:]*np.exp(-2.0*(i_line)*np.pi*1.0j*FR)
		ZTF = np.array(vars['ZTABS'][0]*vars['TWIN'][0])
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
	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
	def frefin(turns, FR, STAREP, EPS):
		'''
		Try to refine the frequency found using slopes & root finding methods
		'''
		EPSI = 1.0e-15
		X2  = FR
		PAS = STAREP
		Y2, A2, B2  = getIntegral(X2, turns)
		X1  = X2 - PAS
		X3  = X2 + PAS
		Y1, A1, B1  = getIntegral(X1, turns)
		Y3, A3, B3  = getIntegral(X3, turns)
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
						Y2, A2, B2 = getIntegral(X2, turns)
						X3 = X2 + PAS
						Y3, A3, B3 = getIntegral(X3, turns)
					else:
						X3 = X2
						Y3, A3, B3 = Y2, A2, B2
						X2 = XX2
						Y2, A2, B2 = getIntegral(X2, turns)
						X1 = X2 - PAS
						Y1, A1, B1 = getIntegral(X1, turns)
				else:
					if Y1>Y3:
						X2 = X1
						Y2, A2, B2 = Y1, A1, B1
					else:
						X2 = X3
						Y2, A2, B2 = Y3, A3, B3

					X1 = X2 - PAS
					X3 = X2 + PAS
					Y1, A1, B1 = getIntegral(X1, turns)
					Y3, A2, B2 = getIntegral(X3, turns)
					if (Y3-Y1)-(Y3-Y2)==0.0:
						PAS=PAS+EPS

			else:
				break
		return X2, Y2, A2, B2
	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
	def fretes(FR, FREFON):
		'''
		If more than one term found, check how different they are
                '''
		IFLAG = 1
		NUMFR = 0
		ECART = np.abs(FREFON)
		for i in range(len(vars['TFS'])):
			TEST = np.abs(vars['TFS'][i] - FR)
			if TEST < ECART:
				if float(TEST)/float(ECART) < tol: # tolerance value was 1e-4 in Laskar's original work.
					IFLAG = -1
					NUMFR = i
					break
				else:
					IFLAG = 0
					continue
		return IFLAG, NUMFR
	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
	def modfre(turns, FR, NUMFR, A, B):
		'''
		If I found something very close to one of the FR before, I assume that this comes from data
		I had not removed successfully => Remove them without orthonormalization
		'''
		ZI  = 0. + 1.0j
		ZOM = 1.0j*FR
		ZA  = 1.0*A + 1.0j*B
		if len(vars['ZAMP'])<= NUMFR:
			vars['ZAMP'][NUMFR] = 0
		vars['ZAMP'][NUMFR] = vars['ZAMP'][NUMFR] + ZA
		i_line = np.linspace(1, turns, num=turns, endpoint=True)
		ZT_tmp = ZA*np.exp(2.0*(i_line)*np.pi*ZOM)
		ZT     = np.array([ZA])
		ZT     = np.append(ZT, ZT_tmp).ravel()
		ZTABS_tmp = vars['ZTABS'] - ZT
		vars['ZTABS'] = ZTABS_tmp
	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
	def proscaa(turns, FS, FS_OLD):
		ZI = 0.0+1.0j
		OM = FS-FS_OLD
		ANGI = 2.0*np.pi*OM
		i_line = np.linspace(1, turns, num=turns, endpoint=True)
		ZT_tmp = np.exp(-2.0*(i_line)*1.0j*np.pi*OM)
		ZT_zero = np.array([1])
		ZT = np.append(ZT_zero, ZT_tmp).ravel()
		ZTF = np.multiply(vars['TWIN'],ZT)
		N = turns + 1
		ZOM = 41.*ZTF[0]+216.*ZTF[1]+27.*ZTF[2]+272.*ZTF[3]+27.*ZTF[4]+216.*ZTF[5]+41.*ZTF[int(N)-1]
		for I in range(1, int(turns/6)):
			ZOM=ZOM+82.0*ZTF[6*I+1-1]+216.0*ZTF[6*I+2-1]+27.0*ZTF[6*I+3-1]+272.0*ZTF[6*I+4-1]+27.0*ZTF[6*I+5-1]+216.0*ZTF[6*I+6-1]

		ZOM=ZOM*(1.0/turns)*(6.0/840.0)
		return ZOM
	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
	def gramsc(turns, FR, A, B):
		'''
		Remove the contribution of the frequency found from the Data and orthonormalize
		'''

		ZTEE = np.zeros(vars['NFS']+1).astype('complex128')
		for i in range(0, vars['NFS']):
			ZTEE[i] = proscaa(turns, FR, vars['TFS'][i])
		NF = vars['NFS']+1
		ZTEE[NF-1] = 1.0+0.0j
		vars['TFS'][NF-1] = FR
		for k in range(1,vars['NFS']+1):
			for i in range(1, vars['NFS']+1):
				for j in range(1,i+1):
					vars['ZALP'][NF-1, k-1] = vars['ZALP'][NF-1, k-1] - np.conj(vars['ZALP'][i-1,j-1])*vars['ZALP'][i-1,k-1]*ZTEE[j-1]

		vars['ZALP'][NF-1, NF-1] = 1.0+0.0j
		DIV  = 1.0
		ZDIV = 0.0+0.0j
		for i in range(0, NF):
			ZDIV = ZDIV + np.conj(vars['ZALP'][NF-1, i])*ZTEE[i]
		DIV = np.sqrt(np.abs(ZDIV))
		vars['ZALP'][NF-1,:] = vars['ZALP'][NF-1,:]/DIV
		ZMUL = complex(A,B)/DIV
		ZI = 0.0+1.0j

		for i in range(0, NF):
			ZOM = 1.0j*vars['TFS'][i]
			ZA  = vars['ZALP'][NF-1,i]*ZMUL
			vars['ZAMP'][i] = vars['ZAMP'][i]+ZA
			ZT_zero = np.array([ZA])
			i_line = np.linspace(1, turns, num=turns, endpoint=True)
			ZT_tmp = ZA*np.exp(2.0*(i_line)*1.0j*np.pi*vars['TFS'][i])
			ZT = np.append(ZT_zero, ZT_tmp).ravel()
			vars['ZTABS'] = vars['ZTABS'] - ZT


	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
	FREFON = 1.0/turns
	NEPS   = 100000000
	EPS    = FREFON/NEPS

	T    = np.linspace(0, turns, num=turns+1, endpoint=True)*2.0*np.pi - np.pi*turns
	vars['TWIN'] = 1.0+np.cos(T/turns)
	vars['TWIN'] = ((2.0**window*math.factorial(window)**2)/float(math.factorial(2*window)))*(1.0+np.cos(T/turns))**window
	vars['ZTABS'] = data[skipTurns:skipTurns+turns+1]

	STAREP = FREFON/3.0
	for term in range(nterms):
		data_for_fft = np.multiply(vars['ZTABS'], vars['TWIN'])[:-1] # .astype('complex128')
		if getFullSpectrum:
			y = np.fft.fft(data_for_fft)
		else:
			y = np.fft.rfft(data_for_fft)

		RTAB = np.sqrt(np.real(y)**2 + np.imag(y)**2)/turns  # normalized
		INDX = np.argmax(RTAB)
		VMAX = np.max(RTAB)

		if INDX == 0 and warnings:
			warn('## PyNAFF::naff: Remove the DC component from the data (i.e. the mean).')
		if INDX <= turns/2.0:
			IFR = INDX - 1
		else:
			IFR = INDX-1-turns

		FR = (IFR+1)*FREFON
		FR, RMD, A, B = frefin(turns, FR, STAREP, EPS)
		IFLAG, NUMFR = fretes(FR, FREFON)
		if IFLAG ==1:
			gramsc(turns, FR, A, B)
			vars['NFS'] = vars['NFS'] + 1
		elif IFLAG == 0:
			# continue
			break  # if I put continue it will find again and again the same freq/ with break it stops repeating
		elif IFLAG == -1:
			modfre(turns, FR, NUMFR, A, B)

	result = []
	for i in range(vars['NFS']):
		AMP = np.abs(vars['ZAMP'][i])
		result.append(np.array([int(i), vars['TFS'][i], AMP, np.real(vars['ZAMP'][i]), np.imag(vars['ZAMP'][i])]))
	return np.array(result)


### - - - ### - - - ### - - - ### - - - ### - - - ### - - - ### - - - ### - - - ### - - - ### - - - ###

# Example
if __name__ == '__main__':
	x = np.linspace(1, 500, num=500, endpoint=True)
	f0, a0 = [0.31, 0.32, 0.33, 0.34, 0.34 + 0.016, 0.34 - 0.016], [1, 0.5, 0.25, 0.12, 0.06, 0.03]
	data = np.array(sum([a * np.sin(2.0 * np.pi * (q * x)) for q, a in zip(f0, a0)]))
	a = naff(data, 300, 20, 0, True)
	print(f"Frequencies:\n{a[:,1]}")
	print(f"Amplitudes:\n{a[:, 2]}")

