# PyNAFF

Authors:

*  Foteini Asvesta (fasvesta .at. cern .dot. ch)
*  Nikos Karastathis (nkarast .at. cern .dot. ch)
*  Panagiotis Zisopoulos (pzisopou .at. cern .dot. ch)

A Python module that implements the [Numerical Analysis of Fundamental Frequencies method of J. Lashkar](http://www.sciencedirect.com/science/article/pii/001910359090084M).
The code works either as a script (as the original code of Lashkar) or loaded as a module in Python/Julia code or jupyter-like notebooks (i.e. SWAN).


## Installation:

The module is ported in [PyPi](https://pypi.org/project/PyNAFF/) so the user can simply run:

```bash
pip install --user PyNAFF
```

or from Git:
```bash
pip install --user git+https://github.com/nkarast/PyNAFF.git
```


## Example of Usage
```python
import PyNAFF as pnf
import numpy as np

t = np.linspace(1, 3000, num=3000, endpoint=True)
Q = 0.12345
signal = np.sin(2.0*np.pi*Q*t)


# Signature: pnf.naff(data, turns=300, nterms=1, skipTurns=0, getFullSpectrum=False, window=1)
# Docstring:
# The driving function for the NAFF algorithm.
# Inputs :
# *  data : NumPy array with TbT data
# *  turns : number of points to consider from the input data
# *  nterms : maximum number of harmonics to search for in the data sample
# *  skipTurns : number of observations (data points) to skip from the start of the input iterable
# *  getFullSpectrum : [True | False]
#                                   If True, a normal FFT is used (both negative and positive freq.)
#                                   If False, an rFFT is used (only positive frequencies)
# *  window : the order of window to be applied on the input data (default =1)
# Returns : Array with frequencies and amplitudes in the format:
#           [order of harmonic, frequency, Amplitude, Re{Amplitude}, Im{Amplitude}]

pnf.naff(signal, turns=500, nterms=1, skipTurns=0, getFullSpectrum=False, window=1)

# outputs an array of arrays for each frequency. Each sub-array includes:
# [order of harmonic, frequency, Amplitude, Re{Amplitude}, Im{Amplitude]


# My frequency is simply 
pnf.naff(signal, 500, 1, 0 , False)[0][1]

```




-- nkarast
