# PyNAFF

Authors: 
*  Foteini Asvesta (fasvesta .at. cern .dot. ch)
*  Nikos Karastathis (nkarast .at. cern .dot. ch)
*  Panagiotis Zisopoulos (pzisopou .at. cern .dot. ch)

**Contact** : Nikos Karastathis (nkarast .at. cern .dot. ch)
**Compatibility** : Python 2.7+, 3.6+

A Python module that implements the [Numerical Analysis of Fundamental Frequencies method of J. Lashkar](http://www.sciencedirect.com/science/article/pii/001910359090084M).


## How to use it

* Load the module (for CERN users contact the authors for the SWAN version)
`import numpy as np
import PyNAFF as pnf`

* Request for 500 turns , 1 term, without skipping any rows in the array, and run with rFFT (False default)

`pnf.naff(signal, 500, 2, 0 , False)`

The output is an array of arrays. Each array contains a frequency term found.
The indices in the array correspond to :
*   Term ID : Integer number denoting the term
* Frequency
* Amplitude
* Re{Amplitude}
* Im{Amplitude}

Example @ Jupyter : https://cernbox.cern.ch/index.php/s/dHDZ2d7ufreC8EV


-- nkarast
