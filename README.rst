PyNAFF
======

Authors:

-  Foteini Asvesta (fasvesta .at. cern .dot. ch)
-  Nikos Karastathis (nkarast .at. cern .dot. ch)
-  Panagiotis Zisopoulos (pzisopou .at. cern .dot. ch)

A Python module that implements the `Numerical Analysis of Fundamental
Frequencies method of J. Lashkar`_. The code works either as a script
(as the original code of Lashkar) or loaded as a module in Python/Julia
code or jupyter-like notebooks (i.e. SWAN).

Example of Usage
----------------

.. code:: python

   import PyNAFF as pnf
   import numpy as np

   t = np.linspace(1, 3000, num=3000, endpoint=True)
   Q = 0.12345
   signal = np.sin(2.0*np.pi*Q*t)
   pnf.naff(signal, 500, 1, 0 , False)
   # outputs an array of arrays for each frequency. Each sub-array includes:
   # [order of harmonic, frequency, Amplitude, Re{Amplitude}, Im{Amplitude]


   # My frequency is simply
   pnf.naff(signal, 500, 1, 0 , False)[0][1]

– nkarast

.. _Numerical Analysis of Fundamental Frequencies method of J. Lashkar: http://www.sciencedirect.com/science/article/pii/001910359090084M
