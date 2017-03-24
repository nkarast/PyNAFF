# PyNAFF

Authors: 

*  Nikos Karastathis (nkarast .at. cern .dot. ch)
*  Panagiotis Zisopoulos (pzisopou .at. cern .dot. ch)

A Python module that implements the [Numerical Analysis of Fundamental Frequencies method of J. Lashkar](http://www.sciencedirect.com/science/article/pii/001910359090084M).
The code works either as a script (as the original code of Lashkar) or loaded as a module in Python/Julia code or jupyter-like notebooks (i.e. SWAN).

## Folder Structure

The project is implemented in two ways.

### Python

This corresponds to the pure Python implementation. The whole analysis is ran in core Python + NumPy. 

The end-user only needs the `PyNAFF.py` file. Then one can import simply the file as a separate module.


### Cython

To speed up the processing time a simple Cython implementation has been developed. The Cython version is -at least- twice as fast, judging from the first benchmarks.

Under the Cython folder, the `setup.py` script implements the distutils Extension. This cythonizes the code into C and then compiles it, including the NumPy-specific libraries.

To setup the module at your local machine:

`python setup.py build_ext --inplace`

This will create the PyNAFF.c, PyNAFF.so files. The file that one needs is the shared object PyNAFF.so.

Running a script from this folder one only needs to import PyNAFF.
Otherwise one extra folder must be included in the local environment, e.g. :

`import Cython.PyNAFF as pnf`


-- nkarast




More info soon...