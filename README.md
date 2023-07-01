# PyNAFF: Python-native Numerical Analysis of Fundamental Frequencies

## 1. Description

NAFF stands for Numerical Analysis of Fundamental Frequencies and the current package is the Python-native implementation of the [work of J. Lashkar](http://www.sciencedirect.com/science/article/pii/001910359090084M). The code is not optimized for performance, but rather kept as close to the original for reference purposes.

## 2. Installation

Using pip
```bash
pip install [--user] PyNAFF
```

Using Git
```bash
pip install --user git+https://github.com/nkarast/PyNAFF.git
```

### 2.1 Dependencies

Dependencies are kept to minimal, however the latest version would require the following libraries
- numpy
- pandas

### 2.2 Recommended Setup

We recommend (although not necessary) using an up-to-date setup such as:
- `Python >= 3.9`
- `numpy>=1.24`
- `pandas>=2.0`

## 3. How to use

Let's generate a sample signal 

```python
import PyNAFF as pnf
import numpy as np

t = np.linspace(1, 3000, num=3000, endpoint=True)
Q = 0.12345
signal = np.sin(2.0*np.pi*Q*t)
```

### 3.1 Before version 1.2

To get the resulting frequency components

```python
pnf.naff(signal, turns=500, nterms=1, skipTurns=0, getFullSpectrum=False, window=1)
# returns: array([[ 0., 0.12345,    0.5,    0.35009355, -0.35697971]])
```


### 3.2 From version 1.2

At version 1.2 the `FundamentalFrequencies` class was introduced which is simply a wrapper around the `naff()` function, but offers some additional features.

One can create a `FundamentalFrequencies` object, quite similar to the previous `naff()` function call.

```python
my_naff = pnf.FundamentalFrequencies(signal, turns=500, nterms=1, skipTurns=0, getFullSpectrum=False, window=1)
```

To run the analysis of the frequencies, simply execute the `run()` method. 

```python
my_results = my_naff.run()
my_results
# returns: array([[ 0., 0.12345,    0.5,    0.35009355, -0.35697971]])
```

The call to the above method returns the results to an array with format `[order of harmonic, frequency, Amplitude, Re{Amplitude}, Im{Amplitude}]`, but also updates the data of the class for further manipulation. This means that now one can get the data in Pandas DataFrame or Dict/JSON format, to simplify downstream analysis tasks.

```python
res_df = my_naff.to_pandas()
```

or

```python
res_dict = my_naff.to_dict()
```


## 4. How to contribute

Please install and test the code. 

In case of questions or recommendations do not hesitate to open an [issue](https://github.com/nkarast/PyNAFF/issues).

We would be delighted if you would be interested in contributing in terms of code maintenance and feature development. Do not hesitate to fork the project and create pull requests.


## 5. Contacts:

*  Foteini Asvesta (fasvesta .at. cern .dot. ch)
*  Panagiotis Zisopoulos (pzisopou .at. cern .dot. ch)
*  Nikos Karastathis 