#
#       Install it with:
#               python setup.py build_ext --inplace
#
from setuptools import setup, find_packages
from os import path
from codecs import open
import numpy as np


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


include_dirs = [np.get_include()]


setup(
    name = "PyNAFF",
    version = '1.1.2',
    description = 'A Python module that implements NAFF algorithm',
    long_description = long_description,
    # projects main page
    url='https://github.com/nkarast/PyNAFF',
    # Author details
    author='Nikos Karastathis',
    author_email='nkarast@gmail.com',
    # licence
    licence = 'GPLv3',
    keywords = "naff numerical analysis frequency fundamental",
    install_requires=['numpy', 'future'],
    packages = find_packages(),
#    ext_modules = cythonize(extensions),
)
