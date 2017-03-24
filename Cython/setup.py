#
#	Install it with:
#		python setup.py build_ext --inplace
#
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

include_dirs = [np.get_include()]

extensions = [
		Extension("PyNAFF", ["PyNAFF.pyx"], include_dirs=include_dirs),
			]

setup(
    name = "PyNAFF",
    ext_modules = cythonize(extensions),
)
