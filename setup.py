from distutils.core import setup
from setuptools import setup, find_packages

setup(
        name = 'PyNAFF',
        version = '1.1.5',
        license='GPLv3',
        description = 'A Python module that implements the Numerical Analysis of Fundamental Frequencies (NAFF) algorithm', 
        author='Nikos Karastathis',
        author_email='nkarast@gmail.com',
        url='https://github.com/nkarast/PyNAFF',
        packages=find_packages(),
        keywords = ['NAFF', 'FREQUENCY ANALYSIS'], 
        install_requires=[    
                'numpy',
            ],
        classifiers=[
        'Development Status :: 5 - Production/Stable',  
        'Intended Audience :: Developers',  
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', 
        'Programming Language :: Python :: 3', 
        'Programming Language :: Python :: 3.9',
        ]
)