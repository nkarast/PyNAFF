from distutils.core import setup
from setuptools import setup, find_packages

setup(
        name = 'PyNAFF',         # How you named your package folder (MyLib)
        #packages = ['PyNAFF'],   # Chose the same as "name"
        version = '1.2',      # Start with a small number and increase it with every change you make
        license='GPLv3',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
        description = 'A Python module that implements the Numerical Analysis of Fundamental Frequencies (NAFF) algorithm',   # Give a short description about your library
        author='Nikos Karastathis',
        author_email='nkarast@gmail.com',   # Type in your E-Mail
        url='https://github.com/nkarast/PyNAFF',
        packages=find_packages(),
        download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
        keywords = ['NAFF', 'FREQUENCY ANALYSIS'],   # Keywords that define your package best
        install_requires=[            # I get to this in a second
                'numpy',
                'pandas',
            ],
        classifiers=[
        'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GPL',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.9',
        ]
    )