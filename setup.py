#!/usr/bin/python
"""representation-dl setup script."""

from setuptools import setup

# Package Version
from representation-dl import __version__ as version

setup(
    name='representation-dl',
    version=version,

    description=(''),
    long_description=open('README.md').read(),
    author='Vanessa D\'Amario,
    author_email='vanessa.damario@dibris.unige.it',
    maintainer='Vanessa D\'Amario',
    maintainer_email='vanessa.damario@dibris.unige.it',
    # url='https://github.com/slipguru/representation-dl',
    # download_url='https://github.com/slipguru/representation-dl/tarball/'+version,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Informatics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    license='FreeBSD',

    packages=['information_networks', 'information_networks.core', 'information_networks.utils', 'information_networks.externals'],
    install_requires=['numpy (>=1.10.1)',
                      'scipy (>=0.16.1)',
                      'scikit-learn (>=0.18)',
                      'matplotlib (>=1.5.1)',
                      'tensorflow'],
    #scripts=['scripts/ade_run.py', 'scripts/ade_analysis.py',
    #         'scripts/ade_GEO2csv.py'],
)
