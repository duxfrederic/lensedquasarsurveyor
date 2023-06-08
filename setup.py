#!/usr/bin/env python

from setuptools import setup, find_packages
with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='lensedquasarsurveyor',
    version='0.1',
    python_requires='>3.8.0',
    description='collection of utilities for the search of lensed quasars',
    author='Frederic Dux, Cameron Lemon',
    author_email='frederic.dux@epfl.ch',
    long_description=long_description,
    license='GPL',
    packages=find_packages(),
    package_data={
        "lensedquasarsurveyor": [],
        #"tests": ["test_data/*"]
    },
    install_requires=[
        'urllib3',
        'wget',
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
        'scikit-image',
        'pandas',
        ],
    entry_points = {
        }
)

