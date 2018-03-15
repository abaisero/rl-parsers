#!/usr/bin/env python

import os
import sys

from setuptools import setup


PYTHON_CURRENT = sys.version_info[0]
PYTHON_REQUIRED = 3

if PYTHON_CURRENT < PYTHON_REQUIRED:
    sys.stderr.write("Python >= 3 is required.")
    sys.exit(1)


setup(
    name='rl_parsers',
    version='0.1.0',
    description='rl_parsers - parsers for old and new RL file formats',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/bigblindbais/rl_parsers',

    packages=['rl_parsers'],
    package_dir={'':'src'},
    test_suite='tests',

    install_requires=['numpy', 'ply'],
    license='MIT',
)
