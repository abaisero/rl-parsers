from setuptools import setup

from rl_parsers import __version__

setup(
    name='rl_parsers',
    version=__version__,
    description='parsers for RL file formats',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/rl-parsers',
    packages=['rl_parsers'],
    install_requires=['numpy', 'ply'],
    license='MIT',
)
