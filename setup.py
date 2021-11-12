from setuptools import find_packages, setup

setup(
    name='rl_parsers',
    version='1.0.0',
    description='parsers for RL file formats',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/rl-parsers',
    packages=find_packages(include=['rl_parsers', 'rl_parsers.*']),
    install_requires=['numpy', 'ply'],
    license='MIT',
)
