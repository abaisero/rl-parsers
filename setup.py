from setuptools import setup

setup(
    name='rl_parsers',
    version='1.0.0',
    description='parsers for RL file formats',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/rl-parsers',
    packages=['rl_parsers'],
    install_requires=['numpy', 'ply'],
    license='MIT',
)
