from setuptools import setup, find_packages


setup(
    name='rl_parsers',
    version='0.1.0',
    description='rl_parsers - parsers for old and new RL file formats',
    packages=find_packages(),
    install_requires=['numpy', 'ply'],
)
