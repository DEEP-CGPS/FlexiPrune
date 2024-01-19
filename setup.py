from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1.0'
DESCRIPTION = 'Pruning of CNNs with distributions'
LONG_DESCRIPTION = 'This package allows to prune in different percentages each layer of a CNN.'

# Setting up
setup(
    name="pruningdistribution",
    version=VERSION,
    author="DEEP-CGPS",
    author_email="<est.cesar.pachon@unimilitar.edu.co>",
    url='https://github.com/DEEP-CGPS/PruningDistribution',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
    ],
    keywords=['python', 'pytorch', 'pruning', 'CNN', 'distribution', 'FLOPs'],
    python_requires='>=3.6'

)