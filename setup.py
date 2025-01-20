from setuptools import setup, find_packages

VERSION = '0.1'

DESCRIPTION = 'Module to use iSIM (instant Similarity) to perform typical Cheminformatics tools: clustering, diversity selection, similarity calculations, activity cliff quantification, chemical space visualization, etc.'

setup(
        name='iChem',
        version=VERSION,
        description=DESCRIPTION
        url=''
        packages = find_packages(),
        install requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'rdkit',
        'scipy',
        'seaborn',
        'scikit-learn'
    ]
)
