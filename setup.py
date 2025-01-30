from pathlib import Path

from setuptools import setup, find_packages

setup(
    name='ocelli',
    version='1.0.0',
    python_requires='>=3.8',
    install_requires=['anndata', 
                      'matplotlib',
                      'numpy',
                      'pandas',
                      'plotly',
                      'ray==2.0.0',
                      'scikit-learn==1.0.2',
                      'scipy',
                      'statsmodels',
                      'umap-learn',
                      'scanpy',
                      'louvain'],
    author='Piotr Rutkowski',
    author_email='prutkowski@ichf.edu.pl',
    description='Single-cell developmental landscapes from multimodal data',
    license='BSD-Clause 2',
    keywords=['single-cell', 'multimodal', 'multiomics', 'multiomics'],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",],
    packages=find_packages(),
    package_data={"ocelli": ["forceatlas2/forceatlas2.jar", "forceatlas2/gephi-toolkit-0.9.2-all.jar"]}
)