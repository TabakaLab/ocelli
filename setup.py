from pathlib import Path

from setuptools import setup, find_packages

setup(
    name='ocelli',
    version='0.1.1',
    python_requires='>=3.7',
    install_requires=['anndata', 
                      'matplotlib',
                      'numpy',
                      'pandas',
                      'plotly',
                      'ray',
                      'scikit-learn',
                      'scipy',
                      'statsmodels',
                      'umap-learn',
                      'scanpy',
                      'louvain'],
    author='Piotr Rutkowski',
    author_email='prutkowski@ichf.edu.pl',
    description='Single-cell developmental landscapes from multimodal data',
    license='BSD-Clause 2',
    url='https://github.com/TabakaLab/ocelli',
    download_url='https://github.com/TabakaLab/ocelli',
    keywords=[
        'single cell',
        'developmental process',
        'multimodal', 
        'multiomics',],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    packages=find_packages(),
    package_data={
        "ocelli": ["forceatlas2/forceatlas2.jar", "forceatlas2/gephi-toolkit-0.9.2-all.jar"]
    }
)
