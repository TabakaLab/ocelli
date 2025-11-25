from pathlib import Path

from setuptools import setup, find_packages

setup(
    name='ocelli',
    version='1.0.0',
    python_requires='>=3.8',
    install_requires=['anndata==0.10.9', 
                      'matplotlib==3.9.4',
                      'numpy==1.26.4',
                      'pandas==2.2.3',
                      'plotly==6.0.0',
                      'ray==2.0.0',
                      'scikit-learn==1.6.1',
                      'scipy==1.13.1',
                      'statsmodels==0.14.4',
                      'umap-learn==0.5.7',
                      'scanpy==1.10.3',
                      'louvain==0.8.2',
                      'nmslib==2.1.2'],
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
