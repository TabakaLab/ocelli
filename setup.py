from pathlib import Path

from setuptools import setup, find_packages

setup(
    name='ocelli',
    version='0.0.28',
    
    python_requires='>=3.7',
    install_requires=['anndata>=0.7.6', 
                      'matplotlib>=3.4.3',
                      'nmslib>=2.1.1',
                      'numpy>=1.20.0',
                      'pandas>=1.3.5',
                      'plotly>=5.3.1',
                      'ray>=1.8.0',
                      'scikit-learn>=1.0',
                      'scipy>=1.7.1',
                      'statsmodels>=0.13.0',
                      'sphinx>=1.4',
                      'umap-learn>=0.5.3',
                      'ipykernel',
                      'nbsphinx'],
    
    author='Piotr Rutkowski',
    author_email='prutkowski@ichf.edu.pl',
    
    description='Single-cell developmental landscapes from multimodal data',
    license='MIT',
    
    url='https://github.com/TabakaLab/ocelli',
    download_url='https://github.com/TabakaLab/ocelli',
    
    keywords=[
        'single cell',
        'developmental process',
        'multimodal', 
        'multiomics',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    packages=find_packages(),
    
    package_data={
        "ocelli": ["forceatlas2/forceatlas2.jar", "forceatlas2/gephi-toolkit-0.9.2-all.jar"]
    }
)
