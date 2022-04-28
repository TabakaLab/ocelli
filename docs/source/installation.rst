Installation
------------

Ocelli requires Python 3.7 or later.

PyPI
^^^^

Pull Ocelli from PyPI_ using (consider using `pip3` to access Python 3)::

    pip install ocelli

Dependencies
^^^^^^^^^^^^

ForceAtlas2 requires the Java Development Kit. Download and install it by running::

    sudo apt update
    sudo apt install default-jdk
    
For maximum performance of approximated nearest neighbors search using ``nmslib``, you can install it from sources::

    pip install --no-binary :all: nmslib

.. _PyPI: https://pypi.org/project/ocelli