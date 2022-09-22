Installation
------------

Ocelli was written for Python 3.7.

Dependencies
^^^^^^^^^^^^

ForceAtlas2 requires the Java Development Kit. On Linux you can download and install it by running::

    sudo apt update
    sudo apt install default-jdk
    
For maximum performance of approximated nearest neighbors search using ``nmslib``, install it from sources::

    pip install --no-binary :all: nmslib

PyPI
^^^^

Pull Ocelli from PyPI_ using (consider using `pip3` to access Python 3)::

    pip install ocelli

.. _PyPI: https://pypi.org/project/ocelli