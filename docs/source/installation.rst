Installation
------------

Ocelli was written for Python 3.7+.

The source code is available on `GitHub <https://github.com/TabakaLab/ocelli>`_. Ocelli is in active development. If you notice any bugs in installation or when running code, please submit bug reports by opening an `issue <https://github.com/TabakaLab/ocelli/issues>`_.

Installation steps
^^^^^^^^^^^^^^^^^^

#. Install Java

    Force-directed layout graph visualization requires Java. You can download Java installation files `here <https://www.java.com/en/download/>`_.
    
    Alternatively, on Linux, you can install Java using ::

        sudo apt update
        sudo apt install default-jdk

#. Install NMSLIB library
    
    `NMSLIB <https://pypi.org/project/nmslib/>`_ library is utilized for efficient computation of approximate nearest neighbors. For maximum performance, install the package from sources ::

        pip3 install --no-binary :all: nmslib
        
    Installation on ARM-based Macs (with M1/M2 chips) is problematic, as a compatible release is not available yet. The installation can be executed with the following script, but it may affect the performance. ::
    
        CFLAGS="-mavx -DWARN(a)=(a)" pip3 install --no-binary :all: nmslib

#. Install Ocelli
    
    Pull Ocelli from `GitHub <https://github.com/TabakaLab/ocelli>`_ using ::

        pip3 install -U git+https://github.com/TabakaLab/ocelli.git@main
