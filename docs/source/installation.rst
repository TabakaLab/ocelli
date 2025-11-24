Installation
============

Ocelli was developed and tested on Python 3.8 and 3.9.

The source code is available on `GitHub <https://github.com/TabakaLab/ocelli>`_. Ocelli is actively maintained and in continuous development. If you encounter any installation issues or bugs, please report them by opening an `issue <https://github.com/TabakaLab/ocelli/issues>`_.

Installation Steps
^^^^^^^^^^^^^^^^^^

To install `ocelli`, follow these steps:

0. **Install Java and Conda** (if not already installed)
   
   Java is required for force-directed layout graph visualization. You can download it from `here <https://www.java.com/en/download/>`_.
   
   Alternatively, on Linux, install Java using:
   
   .. code-block:: bash

       sudo apt update
       sudo apt install default-jdk

   Download and install Miniconda from `here <https://docs.conda.io/projects/miniconda/en/latest/>`_.

1. **Create and activate a conda environment**
   
   Open a terminal and create a new environment named `ocelli`, we recommend using Python 3.9:
   
   .. code-block:: bash

       conda create -n ocelli python=3.9
       conda activate ocelli       

2. **Download and install Ocelli**
   
   .. code-block:: bash

       git clone https://github.com/TabakaLab/ocelli.git
       pip install ocelli/.
