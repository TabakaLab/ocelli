Installation
============

Ocelli was developed and tested on Python 3.8.

The source code is available on `GitHub <https://github.com/TabakaLab/ocelli>`_. Ocelli is actively maintained and in continuous development. If you encounter any installation issues or bugs, please report them by opening an `issue <https://github.com/TabakaLab/ocelli/issues>`_.

Installation Steps
^^^^^^^^^^^^^^^^^^

To install `ocelli`, follow these steps:

1. **Install Conda** (if not already installed)
   
   Download and install Miniconda from `here <https://docs.conda.io/projects/miniconda/en/latest/>`_.

2. **Create a Conda Environment**
   
   Open a terminal and create a new environment named `ocelli` with Python 3.8:
   
   .. code-block:: bash

       conda create -n ocelli python=3.8

3. **Activate the Conda Environment**
   
   Activate the newly created environment:
   
   .. code-block:: bash

       conda activate ocelli

4. **Install Java** (if not already installed)
   
   Java is required for force-directed layout graph visualization. You can download it from `here <https://www.java.com/en/download/>`_.
   
   Alternatively, on Linux, install Java using:
   
   .. code-block:: bash

       sudo apt update
       sudo apt install default-jdk

5. **Install NMSLIB Library**
   
   The `NMSLIB <https://pypi.org/project/nmslib/>`_ library is used for efficient approximate nearest neighbors computation. For optimal performance, install it from source:
   
   .. code-block:: bash

       pip install --no-binary :all: nmslib

   On ARM-based Macs (M-series chips), installation may be problematic as a compatible release is not available. You can attempt the following workaround, but note that it may impact performance:
   
   .. code-block:: bash

       CFLAGS="-mavx -DWARN(a)=(a)" pip install --no-binary :all: nmslib

6. **Clone the Ocelli Repository**
   
   .. code-block:: bash

       git clone https://github.com/TabakaLab/ocelli.git

7. **Navigate to the Ocelli Package Directory**
   
   Change to the directory where you cloned `ocelli`:
   
   .. code-block:: bash

       cd ocelli

8. **Install Ocelli with Dependencies**
   
   Run the following command to install `ocelli` along with its dependencies:
   
   .. code-block:: bash

       pip install -e .

9. **Deactivate the Conda Environment (Optional)**
   
   If you are done working, you can deactivate the environment:
   
   .. code-block:: bash

       conda deactivate
