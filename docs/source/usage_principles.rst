Usage Principles
----------------

Import Ocelli as: ::

    import ocelli as oci


Ocelli has three modules:

- ``oci.pp`` (data preprocessing), 
- ``oci.tl`` (analysis tools), 
- ``oci.pl`` (plotting).

The workflow typically consists of multiple function calls on ``adata``, an :obj:`anndata.AnnData` object. For example, ::

    oci.tl.MDM(adata, **function_params)
    
Ocelli uses :obj:`anndata.AnnData` data structure, resulting in compatibility with numerous single-cell analysis Python packages, e.g., Scanpy_, scVelo_. A thorough introduction to the :obj:`anndata.AnnData` data structure with tutorials can be found here_.

A series of tutorials was prepared to describe Ocelli's usage with various use cases. You can find them in the sidebar.

.. _Scanpy: https://scvelo.readthedocs.io
.. _scVelo: https://scanpy.readthedocs.io
.. _here: https://anndata.readthedocs.io
