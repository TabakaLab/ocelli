Usage Principles
----------------

Import Ocelli as: ::

    import ocelli as oci

Package Structure
^^^^^^^^^^^^^^^^^

Ocelli has three modules: ``oci.pp`` (data preprocessing), ``oci.tl`` (analysis tools), and ``oci.pl`` (plotting).

The workflow typically consists of multiple function calls on ``adata``, an :obj:`anndata.AnnData` object. For example, ::

    oci.tl.MVDM(adata, **function_params)
    
Using AnnData
^^^^^^^^^^^^^

:obj:`anndata.AnnData` is the major data structure used in Ocelli, resulting in compatibility with numerous single-cell analysis Python packages (e.g., Scanpy_, scVelo_). A thorough introduction to the :obj:`anndata.AnnData` data structure with tutorials can be found here_.

Here are some tips for starting data exploration with Ocelli.

1. For multimodal analyses, arrays (:obj:`numpy.ndarray` or :obj:`scipy.sparse.csr_matrix`) storing views (or modalities) should be saved to ``adata.obsm``. They should be already preprocessed, or at least dimensionally reduced. Note: lower-dimensional :obj:`numpy.ndarray` arrays (up to ~ 50-1000 dimensions depending on the number of cells) result in faster analyses.

2. If you have very high-dimensional views such as RNA-seq gene expression matrix, or ATAC-seq peak count matrix, preprocess them in a seperate :obj:`anndata.AnnData` objects. Then create new ``adata``, and copy the preprocessed matrices to ``adata.obsm`` for multimodal exploration.

3. The exception from above procedure is when you have a single-view high-dimensional matrix, and you intend to generate views automatically using ``oci.pp.generate_views``. In such case, save your matrix to ``adata.X`` and Ocelli will save new views automatically to ``adata.obsm``. See Tutorial 3.

4. When views are saved in ``adata.obsm``, save their names as a :obj:`list` to ``adata.uns['view_keys']`` (unless it's done automatically by ``oci.pp.generate_views``). This step ensures Ocelli correctly reads views.

See our tutorials for example Ocelli workflows in Jupyter notebooks.

.. _Scanpy: https://scvelo.readthedocs.io
.. _scVelo: https://scanpy.readthedocs.io
.. _here: https://anndata.readthedocs.io
