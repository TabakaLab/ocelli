API
===

Import Ocelli as: ::

    import ocelli as oci

Data reading (`read`)
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :recursive:

   ocelli.read.h5ad

Preprocessing (`pp`)
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :recursive:

   ocelli.pp.neighbors
   ocelli.pp.lda
   ocelli.pp.generate_modalities

Tools (`tl`)
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :recursive:

   ocelli.tl.modality_weights
   ocelli.tl.scale_weights
   ocelli.tl.MDM
   ocelli.tl.imputation
   ocelli.tl.neighbors_graph
   ocelli.tl.transitions_graph
   ocelli.tl.fa2
   ocelli.tl.umap
   ocelli.tl.zscores
   ocelli.tl.louvain
   ocelli.tl.projection

Plotting (`pl`)
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :recursive:

   ocelli.pl.scatter
   ocelli.pl.scatter_interactive
   ocelli.pl.violin
   ocelli.pl.bar
   ocelli.pl.projections
