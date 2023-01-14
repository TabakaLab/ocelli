.. automodule:: ocelli

API
===

Import Ocelli as::

   import ocelli as oci


Preprocessing (pp)
------------------

**Nearest neighbors**

.. autosummary::
   :toctree: .

   pp.neighbors 
   
**Topic modeling**
   
.. autosummary::
   :toctree: .

   pp.LDA
   pp.modality_generation
   

Tools (tl)
----------

**Multimodal Diffusion Maps**

.. autosummary::
   :toctree: .
   
   tl.weights
   tl.scale_weights
   tl.MDM
   
   
**Graph representations**

.. autosummary::
   :toctree: .
   
   tl.neighbors_graph
   tl.velocity_graph
   tl.timestamp_graph
   
**Plotting tools**

.. autosummary::
   :toctree: .
   
   tl.FA2
   tl.UMAP
   tl.projection

**Gene signatures**

.. autosummary::
   :toctree: .
   
   tl.mean_z_scores
   
   
Plotting (pl)
-------------

.. autosummary::
   :toctree: .
   
   pl.scatter
   pl.scatter_interactive
   pl.weights
   pl.weights_per_cluster
   pl.projections
