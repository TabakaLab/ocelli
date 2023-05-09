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

**Multimodal weights**

.. autosummary::
   :toctree: .
   
   tl.weights
   tl.scale_weights


**Multimodal Diffusion Maps**

.. autosummary::
   :toctree: .  
   
   tl.MDM
   
**Multimodal imputation**

.. autosummary::
   :toctree: .  
   
   tl.imputation
   
   
**Graph representations**

.. autosummary::
   :toctree: .
   
   tl.neighbors_graph
   tl.transitions_graph
   
**Dimension reduction**

.. autosummary::
   :toctree: .
   
   tl.FA2
   tl.UMAP
   tl.projection

**Gene signatures**

.. autosummary::
   :toctree: .
   
   tl.mean_z_scores
   
**Clustering**

.. autosummary::
   :toctree: .
   
   tl.louvain
   
   
Plotting (pl)
-------------

.. autosummary::
   :toctree: .
   
   pl.scatter
   pl.scatter_interactive
   pl.projections
   pl.violin
   pl.bar
   