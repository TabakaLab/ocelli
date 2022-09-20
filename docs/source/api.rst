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
   pp.generate_views
   

Tools (tl)
----------

**Multi-view diffusion maps**

.. autosummary::
   :toctree: .
   
   tl.weights
   tl.scale_weights
   tl.MVDM
   
   
**Graph representations**

.. autosummary::
   :toctree: .
   
   tl.nn_graph
   tl.vel_graph
   
**Plotting tools**

.. autosummary::
   :toctree: .
   
   tl.FA2
   tl.UMAP
   tl.project_2d

**Gene signatures**

.. autosummary::
   :toctree: .
   
   tl.mean_z_scores
   
   
Plotting (pl)
-------------

.. autosummary::
   :toctree: .
   
   pl.scatter
   pl.weights
   pl.topics
