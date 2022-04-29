.. automodule:: ocelli

API
===

Import Ocelli as::

   import ocelli as oci


Preprocessing (pp)
------------------

**Nearest neighbors**

.. autosummary::
   :toctree: pp

   neighbors 
   
**Topic modeling**
   
.. autosummary::
   :toctree: pp

   latent_dirichlet_allocation
   generate_views
   

Tools (tl)
----------

**Multi-view diffusion maps**

.. autosummary::
   :toctree: .
   
   tl.weights
   tl.scale_weights
   tl.multi_view_diffusion_maps
   
   
**Graph representations**

.. autosummary::
   :toctree: .
   
   tl.nn_graph
   tl.vel_graph
   
**Plotting tools**

.. autosummary::
   :toctree: .
   
   tl.forceatlas2
   tl.project_2d

**Gene signatures**

.. autosummary::
   :toctree: .
   
   tl.z_scores
   
   
Plotting (pl)
-------------

.. autosummary::
   :toctree: .
   
   pl.scatter
   pl.weights
   pl.topics
