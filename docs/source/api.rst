.. module:: ocelli
.. automodule:: ocelli

API
===

Import Ocelli as::

   import ocelli as oci


Preprocessing (pp)
------------------

.. module:: ocelli.pp
.. currentmodule:: ocelli

**Nearest neighbors**

.. autosummary::
   :toctree: generated/

   pp.neighbors 
   
**Topic modeling**
   
.. autosummary::
   :toctree: generated/

   pp.latent_dirichlet_allocation
   pp.generate_views
   

Tools (tl)
----------

.. module:: ocelli.tl
.. currentmodule:: ocelli

**Multi-view diffusion maps**

.. autosummary::
   :toctree: generated/
   
   tl.weights
   tl.scale_weights
   tl.multi_view_diffusion_maps
   
   
**Graph representations**

.. autosummary::
   :toctree: generated/
   
   tl.nn_graph
   tl.vel_graph
   
**Plotting tools**

.. autosummary::
   :toctree: generated/
   
   tl.forceatlas2
   tl.project_2d

**Gene signatures**

.. autosummary::
   :toctree: generated/
   
   tl.z_scores
   
   
Plotting (pl)
-------------

.. module:: ocelli.pl
.. currentmodule:: ocelli

.. autosummary::
   :toctree: generated/
   
   pl.scatter
   pl.weights
   pl.topics
