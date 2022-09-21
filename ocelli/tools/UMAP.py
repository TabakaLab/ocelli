import anndata
import os
import numpy as np
import pandas as pd
import pkg_resources
import umap
from scipy.sparse import coo_matrix, diags, issparse
from scipy.sparse.linalg import eigsh
from multiprocessing import cpu_count
import ray



def UMAP(adata: anndata.AnnData,
         n_components: int = 2,
         n_neighbors: int = 15,
         min_dist: float = 0.1,
         spread: float = 1.,
         random_state = None,
         obsm_key = None,
         output_key: str = 'X_umap',
         copy=False):
    """UMAP

    Dimensionality reduction using UMAP.

    This funstion is a wrapper for umap-learn implementation.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_components
        The dimension of the space to embed into. (default: 2)
    n_neighbors
        The size of local neighborhood (in terms of number of neighboring sample points) 
        used for manifold approximation. Larger values result in more global views
        of the manifold, while smaller values result in more local data being preserved. 
        In general values should be in the range 2 to 100.
    min_dist
        The effective minimum distance between embedded points.
        Smaller values will result in a more clustered/clumped embedding where nearby
        points on the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points.
    spread
        The effective scale of embedded points. 
        In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)
    obsm_key
        ``adata.obsm[obsm_key]`` stores an array for dimension reduction.
        If :obj:`None`, ``adata.X`` is used. (default: :obj:`None`)
    output_key
        UMAP embedding is saved to ``adata.obsm[output_key]``. (default: `X_umap`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)
        
    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[output_key]`` (:class:`numpy.ndarray` of shape ``(n_obs, n_components)`` storing 
        a UMAP data representation).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """
    
    X = adata.X if obsm_key is None else adata.obsm[obsm_key]
    if issparse(X):
        X = X.toarray()

    reducer = umap.UMAP(n_components=n_components, 
                        n_neighbors=n_neighbors,
                        min_dist=min_dist, 
                        spread=spread,
                        random_state=random_state)
    adata.obsm[output_key] = reducer.fit_transform(X)
    
    return adata if copy else None
