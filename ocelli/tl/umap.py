import anndata as ad
from umap import UMAP
from scipy.sparse import issparse
from multiprocessing import cpu_count


def umap(adata: ad.AnnData,
         x = None,
         n_components: int = 2,
         n_neighbors: int = 20,
         min_dist: float = 0.1,
         spread: float = 1.,
         out: str = 'X_umap',
         random_state = None,
         n_jobs: int = -1,
         copy=False):
    """
    UMAP dimensionality reduction

    Uniform Manifold Approximation and Projection (UMAP) is a dimensionality reduction technique that
    is particularly useful for visualizing high-dimensional data in 2D or 3D space. This function is a 
    wrapper for the `umap-learn` library.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param x: Key in `adata.obsm` storing the data to be reduced. If `None`, `adata.X` is used. (default: `None`)
    :type x: str or None

    :param n_components: Number of dimensions in the UMAP embedding. (default: 2)
    :type n_components: int

    :param n_neighbors: Number of neighboring points used for manifold approximation. Larger values
        preserve more global structures, while smaller values focus on local structures. (default: 20)
    :type n_neighbors: int

    :param min_dist: Minimum distance between embedded points. Smaller values result in tighter clusters,
        while larger values spread points more evenly. (default: 0.1)
    :type min_dist: float

    :param spread: Scale of the embedded points. Works in conjunction with `min_dist` to control clustering. (default: 1.0)
    :type spread: float

    :param out: Key in `adata.obsm` where the UMAP embedding will be stored. (default: `'X_umap'`)
    :type out: str

    :param random_state: Seed for reproducibility. If `None`, no seed is set. (default: `None`)
    :type random_state: int or None

    :param n_jobs: Number of parallel jobs to use. If `-1`, all available CPUs are used. (default: -1)
    :type n_jobs: int

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the UMAP embedding stored in `adata.obsm[out]`.
        - If `copy=True`: Returns a modified copy of `adata` with the UMAP embedding.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm['embedding'] = np.random.rand(100, 10)

            # Compute UMAP embedding
            oci.tl.umap(adata, x='embedding', n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    """
    
    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])
    
    X = adata.X if x is None else adata.obsm[x]
    if issparse(X):
        X = X.toarray()

    reducer = UMAP(n_components=n_components, 
                   n_neighbors=n_neighbors,
                   min_dist=min_dist, 
                   spread=spread,
                   random_state=random_state,
                   n_jobs=n_jobs)
    adata.obsm[out] = reducer.fit_transform(X)
    
    return adata if copy else None
