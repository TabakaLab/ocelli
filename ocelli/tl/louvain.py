import scanpy as sc


def louvain(adata, 
            x: str = None, 
            out: str = 'louvain',
            n_neighbors: int = 20, 
            resolution: float = 1.,
            random_state = None,
            copy: bool = False):
    """
    Louvain clustering

    Computes Louvain clusters based on the nearest neighbor graph of the data. 
    This function is a wrapper for `scanpy.tl.louvain`.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param x: Key in `adata.obsm` storing the representation for clustering. If `None`, `adata.X` is used. (default: `None`)
    :type x: str or None

    :param out: Key in `adata.obs` where Louvain cluster labels are stored. (default: `'louvain'`)
    :type out: str

    :param n_neighbors: Number of nearest neighbors to use in graph construction. (default: 20)
    :type n_neighbors: int

    :param resolution: Resolution parameter for Louvain clustering. Higher values result in more clusters. (default: 1.0)
    :type resolution: float

    :param random_state: Seed for reproducibility. If `None`, no seed is set. (default: `None`)
    :type random_state: int or None

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the Louvain clusters stored in `adata.obs[out]`.
        - If `copy=True`: Returns a modified copy of `adata` with the Louvain clusters.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm['embedding'] = np.random.rand(100, 10)

            # Compute Louvain clusters
            oci.tl.louvain(adata, x='embedding', n_neighbors=15, resolution=0.8, random_state=42)
    """

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=x)
    sc.tl.louvain(adata, resolution=resolution, random_state=random_state, key_added=out)
    
    return adata if copy else None
