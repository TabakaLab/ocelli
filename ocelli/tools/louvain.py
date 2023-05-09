import scanpy as sc


def louvain(adata, 
            x: str = None, 
            out: str = 'louvain',
            n_neighbors: int = 20, 
            resolution: float = 1.,
            random_state = None,
            copy: bool = False):
    """Louvain clustering
    
    Calculates Louvain clusters. It is a wrapper of :obj:`scanpy.tl.louvain`.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x
        `adata.obsm` key. If :obj:`None`, `adata.X` is clustered. (default: :obj:`None`)
    out
        `adata.obs` key under which clusters are saved. (default: `louvain`)
    n_neighbors
        Number of nearest neighbors. (default: 20)
    resolution
        Resolution of Louvain clustering. Larger `resolution` translates to more clusters. (default: 1.)
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)
        
    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obs[out]`` (Louvain cluster labels).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=x)
    sc.tl.louvain(adata, resolution=resolution, random_state=random_state, key_added=out)
    
    return adata if copy else None
