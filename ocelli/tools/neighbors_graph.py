import anndata as ad
import numpy as np


def neighbors_graph(adata: ad.AnnData,
                    x: str,
                    n_edges: int = 10,
                    out: str = 'graph',      
                    verbose: bool = False,
                    copy: bool = False):
    """Nearest neighbors-based graph
    
    Before constructing the graph, you must perform a nearest neighbors search 
    in the embedding space by running :class:`ocelli.pp.neighbors`.

    This function computes a nearest neighbors-based graph.
    Each graph node has `n_edges` edges coming out.
    They correspond to the respective cell's nearest neighbors.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x
        `adata.obsm` key for which nearest neighbors were precomputed.
    n_edges
        Number of edges coming out of each node. (default: 10)
    out
        `adata.obsm` key where graph is saved. (default: `graph`)
    verbose
        Print progress notifications. (default: `False`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: `False`)
        
    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields:
        ``adata.obsm[out]`` (graph).
    :class:`anndata.AnnData`
        When `copy=True` is set, a copy of `adata` with those fields is returned.
    """
    
    if 'neighbors_{}'.format(x) not in adata.obsm:
        raise(KeyError('No nearest neighbors found in adata.obsm[neighbors_{}]. Run ocelli.pp.neighbors.'.format(x)))
        
    adata.obsm[out] = np.asarray(adata.obsm['neighbors_{}'.format(x)][:, :n_edges])

    if verbose:
        print('Nearest neighbors-based graph constructed.')
    
    return adata if copy else None
