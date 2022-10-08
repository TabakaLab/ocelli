import anndata
import numpy as np

def neighbors_graph(adata: anndata.AnnData,
                    n_edges: int = 10,
                    neighbors_key: str = 'X_mdm',
                    graph_key: str = 'graph',      
                    verbose: bool = False,
                    copy: bool = False):
    """Nearest neighbors-based graph

    From each graph node, ``n_edges`` edges come out. They correspond to respective cell's nearest neighbors.
    
    Before constructing the graph, you must perform a nearest neighbors search in the embedding space. 
    To do so, run ``ocelli.pp.neighbors(adata, modalities=[neighbors_key])``,
    where ``neighbors_key`` is a :class:`str`, and ``adata.obsm[neighbors_key]`` stores the embedding.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    n_edges
        number of edges coming out of each node. (default: 10)
    neighbors_key
        Stores ``adata.obsm`` key used for calculating nearest neighbors. (default: `X_mdm`)
    graph_key
        The graph is saved to ``adata.obsm[graph_key]``. (default: `graph`)
    verbose
        Print progress notifications. (default: ``False``)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)
        
    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[graph_key]`` (:class:`numpy.ndarray`).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """
    
    if 'neighbors_{}'.format(neighbors_key) not in adata.obsm:
        raise(KeyError('No nearest neighbors found in adata.obsm[neighbors_{}]. Run ocelli.pp.neighbors.'.format(neighbors_key)))
        
    adata.obsm[graph_key] = np.asarray(adata.obsm['neighbors_{}'.format(neighbors_key)][:, :n_edges])

    if verbose:
        print('[{}] Nearest neighbors-based graph constructed.'.format(neighbors_key))
    
    return adata if copy else None
