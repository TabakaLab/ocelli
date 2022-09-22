import anndata
import numpy as np

def nn_graph(adata: anndata.AnnData,
             n: int = 10,
             neighbors_key: str = 'neighbors_mdm',
             graph_key: str = 'graph',
             verbose: bool = False,
             copy: bool = False):
    """Nearest neighbors-based graph

    From each graph node, ``n`` edges come out. They correspond to respective cell's nearest neighbors.
    
    Before constructing the graph, you must perform a nearest neighbors search in the Multimodal Diffusion Maps space. 
    To do so, run ``ocelli.pp.neighbors(adata, views=X_mdm)``,
    where ``X_mdm`` is a :class:`str`, and ``adata.obsm[X_mdm]`` stores a Multimodal Diffusion Maps embedding.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    n
        number of edges coming out of each node. (default: 10)
    neighbors_key
        ``adata.uns[neighbors_key]`` stores nearest neighbors indices from
        the MDM space (:class:`numpy.ndarray` of shape ``(1, n_cells, n_neighbors)``). (default: ``neighbors_mdm``)
    graph_key
        The graph is saved to ``adata.obsm[graph_key]``. (default: ``graph``)
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
    
    if neighbors_key not in adata.uns:
        raise(KeyError('No nearest neighbors found in adata.uns[{}]. Run ocelli.pp.neighbors.'.format(neighbors_key)))
        
    adata.obsm[graph_key] = np.asarray(adata.uns[neighbors_key][0, :, :n])

    if verbose:
        print('Nearest neighbors-based graph constructed.')
    
    return adata if copy else None
