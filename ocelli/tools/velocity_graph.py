import anndata
import numpy as np
from scipy.sparse import issparse


def velocity_graph(adata: anndata.AnnData,
                   n_edges: int = 10,
                   neighbors_key: str = 'X_mdm',
                   transitions_key: str = 'velocity_graph',
                   graph_key: str = 'graph',
                   n_jobs: int = -1,
                   verbose: bool = False,
                   copy: bool = False):
    """RNA velocity-based graph

    From each graph node, ``n`` edges come out. They correspond to cells' nearest neighbors
    with the highest cell transition probabilities. If in a cell's neighborhood there is less
    than ``n`` cells with non-zero cell transitions, the remaining edges are connected
    to the nearest neighbors in the Multimodal Diffusion Maps (MDM) space. 
        
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
    transitions_key
        ``adata.uns[transitions_key]`` stores the cell transition probability square matrix.
        (default: ``velocity_graph``)
    graph_key
        The graph is saved to ``adata.obsm[graph_key]``. (default: `graph`)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
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

    df = list()
    for i, neighbors in enumerate(adata.obsm['neighbors_{}'.format(neighbors_key)]):
        velocities = adata.uns[transitions_key][i, neighbors]

        if issparse(velocities):
            velocities = velocities.toarray()

        velocities = velocities.flatten()

        thr = n_edges if np.count_nonzero(velocities) > n_edges else np.count_nonzero(velocities)

        selected = list() if thr == 0 else list(neighbors[np.argpartition(velocities, kth=-thr)[-thr:]])

        if len(selected) != n_edges:
            for _ in range(n_edges - thr):
                for idx in neighbors:
                    if idx not in selected:
                        selected.append(idx)
                        break
        df.append(selected)

    adata.obsm[graph_key] = np.asarray(df)

    if verbose:
        print('RNA velocity-based graph constructed.')
    
    return adata if copy else None
