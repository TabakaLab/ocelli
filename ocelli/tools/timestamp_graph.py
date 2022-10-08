import anndata
import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing import cpu_count
from scipy.sparse import issparse
from tqdm import tqdm


def timestamp_graph(adata: anndata.AnnData,
                    transitions_key: str,
                    timestamps_key: str,
                    n_edges: int = 10,
                    neighbors_key: str = 'X_mdm',
                    graph_key: str = 'graph',
                    n_jobs: int = -1,
                    verbose: bool = False,
                    copy: bool = False):
    """Timestamp-based graph

    From each graph node, ``n`` edges come out. They correspond to cells' nearest neighbors
    with the highest cell transition probabilities. If in a cell's neighborhood there is less
    than ``n`` cells with non-zero cell transitions, the remaining edges are connected
    to the nearest neighbors in the Multimodal Diffusion Maps (MDM) space from the subsequent timestamp.
    
    Before constructing the graph, you must perform a nearest neighbors search in the embedding space. 
    To do so, run ``ocelli.pp.neighbors(adata, modalities=[neighbors_key])``,
    where ``neighbors_key`` is a :class:`str`, and ``adata.obsm[neighbors_key]`` stores the embedding.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    transitions_key
        Stores ``adata.uns`` key with a square cell transition probability matrix.
    timestamp_key
        Stores ``adata.obs`` key with cell timestamps. Timestamps must bu numerical.
    n_edges
        number of edges coming out of each node. (default: 10)
    neighbors_key
        Stores ``adata.obsm`` key used for calculating nearest neighbors. (default: `X_mdm`)
    graph_key
        The graph is saved to ``adata.uns[graph_key]``. (default: ``graph``)    
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

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

    if timestamps_key not in adata.obs:
        raise (KeyError('No timestamps found in adata.obs["{}"].'.format(timestamps_key)))

    if neighbors_key not in adata.obsm:
        raise (KeyError('No embedding found in adata.obsm["{}"].'.format(neighbors_key)))

    transition_activity = list()
    
    graph = [[] for _ in range(len(adata.obs.index))]

    for cell_id, cell_nn in enumerate(adata.obsm['neighbors_{}'.format(neighbors_key)]):
        cell_transitions = adata.uns[transitions_key][cell_id, cell_nn]
        cell_transitions = cell_transitions.toarray().flatten() if issparse(cell_transitions) else cell_transitions.flatten()
        
        thr = n_edges if np.count_nonzero(cell_transitions) >= n_edges else np.count_nonzero(cell_transitions)
        
        selected = list() if thr == 0 else list(cell_nn[np.argpartition(cell_transitions, kth=-thr)[-thr:]])

        transition_activity.append(thr)
        graph[cell_id] += selected

    timestamps = np.unique(adata.obs[timestamps_key])
    n_timestamps = timestamps.shape[0]
    adata.obs['temp'] = [i for i in range(adata.shape[0])]
    
    for tstamp_id, tstamp in tqdm(enumerate(timestamps)):
        if tstamp_id < n_timestamps - 1:
            t0 = adata[adata.obs[timestamps_key] == timestamps[tstamp_id]]
            t1 = adata[adata.obs[timestamps_key] == timestamps[tstamp_id + 1]]
            
            neigh = NearestNeighbors(n_neighbors=n_edges + 1, n_jobs=n_jobs)
            neigh.fit(t1.obsm[neighbors_key])

            for cell_id, cell_nn in enumerate(neigh.kneighbors(t0.obsm[neighbors_key])[1][:, 1:]):                
                graph[t0.obs['temp'][cell_id]] += list(t1.obs['temp'][cell_nn])[:n_edges - transition_activity[t0.obs['temp'][cell_id]]]

        else:
            for cell_id, cell_nn in enumerate(neigh.kneighbors(t1.obsm[neighbors_key])[1][:, 1:]):
                graph[t1.obs['temp'][cell_id]] += list(t1.obs['temp'][cell_nn])[:n_edges - transition_activity[t1.obs['temp'][cell_id]]]

    adata.obsm[graph_key] = np.asarray(graph)
    
    adata.obs.pop('temp')

    if verbose:
        print('Timestamp-based graph constructed.')
    
    return adata if copy else None
