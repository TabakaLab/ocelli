import anndata as ad
import numpy as np
from scipy.sparse import issparse


def transitions_graph(adata: ad.AnnData,
                      x: str,
                      transitions: str,
                      n_edges: int = 10,
                      timestamps: str = None,
                      out: str = 'graph',
                      n_jobs: int = -1,
                      verbose: bool = False,
                      copy: bool = False):
    """Transitions-based graph
    
    Before constructing the graph, you must perform a nearest neighbors search 
    in the embedding space by running :class:`ocelli.pp.neighbors`.
    
    This function computes a transitions-based graph using transition probabilities 
    between cells (e.g., RNA velocity), stored in `adata.uns[transitions]` 
    and, optionally, cell timestamps, stored in `adata.obs[timestamps]`. 
    Transitions should be stored as a square matrix, `n_cells` by `n_cells`.
    Each graph node has `n_edges` edges coming out. 
    They correspond to the cells' nearest neighbors
    with the highest cell transition probabilities.
    For example, if you calculated 50 nearest neighbors and set `n_edges=10`,
    the resulting graph will connect a cell to 10 cells among its 
    50 nearest neighbors with the highest transition probabilities.
    If in a cell's neighborhood, there are less than ``n_edges``
    cells with non-zero transitions, the remaining edges are connected 
    to nearest neighbors.
    
    Assume that cell is from timestamp `t`. If you specify `timestamps`,
    the nearest neighbors in the last step of the graph construction are selected among cells from timestamp `t+1`.
    Cells from the final timestamp are exceptions - neighbors are selected from the same timestamp.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    x
        `adata.obsm` key for which nearest neighbors were precomputed.
    transitions
        `adata.uns` key storing a cell transition probability square matrix (of shape `(n_cells, n_cells)`).
    n_edges
        Number of edges coming out of each node. (default: 10)
    timestamps
        `adata.obs` key storing cell timestamps. Timestamps must bu numerical. (default: :obj:`None`)
    out
        `adata.obsm` key where graph is saved. (default: `graph`)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    verbose
        Print progress notifications. (default: `False`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: `False`)
        
    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields:
        `adata.obsm[out]` (graph).
    :class:`anndata.AnnData`
        When `copy=True` is set, a copy of `adata` with those fields is returned.
    """

    if timestamps is None:
        df = list()
        for i, neighbors in enumerate(adata.obsm['neighbors_{}'.format(x)]):
            velocities = adata.uns[transitions][i, neighbors]

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

        adata.obsm[out] = np.asarray(df)


    else:
        n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

        if timestamps not in adata.obs:
            raise (KeyError('No timestamps found in adata.obs["{}"].'.format(timestamps)))

        if x not in adata.obsm:
            raise (KeyError('No embedding found in adata.obsm["{}"].'.format(x)))

        transition_activity = list()

        graph = [[] for _ in range(len(adata.obs.index))]

        for cell_id, cell_nn in enumerate(adata.obsm['neighbors_{}'.format(x)]):
            cell_transitions = adata.uns[transitions][cell_id, cell_nn]
            cell_transitions = cell_transitions.toarray().flatten() if issparse(cell_transitions) else cell_transitions.flatten()

            thr = n_edges if np.count_nonzero(cell_transitions) >= n_edges else np.count_nonzero(cell_transitions)

            selected = list() if thr == 0 else list(cell_nn[np.argpartition(cell_transitions, kth=-thr)[-thr:]])

            transition_activity.append(thr)
            graph[cell_id] += selected

        timestamps_unique = np.unique(adata.obs[timestamps])
        n_timestamps = timestamps_unique.shape[0]
        adata.obs['temp'] = [i for i in range(adata.shape[0])]

        for tstamp_id, tstamp in tqdm(enumerate(timestamps_unique)):
            if tstamp_id < n_timestamps - 1:
                t0 = adata[adata.obs[timestamps] == timestamps_unique[tstamp_id]]
                t1 = adata[adata.obs[timestamps] == timestamps_unique[tstamp_id + 1]]

                neigh = NearestNeighbors(n_neighbors=n_edges + 1, n_jobs=n_jobs)
                neigh.fit(t1.obsm[x])

                for cell_id, cell_nn in enumerate(neigh.kneighbors(t0.obsm[x])[1][:, 1:]):                
                    graph[t0.obs['temp'][cell_id]] += list(t1.obs['temp'][cell_nn])[:n_edges - transition_activity[t0.obs['temp'][cell_id]]]

            else:
                for cell_id, cell_nn in enumerate(neigh.kneighbors(t1.obsm[x])[1][:, 1:]):
                    graph[t1.obs['temp'][cell_id]] += list(t1.obs['temp'][cell_nn])[:n_edges - transition_activity[t1.obs['temp'][cell_id]]]

        adata.obsm[out] = np.asarray(graph)

        adata.obs.pop('temp')

    if verbose:
        print('Transitions-based graph constructed.')

    return adata if copy else None
