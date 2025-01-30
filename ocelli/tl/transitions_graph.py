import anndata as ad
import numpy as np
from scipy.sparse import issparse
from multiprocessing import cpu_count
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def transitions_graph(adata: ad.AnnData,
                      x: str,
                      transitions: str,
                      n_edges: int = 10,
                      timestamps: str = None,
                      out: str = 'graph',
                      n_jobs: int = -1,
                      verbose: bool = False,
                      copy: bool = False):
    """
    Transitions-based graph construction

    Constructs a transitions-based graph using transition probabilities between cells, such as RNA velocity.
    The graph connects each cell to its `n_edges` nearest neighbors with the highest transition probabilities 
    stored in `adata.uns[transitions]`. Optionally, cell timestamps can be used to constrain neighbors 
    to specific temporal steps.

    .. note::
        Before using this function, you must run `ocelli.pp.neighbors` to compute the nearest neighbors 
        for the specified embedding.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param x: Key in `adata.obsm` corresponding to the embedding for which nearest neighbors were computed.
    :type x: str

    :param transitions: Key in `adata.uns` storing the transition probability matrix (shape `(n_cells, n_cells)`).
    :type transitions: str

    :param n_edges: Number of edges (connections) per graph node. (default: 10)
    :type n_edges: int

    :param timestamps: Key in `adata.obs` storing numerical timestamps. If `None`, no temporal constraints are applied. (default: `None`)
    :type timestamps: str or None

    :param out: Key in `adata.obsm` where the constructed graph is saved. (default: `'graph'`)
    :type out: str

    :param n_jobs: Number of parallel jobs to use. If `-1`, all CPUs are used. (default: `-1`)
    :type n_jobs: int

    :param verbose: Whether to print progress notifications. (default: `False`)
    :type verbose: bool

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the constructed graph stored in `adata.obsm[out]`.
        - If `copy=True`: Returns a modified copy of `adata` with the constructed graph.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm['embedding'] = np.random.rand(100, 10)
            adata.uns['transitions'] = np.random.rand(100, 100)
            adata.obs['timestamps'] = np.random.choice([0, 1, 2], size=100)

            # Compute nearest neighbors
            oci.pp.neighbors(adata, x=['embedding'], n_neighbors=20)

            # Construct transitions-based graph
            oci.tl.transitions_graph(
                adata,
                x='embedding',
                transitions='transitions',
            #   timestamps='timestamps',
                n_edges=10,
                verbose=True
            )
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
