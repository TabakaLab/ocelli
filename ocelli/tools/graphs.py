from anndata import AnnData
import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing import cpu_count

def nn_graph(adata: AnnData,
                    n: int = 10,
                    neighbors_key: str = 'neighbors_mvdm',
                    graph_key: str = 'graph',
                    copy: bool = False):
    """Nearest neighbors-based graph

    From each graph node ``n`` edges come out. They correspond to respective cell's nearest neighbors.
    
    Before constructing the graph, you must perform a nearest neighbors search in the multi-view diffusion maps space. 
    To do so, run ``oci.pp.neighbors(adata, view_keys=x_mvdm)``,
    where ``x_mvdm`` is a :class:`str`, and ``adata.obsm[x_mvdm]`` stores a multi-view diffusion maps embedding.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    n
        The number of edges coming out of each node. (default: 10)
    neighbors_key
        ``adata.uns[neighbors_key]`` stores the nearest neighbors indices from
        the MVDM space (:class:`numpy.ndarray` of shape ``(1, n_cells, n_neighbors)``). (default: ``neighbors_mvdm``)
    graph_key
        The graph is saved to ``adata.obsm[graph_key]``. (default: ``graph``)
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
        raise(KeyError('No nearest neighbors found in adata.uns["{}"]. Run oci.pp.neighbors.'.format(neighbors_key)))
        
    adata.obsm[graph_key] = np.asarray(adata.uns[neighbors_key][0, :, :n])

    print('Nearest neighbors-based graph constructed.')
    
    return adata if copy else None

def vel_graph(adata: AnnData,
              n: int = 10,
              neighbors_key: str = 'neighbors_mvdm',
              cell_transitions_key: str = 'velocity_graph',
              graph_key: str = 'graph',
              use_timestamps: bool = False,
              timestamps_key: str = 'timestamps',
              x_key: str = 'x_mvdm',
              n_jobs: int = -1,
              copy: bool = False):
    """RNA velocity-based graph

    From each graph node ``n`` edges come out. They correspond to cells' nearest neighbors
    with the highest cell transition probabilities. If in a cell's neighborhood there is less
    than ``n`` cells with non-zero cell transitions, the remaining edges are connected
    to the nearest neighbors in the multi-view diffusion maps space. 
    
    If ``use_timestamps = True``, the remaining edges are connected to the nearest neighbors in the multi-view diffusion maps space
    that have the subsequent timestamp. By default, timestamps are not utilized.
    
    Before constructing the graph, you must perform a nearest neighbors search in the multi-view diffusion maps space. 
    To do so, run ``oci.pp.neighbors(adata, view_keys=x_mvdm)``,
    where ``x_mvdm`` is a :class:`str`, and ``adata.obsm[x_mvdm]`` stores a multi-view diffusion maps embedding.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    n
        The number of edges coming out of each node. (default: 10)
    neighbors_key
        ``adata.uns[neighbors_key]`` stores the nearest neighbors indices
        from the MVDM space (:class:`numpy.ndarray` of shape ``(1, n_cells, n_neighbors)``). (default: ``neighbors_mvdm``)
    cell_transitions_key
        ``adata.uns[transitions_key]`` stores the cell transition probability square matrix.
        (default: ``velocity_graph``)
    graph_key
        The graph is saved to ``adata.uns[graph_key]``. (default: ``graph``)
    use_timestamps
        If ``True``, timestamps are used when constructing the graph.
    timestamps_key
        Used only if ``use_timestamps = True``. ``adata.obs[timestamps_key]`` stores cell timestamps. (default: ``timestamps``)
    x_key
         Used only if ``use_timestamps = True``. ``adata.obsm[x_key]`` stores the MVDM embedding.
         It is used for calculating nearest neighbors. (default: ``x_mvdm``)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
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
        raise(KeyError('No nearest neighbors found in adata.uns["{}"]. Run oci.pp.neighbors on MVDM embeddings.'.format(neighbors_key)))
    
    if cell_transitions_key not in adata.uns:
        raise(KeyError('No velocity transitions found in adata.uns["{}"].'.format(cell_transitions_key)))

    if use_timestamps:
        n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

        if timestamps_key not in adata.obs:
            raise (KeyError('No timestamps found in adata.obs["{}"].'.format(timestamps_key)))

        if x_key not in adata.obsm:
            raise (KeyError('No embeddings found in adata.obsm["{}"].'.format(x_key)))

        velocity_activity = list()
        adata.obs['id'] = np.asarray([i for i in range(len(adata.obs.index))])
        df = [[] for _ in range(len(adata.obs.index))]

        for i, el in enumerate(adata.uns[neighbors_key][0]):
            velocities = adata.uns[cell_transitions_key][i, el].toarray().flatten()
            thr = n if np.count_nonzero(velocities) > n else np.count_nonzero(velocities)
            if thr == 0:
                selected = list()
            else:
                selected = list(el[np.argpartition(velocities, kth=-thr)[-thr:]])
            velocity_activity.append(thr)
            df[i] += selected

        tstamps = np.unique(adata.obs[timestamps_key])
        for i, cl in enumerate(tstamps):
            try:
                y_i1 = adata[adata.obs[timestamps_key] == tstamps[i + 1]]
                y_i0 = adata[adata.obs[timestamps_key] == tstamps[i]]
                neigh = NearestNeighbors(n_neighbors=n + 1, n_jobs=n_jobs)
                neigh.fit(y_i1.obsm[x_key])
                nn_ids = neigh.kneighbors(y_i0.obsm[x_key])[1][:, 1:]

                for j, el in enumerate(nn_ids):
                    df[y_i0.obs['id'][j]] += list(y_i1.obs['id'][el])[:n - velocity_activity[y_i0.obs['id'][j]]]

            except IndexError:
                y_i0 = adata[adata.obs[timestamps_key] == tstamps[i]]
                neigh = NearestNeighbors(n_neighbors=n + 1, n_jobs=n_jobs)
                neigh.fit(y_i0.obsm[x_key])
                nn_ids = neigh.kneighbors(y_i0.obsm[x_key])[1][:, 1:]

                for j, el in enumerate(nn_ids):
                    df[y_i0.obs['id'][j]] += list(y_i0.obs['id'][el])[:n - velocity_activity[y_i0.obs['id'][j]]]
    else:
        df = list()
        for i, el in enumerate(adata.uns[neighbors_key][0]):
            velocities = adata.uns[cell_transitions_key][i, el].toarray().flatten()
            thr = n if np.count_nonzero(velocities) > n else np.count_nonzero(velocities)
            if thr == 0:
                selected = list()
            else:
                selected = list(el[np.argpartition(velocities, kth=-thr)[-thr:]])
            if len(selected) != n:
                for _ in range(n - thr):
                    for idx in el:
                        if idx not in selected:
                            selected.append(idx)
                            break
            df.append(selected)

    adata.obsm[graph_key] = np.asarray(df)

    print('RNA velocity-based graph constructed.')
    
    return adata if copy else None
