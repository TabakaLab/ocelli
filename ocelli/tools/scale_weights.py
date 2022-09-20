import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import ray
from multiprocessing import cpu_count
from anndata import AnnData
import pandas as pd


class WeightEstimator():
    """The multi-view cell weights class"""
    
    def __init__(self, n_jobs=cpu_count()):
        if not ray.is_initialized():
            ray.init(num_cpus=n_jobs)
        self.n_jobs = n_jobs
        

    @staticmethod
    @ray.remote
    def __weights_worker(views, nn, ecdfs, split, index):
        weights = list()
        for cell in split[index]:
            cell_scores = list()
            for v1_id, _ in enumerate(views):
                view_scores = list()
                nn_ids = nn[v1_id][cell]
                for v2_id, v2 in enumerate(views):
                    if v1_id != v2_id:
                        try:
                            axis_distances = np.linalg.norm(v2[nn_ids].toarray() - v2[cell].toarray(), axis=1) 
                        except AttributeError:
                            axis_distances = np.linalg.norm(v2[nn_ids] - v2[cell], axis=1) 
                        view_scores.append(ecdfs[v2_id](axis_distances))
                    else:
                        view_scores.append(np.zeros(nn_ids.shape))
                cell_scores.append(view_scores)
            weights.append(cell_scores)

        weights = np.asarray(weights)
        weights = np.median(weights, axis=3)
        weights = np.sum(weights, axis=1)

        return weights


    @staticmethod
    @ray.remote
    def __weights_scaler_worker(weights, nn, split, index, alpha=10):
        scaled = np.zeros((len(split[index]), weights.shape[1]))
        for i in split[index]:
            scaled[i - split[index][0]] = np.mean(weights[nn[np.argmax(weights[i])][i], :], axis=0)
            
        for i, row in enumerate(scaled):
            if np.max(row) != 0:
                scaled[i] = row / np.max(row)
            row_exp = np.exp(scaled[i])**alpha
            scaled[i] = row_exp / np.sum(row_exp)
            
        return scaled


    def estimate(self, views, nn=None, n_pairs=1000):
        n_views = len(views)
        n_cells = views[0].shape[0]
        if n_views > 1:
    
            pairs = np.random.choice(range(n_cells), size=(n_pairs, 2))
            ecdfs = list()
            for v in views:
                view_dists = list()
                for i, _ in enumerate(pairs):
                    try:
                        pair_dist = np.linalg.norm(v[pairs[i, 0]].toarray() - v[pairs[i, 1]].toarray(), axis=None) 
                    except AttributeError:
                        pair_dist = np.linalg.norm(v[pairs[i, 0]] - v[pairs[i, 1]], axis=None) 
                    view_dists.append(pair_dist)
                ecdfs.append(ECDF(view_dists))

            split = np.array_split(range(n_cells), self.n_jobs)

            views_ref = ray.put(views)
            nn_ref = ray.put(nn)
            ecdfs_ref = ray.put(ecdfs)
            split_ref = ray.put(split)

            weights = [self.__weights_worker.remote(views_ref, nn_ref, ecdfs_ref, split_ref, i) 
                       for i in range(self.n_jobs)]
            weights = ray.get(weights)
            weights = np.vstack(weights)
            weights_ref = ray.put(weights)

            weights_scaled = [self.__weights_scaler_worker.remote(weights_ref, nn_ref, split_ref, i) 
                              for i in range(self.n_jobs)]
            weights = np.concatenate(ray.get(weights_scaled), axis=0)

        else:
            weights = np.ones((n_cells, 1))

        return weights


def weights(adata: AnnData, 
            n_pairs: int = 1000, 
            view_keys = None,
            neighbors_key: str = 'neighbors',
            weights_key: str = 'weights', 
            n_jobs: int = -1,
            copy: bool = False):
    """Multimodal cell-specific weights
    
    For each cell view-specific weights are computed. 
    They estimate which views contribute to the development of cell's neighborhood.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_pairs
        The number of cell pairs used to estimate empirical cumulative
        distribution functions of distances between cells.
    view_keys
        If :obj:`None`, view keys are loaded from ``adata.uns['key_views']``. Otherwise,
        ``view_keys`` should be a :class:`list` of ``adata.obsm`` keys,
        where views are stored. (default: :obj:`None`)
    neighbors_key
        ``adata.uns[neighbors_key]`` stores the nearest neighbor indices 
        (:class:`numpy.ndarray` of shape ``(n_views, n_cells, n_neighbors)``).
        (default: ``neighbors``)
    weights_key
        The multi-view cell weights will be saved to ``adata.obsm[weights_key]``. (default: ``weights``)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[weights_key]`` (:class:`pandas.DataFrame`).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    if neighbors_key not in adata.uns:
        raise(KeyError('No nearest neighbors found in adata.uns[{}]. Run oci.pp.neighbors.'.format(neighbors_key)))

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

    if view_keys is None:
        if 'view_keys' not in list(adata.uns.keys()):
            raise(NameError('No view keys found in adata.uns["view_keys"].'))
        view_keys = adata.uns['view_keys']
 
    if len(view_keys) == 0:
        raise(NameError('No view keys found in adata.uns["view_keys"].'))

    we = WeightEstimator(n_jobs=n_jobs)
    weights = we.estimate(views = [adata.obsm[key] for key in view_keys], 
                          nn = adata.uns[neighbors_key], 
                          n_pairs = n_pairs)

    adata.obsm[weights_key] = pd.DataFrame(weights, index=adata.obs.index, columns=view_keys)
    
    print('Multi-view weights estimated.')

    return adata if copy else None

def scale_weights(adata: AnnData,
                  weights_key: str = 'weights',
                  observations: list = [],
                  views: list = [],
                  kappa: float = 1.,
                  copy: bool = False):
    """Weight scaling
    
    Weights of selected cells and views are scaled by the factor of ``kappa``.
    If you wish to increase the impact of certain views for some cells, 
    select them and increase ``kappa``.
    
    When selecting views (``views``) and cells (``observations``), pay attention to data types of 
    ``adata.obsm[weights_key].index`` and ``adata.obsm[weights_key].columns``. 
    Your input must match these types.

    Parameters
    ----------
    adata
        The annotated data matrix.
    weights_key
        ``adata.obsm[weights_key]`` stores weights. (default: ``weights``)
    observations
        ``adata.obsm[weights_key].index`` elements storing selected cells. (default: ``[]``)
    views
        ``adata.obsm[weights_key].columns`` elements storing selected views. (default: ``[]``)
    kappa
        The scaling factor. (default: 1)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[weights_key]`` (:class:`pandas.DataFrame`).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """
    
    if weights_key not in adata.obsm:
        raise (KeyError('No weights found in adata.uns["{}"]. Run oci.tl.weights.'.format(weights_key)))
        
    np_obs_ids = list()
    for i, el in enumerate(adata.obsm[weights_key].index):
        if el in observations:
            np_obs_ids.append(i)
    
    np_view_ids = list()
    for i, el in enumerate(adata.obsm[weights_key].columns):
        if el in views:
            np_view_ids.append(i)
    
    w = np.asarray(adata.obsm[weights_key])
    w[np.ix_(np.unique(np_obs_ids), np.unique(np_view_ids))] *= kappa
    adata.obsm[weights_key] = pd.DataFrame(w, 
                                           index=adata.obsm[weights_key].index, 
                                           columns=adata.obsm[weights_key].columns)
    
    print('Multi-view weights scaled.')
    
    return adata if copy else None
