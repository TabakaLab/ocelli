import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import ray
from multiprocessing import cpu_count
import anndata
import pandas as pd


class WeightEstimator():
    """Multimodal cell weights class"""
    
    def __init__(self, n_jobs=cpu_count()):
        if not ray.is_initialized():
            ray.init(num_cpus=n_jobs)
        self.n_jobs = n_jobs
        

    @staticmethod
    @ray.remote
    def __weights_worker(modalities, nn, ecdfs, split, index):
        weights = list()
        for cell in split[index]:
            cell_scores = list()
            for v1_id, _ in enumerate(modalities):
                modality_scores = list()
                nn_ids = nn[v1_id][cell]
                for v2_id, v2 in enumerate(modalities):
                    if v1_id != v2_id:
                        try:
                            axis_distances = np.linalg.norm(v2[nn_ids].toarray() - v2[cell].toarray(), axis=1) 
                        except AttributeError:
                            axis_distances = np.linalg.norm(v2[nn_ids] - v2[cell], axis=1) 
                        modality_scores.append(ecdfs[v2_id](axis_distances))
                    else:
                        modality_scores.append(np.zeros(nn_ids.shape))
                cell_scores.append(modality_scores)
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


    def estimate(self, modalities, nn=None, n_pairs=1000):
        n_modalities = len(modalities)
        n_cells = modalities[0].shape[0]
        if n_modalities > 1:
    
            pairs = np.random.choice(range(n_cells), size=(n_pairs, 2))
            ecdfs = list()
            for v in modalities:
                modality_dists = list()
                for i, _ in enumerate(pairs):
                    try:
                        pair_dist = np.linalg.norm(v[pairs[i, 0]].toarray() - v[pairs[i, 1]].toarray(), axis=None) 
                    except AttributeError:
                        pair_dist = np.linalg.norm(v[pairs[i, 0]] - v[pairs[i, 1]], axis=None) 
                    modality_dists.append(pair_dist)
                ecdfs.append(ECDF(modality_dists))

            split = np.array_split(range(n_cells), self.n_jobs)

            modalities_ref = ray.put(modalities)
            nn_ref = ray.put(nn)
            ecdfs_ref = ray.put(ecdfs)
            split_ref = ray.put(split)

            weights = [self.__weights_worker.remote(modalities_ref, nn_ref, ecdfs_ref, split_ref, i) 
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

    
def weights(adata: anndata.AnnData, 
            n_pairs: int = 1000, 
            modalities = None,
            neighbors_key: str = 'neighbors',
            weights_key: str = 'weights',
            n_jobs: int = -1,
            random_state = None,
            verbose: bool = False,
            copy: bool = False):
    """Multimodal cell-specific weights
    
    Computes cell-specific weights for each modality.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_pairs
        Number of cell pairs used to estimate empirical cumulative
        distribution functions of intercellular distances. (default: 1000)
    modalities
        A list of ``adata.obsm`` keys storing modalities.
        If :obj:`None`, modalities' keys are loaded from ``adata.uns[modalities]``. (default: :obj:`None`)
    neighbors_key
        ``adata.uns[neighbors_key]`` stores the nearest neighbor indices
        (:class:`numpy.ndarray` of shape ``(n_modalities, n_cells, n_neighbors)``).
        (default: ``neighbors``)
    weights_key
        Weights will be saved to ``adata.obsm[weights_key]``. (default: `weights`)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)
    verbose
        Print progress notifications. (default: ``False``)
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
        raise(KeyError('No nearest neighbors found in adata.uns[{}]. Run ocelli.pp.neighbors.'.format(neighbors_key)))

    if modalities is None:
        if 'modalities' not in list(adata.uns.keys()):
            raise(NameError('No modality keys found in adata.uns["modalities"].'))
        modalities = adata.uns['modalities']
 
    if len(modalities) == 0:
        raise(NameError('No modality keys found in adata.uns["modalities"].'))
        
    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])
    
    if random_state is not None:
        np.random.seed(random_state)

    we = WeightEstimator(n_jobs=n_jobs)
    weights = we.estimate(modalities=[adata.obsm[key] for key in modalities], 
                          nn=adata.uns[neighbors_key], 
                          n_pairs=n_pairs)

    adata.obsm[weights_key] = pd.DataFrame(weights, index=adata.obs.index, columns=modalities)
    
    if verbose:
        print('Multimodal cell-specific weights estimated.')

    return adata if copy else None
