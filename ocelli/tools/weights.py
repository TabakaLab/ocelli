import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import ray
from multiprocessing import cpu_count
import anndata
import pandas as pd
from scipy.sparse import issparse


@ray.remote
def weights_worker(modalities, nn, ecdfs, split):
    w = list()
    for cell in split:
        cell_scores = list()
        for m0, _ in enumerate(modalities):
            modality_scores = list()
            nn_ids = nn[m0][cell]
            for m1, M1 in enumerate(modalities):
                if m0 != m1:
                    if issparse(M1):
                        axis_distances = np.linalg.norm(M1[nn_ids].toarray() - M1[cell].toarray(), axis=1)
                    else:
                        axis_distances = np.linalg.norm(M1[nn_ids] - M1[cell], axis=1)
                    modality_scores.append(ecdfs[m1](axis_distances))
                else:
                    modality_scores.append(np.zeros(nn_ids.shape))
            cell_scores.append(modality_scores)
        w.append(cell_scores)

    w = np.sum(np.median(np.asarray(w), axis=3), axis=1)

    return w


@ray.remote
def scaling_worker(w, nn, split, alpha=10):
    weights_scaled = np.asarray([np.mean(w[nn[np.argmax(w[obs])][obs], :], axis=0) for obs in split])

    for i, row in enumerate(weights_scaled):
        if np.max(row) != 0:
            weights_scaled[i] = row / np.max(row)
        row_exp = np.exp(weights_scaled[i]) ** alpha
        weights_scaled[i] = row_exp / np.sum(row_exp)

    return weights_scaled


def weights(adata: anndata.AnnData,
            modalities=None,
            out: str = 'weights',
            n_pairs: int = 1000,
            n_jobs: int = -1,
            random_state = None,
            verbose: bool = False,
            copy: bool = False):
    """Multimodal weights
    
    Computes cell-specific weights for each modality.

    Parameters
    ----------
    adata
        The annotated data matrix.
    modalities
        List of `adata.obsm` keys storing modalities.
        If :obj:`None`, modalities' keys are loaded from `adata.uns[modalities]`. (default: :obj:`None`)
    out
       `adata.obsm` key where multimodal weights are saved. (default: `weights`)
    n_pairs
        Number of cell pairs used to estimate empirical cumulative
        distribution functions of distances between cells. (default: 1000)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)
    verbose
        Print progress notifications. (default: `False`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: `False`)

    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields:
        `adata.obsm[out]` (multimodal weights).
    :class:`anndata.AnnData`
        When `copy=True` is set, a copy of `adata` with those fields is returned.
    """
    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

    if not ray.is_initialized():
        ray.init(num_cpus=n_jobs)

    modality_names = adata.uns['modalities'] if modalities is None else modalities

    if random_state is not None:
        np.random.seed(random_state)

    n_modalities = len(modality_names)
    modalities = [adata.obsm[m].toarray() if issparse(adata.obsm[m]) else adata.obsm[m] for m in modality_names]

    n_obs = adata.shape[0]

    if n_modalities > 1:
        pairs = np.random.choice(range(n_obs), size=(n_pairs, 2))
        ecdfs = list()
        for m in modalities:
            modality_dists = [np.linalg.norm(m[pairs[i, 0]] - m[pairs[i, 1]], axis=None) for i in range(n_pairs)]
            ecdfs.append(ECDF(modality_dists))

        splits = np.array_split(range(n_obs), n_jobs)
        modalities_ref = ray.put(modalities)
        nn_ref = ray.put([adata.obsm['neighbors_{}'.format(m)] for m in modality_names])
        ecdfs_ref = ray.put(ecdfs)
        weights = ray.get([weights_worker.remote(modalities_ref, nn_ref, ecdfs_ref, split) for split in splits])
        weights = np.vstack(weights)

        weights_ref = ray.put(weights)
        weights = ray.get([scaling_worker.remote(weights_ref, nn_ref, split) for split in splits])
        weights = np.concatenate(weights, axis=0)
    else:
        weights = np.ones((n_obs, 1))

    adata.obsm[out] = pd.DataFrame(weights, index=list(adata.obs.index), columns=modality_names)

    if verbose:
        print('Multimodal weights estimated.')

    if ray.is_initialized():
        ray.shutdown()

    return adata if copy else None
