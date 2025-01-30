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


def modality_weights(adata: anndata.AnnData,
                     modalities=None,
                     out: str = 'weights',
                     n_pairs: int = 1000,
                     n_jobs: int = -1,
                     random_state = None,
                     verbose: bool = False,
                     copy: bool = False):
    """
    Compute multimodal weights for each cell

    This function calculates cell-specific weights for each modality based on the distances 
    between cells in the corresponding modality spaces. The weights are normalized to emphasize 
    the contribution of each modality for each cell.

    .. note::
        It is necessary to run ``ocelli.pp.neighbors`` before using this function, as the computed 
        nearest neighbors are required for estimating multimodal weights.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param modalities: List of keys in `adata.obsm` storing modalities. If `None`, the list is taken 
        from `adata.uns['modalities']`. (default: `None`)
    :type modalities: list or None

    :param out: Key in `adata.obsm` where the computed multimodal weights are saved. (default: `'weights'`)
    :type out: str

    :param n_pairs: Number of cell pairs used to estimate empirical cumulative distribution functions 
        (ECDFs) of distances. (default: 1000)
    :type n_pairs: int

    :param n_jobs: Number of parallel jobs to use. If `-1`, all available CPUs are used. (default: `-1`)
    :type n_jobs: int

    :param random_state: Seed for reproducibility. (default: `None`)
    :type random_state: int or None

    :param verbose: Whether to print progress notifications. (default: `False`)
    :type verbose: bool

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the following field:
            - `adata.obsm[out]`: DataFrame containing cell-specific weights for each modality.
        - If `copy=True`: Returns a modified copy of `adata` with the weights.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np
            import pandas as pd

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm['modality1'] = np.random.rand(100, 10)
            adata.obsm['modality2'] = np.random.rand(100, 15)
            adata.uns['modalities'] = ['modality1', 'modality2']

            # Compute nearest neighbors
            oci.pp.neighbors(adata, x=['modality1', 'modality2'], n_neighbors=20)

            # Compute multimodal weights
            oci.tl.modality_weights(adata, n_pairs=500, n_jobs=4, verbose=True)
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
