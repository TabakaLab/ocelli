import anndata as ad
import numpy as np
from scipy.stats import zscore
import ray
from multiprocessing import cpu_count
from scipy.sparse import issparse


def scale(X, vmin, vmax):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] < vmin:
                X[i, j] = vmin
            elif X[i, j] > vmax:
                X[i, j] = vmax
    return X


@ray.remote
def worker(X):
    return zscore(X, nan_policy='omit')


def mean_z_scores(adata: ad.AnnData, 
                  markers: list, 
                  x: str = None, 
                  vmin: float = -3., 
                  vmax: float = 3., 
                  out: str = 'mean_z_scores',
                  n_jobs: int = -1,
                  copy: bool = False):
    """Gene signature mean z-scores
    
    Normalize and logarithmize the count matrix first.
    
    Computes z-scores for markers given as a :class:`list` of integer indices.
    These indices indicate which columns from `adata.X` or `adata.obsm[x]` are interpreted as markers.
    Z-scores are then fitted to `(vmin, vmax)` scale and subsequently averaged for each cell independently over markers.

    Parameters
    ----------
    adata
        The annotated data matrix.
    markers
        :class:`list` of integer indices of markers that compose the signature. 
        These are column indices of `adata.X` or `adata.obsm[x]`.
    x
        `adata.obsm` key storing log-normalized count matrix of features, including gene signature markers.
        If :obj:`None`, ``adata.X`` is used. (default: :obj:`None`)
    vmin
        Before averaging, z-scores below `vmin` are changed to `vmin`. (default: -3)
    vmax
        Before averaging, z-scores above `vmax` are changed to `vmax`. (default: 3)
    out
        `adata.obs` key where mean z-scores are saved. (default: `mean_z_scores`)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: `False`)

    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields:
        `adata.obs[out]` (mean z-scores).
    :class:`anndata.AnnData`
        When `copy=True` is set, a copy of `adata` with those fields is returned.
    """

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])
    
    if not ray.is_initialized():
        ray.init(num_cpus=n_jobs)
    
    X = adata.obsm[x] if x is not None else adata.X
        
    if issparse(X):
        X = X.toarray()

    output = np.nan_to_num(ray.get([worker.remote(X[:, marker]) for marker in markers]))
    output = scale(output, vmin, vmax)

    adata.obs[out] = np.mean(output, axis=0)
    
    if ray.is_initialized():
        ray.shutdown()

    return adata if copy else None
