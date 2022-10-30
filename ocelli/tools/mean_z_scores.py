from anndata import AnnData
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


def mean_z_scores(adata: AnnData, 
                  markers: list, 
                  obsm_key = None, 
                  vmin: float = -3., 
                  vmax: float = 3., 
                  output_key: str = 'mean_z_scores',
                  n_jobs: int = -1,
                  copy: bool = False):
    """Marker mean z-scores
    
    Normalize and logarithmize count matrix first.
    
    Computes z-scores for markers given as a :class:`list` of integer indices.
    These indices indicate which columns from ``adata.X`` or ``adata.obsm[obsm_key]`` are interpreted as markers.
    Z-scores are then fitted to ``[vmin, vmax]`` scale and subsequently averaged for each cell independently over markers.

    Parameters
    ----------
    adata
        The annotated data matrix.
    markers
        :class:`list` of integer indices of markers that compose the signature. 
        These are column indices of ``adata.X`` or ``adata.obsm[obsm_key]``.
    obsm_key
        ``adata.obsm[obsm_key]`` stores a count matrix.
        If :obj:`None`, ``adata.X`` is used. (default: :obj:`None`)
    vmin
        Counts from a logarithmized count matrix below ``vmin`` are changed to ``vmin``. (default: -3)
    vmax
        Counts from a logarithmized count matrix above ``vmax`` are changed to ``vmax``. (default: 3)
    output_key
        Calculated z-scores will be saved to ``adata.obs[output_key]``. (default: ``mean_z_scores``)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obs[output_key]`` (:class:`numpy.ndarray` of shape ``(n_cells,)``
        storing marker mean z-scores).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])
    
    if not ray.is_initialized():
        ray.init(num_cpus=n_jobs)
    
    X = adata.obsm[obsm_key] if obsm_key is not None else adata.X
        
    if issparse(X):
        X = X.toarray()

    output = np.nan_to_num(ray.get([worker.remote(X[:, marker]) for marker in markers]))
    output = scale(output, vmin, vmax)

    adata.obs[output_key] = np.mean(output, axis=0)
    
    if ray.is_initialized():
        ray.shutdown()

    return adata if copy else None