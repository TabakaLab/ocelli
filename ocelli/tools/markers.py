from anndata import AnnData
import numpy as np
from scipy.stats import zscore
import ray
from multiprocessing import cpu_count
from scipy.sparse import issparse


def scale(X, vmin, vmax):
    X = np.squeeze(X, 1)
    for i, val in enumerate(X):
        if val < vmin:
            X[i] = vmin
        elif val > vmax:
            X[i] = vmax
    return X

@ray.remote
def worker(adata, marker, vmin, vmax):
    try:
        x = adata[:, marker].X.toarray()
    except:
        x = adata[:, marker].X
    
    return zscore(scale(x, vmin, vmax), nan_policy='omit')

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
    Z-scores are then averaged for each cell independently over markers.

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
    
    if obsm_key is not None:
        X = adata.obsm[obsm_key]
    else:
        X = adata.X
        
    if issparse(X):
        X = X.toarray()
    
    X = X[:, markers]

    adata_ref = ray.put(adata)
    output = np.nan_to_num(ray.get([worker.remote(adata_ref, marker, vmin, vmax) for marker in markers]))

    adata.obs[output_key] = np.mean(output, axis=0)

    return adata if copy else None