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


def zscores(adata: ad.AnnData, 
            markers: list, 
            x: str = None, 
            vmin: float = -5., 
            vmax: float = 5., 
            out: str = 'mean_z_scores',
            n_jobs: int = -1,
            copy: bool = False):
    """
    Gene signature mean z-scores

    Computes mean z-scores for a specified gene signature based on marker indices. 
    The z-scores are scaled to fit within the range (`vmin`, `vmax`) before being averaged for each cell.

    .. note::
        Ensure that the count matrix has been normalized and log-transformed before using this function.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param markers: List of column indices corresponding to marker genes in `adata.X` or `adata.obsm[x]`.
    :type markers: list

    :param x: Key in `adata.obsm` storing the data matrix for z-score computation. If `None`, `adata.X` is used. (default: `None`)
    :type x: str or None

    :param vmin: Minimum value to clip the z-scores before averaging. (default: -5.0)
    :type vmin: float

    :param vmax: Maximum value to clip the z-scores before averaging. (default: 5.0)
    :type vmax: float

    :param out: Key in `adata.obs` where the mean z-scores will be stored. (default: `'mean_z_scores'`)
    :type out: str

    :param n_jobs: Number of parallel jobs to use. If `-1`, all available CPUs are used. (default: `-1`)
    :type n_jobs: int

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the mean z-scores stored in `adata.obs[out]`.
        - If `copy=True`: Returns a modified copy of `adata` with the mean z-scores.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))

            # Indices of marker genes
            markers = [0, 5, 10]

            # Compute mean z-scores
            oci.tl.zscores(adata, markers=markers, vmin=-3, vmax=3, out='signature_zscores')
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
