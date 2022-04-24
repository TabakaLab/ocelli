from anndata import AnnData
import pandas as pd
import numpy as np
from scipy.stats import zscore
import ray
from multiprocessing import cpu_count


@ray.remote
def sig_worker(adata, 
               var_ids, 
               x_key):
    dataframe = pd.DataFrame([])

    for gene in var_ids:
        try:
            if x_key is None:
                try:
                    dataframe[gene] = zscore(adata[:, gene].X.toarray().flatten())
                except:
                    dataframe[gene] = zscore(adata[:, gene].X.flatten())
            else:
                try:
                    dataframe[gene] = zscore(adata.obsm[x_key][:, gene].flatten())
                except:
                    dataframe[gene] = zscore(adata.obsm[x_key][:, gene].toarray().flatten())
        except:
            pass

    return dataframe


def z_scores(adata: AnnData,
             x_key = None,
             markers: list = [],
             vmax: float = 3.,
             vmin: float = -3.,
             signature_key: str = 'signature',
             n_jobs = -1,
             copy: bool = False):
    """Signature mean z-scores
    
    Computes z-scores for found markers from 
    a signature (given as a :class:`list` of markers).
    Z-scores are then averaged for each cell independently over found  markers.

    Parameters
    ----------
    adata
        The annotated data matrix.
    x_key
        ``adata.obsm[x_key]`` stores a count matrix (e.g. a gene expression matrix).
        If :obj:`None`, ``adata.X`` is used. (default: :obj:`None`)
    markers
        :class:`list` of markers that compose the signature. 
        These are column indices of ``adata.obsm[x_key]``. (default: ``[]``)
    vmax
        All z-scores above ``vmax`` are changed to ``vmax``. (default: 3)
    vmin
        All z-scores below ``vmin`` are changed to ``vmin``. (default: -3)
    signature_key
        Signature z-scores will be saved saved to ``adata.obs[signature_key]``. (default: ``signature``)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obs[signature_key]`` (:class:`numpy.ndarray` of shape ``(n_cells,)``
        storing signature mean z-scores).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """
    
    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])
    
    if len(markers) == 0:
        raise(AssertionError('No markers defined.'))

    if x_key is not None:
        if x_key not in list(adata.obsm.keys()):
            raise(KeyError('No matrix found.'))

    if not ray.is_initialized():
        ray.init(num_cpus=n_jobs)

    adata_ref = ray.put(adata)
    split = np.array_split(range(len(markers)), n_jobs)

    df = ray.get([sig_worker.remote(adata_ref, np.asarray(markers)[split[i]], x_key) for i in range(n_jobs)])
    df = pd.concat(df, axis=1)
    df.index = adata.obs.index
    
    z_scores = np.asarray(df.mean(axis=1))
    
    for i, val in enumerate(z_scores):
        if val < vmin:
            z_scores[i] = vmin
        elif val > vmax:
            z_scores[i] = vmax

    adata.obs[signature_key] = z_scores

    print('Mean z-scores calculated. {} out of {} signature markers were used.'.format(len(df.columns), len(markers)))
    
    return adata if copy else None
