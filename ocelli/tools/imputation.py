import anndata as ad
from scipy.sparse import issparse, csr_matrix
import numpy as np
import warnings


# mute warnings concerning sparse matrices
warnings.filterwarnings('ignore')


def imputation(adata: ad.AnnData, 
               t: int = 5,
               kappa: float = 1.,
               features: list = None, 
               eigvals: str = 'eigenvalues',
               eigvecs: str = 'eigenvectors',
               scale: float = 1.,
               copy: bool = False):
    """Diffusion-based multimodal imputation
    
    Iteratively imputes a count matrix using the multimodal eigenvectors and eigenvalues.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    t
        A number of imputation iterations. (default: 5)
    kappa
        High values of `kappa` increase the imputed signal 
        while preserving the maximum expression value. (default: 1)
    features
        `adata.X` columns (indexed with `adata.var.index` names) that will be imputed.
        If :obj:`None`, all columns are taken. (default: :obj:`None`)
    eigvals
        `adata.uns` key storing eigenvalues. (default: `eigenvalues`)
    eigvecs
        `adata.uns` key storing eigenvectors. (default: `eigenvectors`)
    scale
        High values of `scale` increase the imputed signal 
        together with the maximum expression value. (default: 1.)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: `False`)
        
    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields:
        `adata.X` (Imputed count matrix).
    :class:`anndata.AnnData`
        When `copy=True` is set, a copy of ``adata`` with those fields is returned.
    """
    
    if eigvals not in adata.uns.keys():
        raise(NameError('No eigenvalues found in adata.uns[\'{}\'].'.format(eigvals)))

    if eigvecs not in adata.uns.keys():
        raise(NameError('No eigenvalues found in adata.uns[\'{}\'].'.format(eigvecs)))

    features = list(adata.var.index) if features is None else list(features)
        
    eigvals_t = adata.uns[eigvals]**t
    imputed = (adata.uns[eigvecs] * eigvals_t) @ (adata.uns[eigvecs].T @ adata[:, features].X)
    
    max_values = adata[:, features].X.max(axis=0)
    if issparse(max_values):
        max_values = max_values.toarray().flatten()
    
    imputed_max = [x if x > 0 else 1 for x in imputed.max(axis=0)]
    scaling_factors = kappa * max_values / imputed_max
    imputed = scale * np.clip(imputed * scaling_factors, 0, max_values)

    adata[:, features].X = csr_matrix(imputed) if issparse(adata.X) else imputed
    adata.X.eliminate_zeros()
    
    return adata if copy else None
