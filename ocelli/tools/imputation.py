import anndata as ad
from scipy.sparse import lil_matrix, diags
import numpy as np
from tqdm import tqdm
import warnings


# mute warnings concerning sparse matrices
warnings.filterwarnings('ignore')


def imputation(adata: ad.AnnData, 
               t: int = 5, 
               features: list = None, 
               eigvals: str = 'eigenvalues',
               eigvecs: str = 'eigenvectors',
               copy: bool = False):
    """Multimodal count matrix imputation
    
    Iteratively imputes a count matrix using the MDM affinity matrix.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    t
        A number of imputation iterations. (default: 5)
    features
        `adata.X` columns (indexed with `adata.var.index` names) that will be imputed.
        If :obj:`None`, all columns are taken. (default: :obj:`None`)
    eigvals
        `adata.uns` key storing eigenvalues. (default: `eigenvalues`)
    eigvecs
        `adata.uns` key storing eigenvectors. (default: `eigenvectors`)
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
    
    imputed = lil_matrix(adata.uns[eigvecs]).dot(diags(adata.uns[eigvals]**t)).dot(
        lil_matrix(adata.uns[eigvecs]).T.dot(adata[:, features].X))
    
    scaling_factors = adata[:, features].X.max(axis=0).toarray().flatten() / imputed.max(axis=0).toarray().flatten()
    adata[:, features].X = imputed.multiply(scaling_factors)

    return adata if copy else None
