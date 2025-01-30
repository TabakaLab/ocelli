import anndata as ad
from scipy.sparse import issparse, csr_matrix
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def imputation(adata: ad.AnnData, 
               t: int = 5,
               kappa: float = 1.,
               features: list = None, 
               eigvals: str = 'eigenvalues',
               eigvecs: str = 'eigenvectors',
               scale: float = 1.,
               copy: bool = False):
    """
    Diffusion-based multimodal imputation.

    This function performs iterative imputation of a count matrix (`adata.X`) using multimodal 
    eigenvectors and eigenvalues. The imputation amplifies signals while preserving or enhancing 
    the structure in the data.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param t: Number of imputation iterations. Higher values increase imputation smoothing. (default: 5)
    :type t: int

    :param kappa: Scaling factor for signal amplification while preserving maximum expression values. (default: 1.0)
    :type kappa: float

    :param features: List of feature names (`adata.var.index`) to be imputed. If `None`, all features are imputed. (default: `None`)
    :type features: list or None

    :param eigvals: Key in `adata.uns` storing eigenvalues. (default: `'eigenvalues'`)
    :type eigvals: str

    :param eigvecs: Key in `adata.uns` storing eigenvectors. (default: `'eigenvectors'`)
    :type eigvecs: str

    :param scale: Scaling factor for signal amplification. Higher values increase imputed signal and maximum expression values. (default: 1.0)
    :type scale: float

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the imputed count matrix in `adata.X`.
        - If `copy=True`: Returns a modified copy of `adata` with the imputed count matrix.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.var.index = [f"gene_{i}" for i in range(50)]

            # Add mock eigenvalues and eigenvectors
            adata.uns["eigenvalues"] = np.random.rand(10)
            adata.uns["eigenvectors"] = np.random.rand(100, 10)

            # Perform imputation
            oci.tl.imputation(adata, t=5, kappa=1.5, features=["gene_1", "gene_2"])
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
    if issparse(adata.X):
        adata.X.eliminate_zeros()
    
    return adata if copy else None
