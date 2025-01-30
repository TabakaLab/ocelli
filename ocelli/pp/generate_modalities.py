import numpy as np
import pandas as pd
import anndata
from scipy.sparse import issparse
import scanpy as scp
import sys
import os


def generate_modalities(adata: anndata.AnnData,
                        topics: str = 'X_lda',
                        n_features: int = 100,
                        log_norm: bool = True,
                        weights: str = 'weights',
                        verbose: bool = False,
                        copy: bool = False):
    """
    Generate topic-based modalities from unimodal data.

    This function creates topic-based modalities using the topic-feature distribution from Latent Dirichlet Allocation (LDA).
    Features (e.g., genes) are grouped based on their highest topic assignment, and the top `n_features` 
    per group are retained to form new modalities. Modalities are saved as `numpy.ndarray` arrays 
    in `adata.obsm[modality*]`, where `*` denotes the topic ID. 

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param topics: Key in `adata.varm` storing the topic-feature distribution matrix 
        (`numpy.ndarray` of shape `(n_vars, n_topics)`). (default: `X_lda`)
    :type topics: str

    :param n_features: Maximum number of features to retain for each topic-based modality. (default: 100)
    :type n_features: int

    :param log_norm: Whether to log-normalize the generated modalities. (default: `True`)
    :type log_norm: bool

    :param weights: Key in `adata.obsm` where the topic-based weights matrix will be saved. 
        This matrix is row-normalized so each row sums to 1. (default: `weights`)
    :type weights: str

    :param verbose: Whether to print progress notifications. (default: `False`)
    :type verbose: bool

    :param copy: Whether to return a copy of `anndata.AnnData`. If `False`, modifies the input object in-place. (default: `False`)
    :type copy: bool

    :returns: 
        - If `copy=False`: Updates the input `adata` with:
            - `adata.uns["modalities"]`: List of generated modalities.
            - `adata.obsm[modality*]`: Arrays of generated modalities, where `*` denotes the topic ID.
            - `adata.uns[vars_*]`: Feature names used for each modality, where `*` denotes the topic ID.
            - `adata.obsm[weights]`: Topic-based weights matrix.
        - If `copy=True`: Returns a modified copy of `adata` with the above updates.
    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example AnnData object
            adata = AnnData(X=np.random.random((100, 50)))
            adata.varm["X_lda"] = np.random.random((50, 5))  # Mock LDA output

            # Generate modalities
            oci.pp.generate_modalities(adata, topics="X_lda", n_features=50, verbose=True)
    """

    if topics not in list(adata.varm.keys()):
        raise (KeyError('No topic modeling components found. Run ocelli.pp.LDA.'))

    n_topics = adata.varm[topics].shape[1]
    
    topic_assignment = np.argmax(adata.varm[topics], axis=1)
    
    d_topic_assignment = dict()
    
    for i, t in enumerate(topic_assignment):
        if t in d_topic_assignment:
            d_topic_assignment[t].append(i)
        else:
            d_topic_assignment[t] = [i]
            
    modalities = np.unique(list(d_topic_assignment.keys()))

    adata.obsm[weights] = pd.DataFrame(adata.obsm[topics][:, modalities]/adata.obsm[topics][:, modalities].sum(axis=1)[:,None],
                                       index=list(adata.obs.index), columns=['modality{}'.format(m) for m in modalities])
    
    for m in modalities:
        arg_sorted = np.argsort(adata.varm[topics][d_topic_assignment[m], m])[-n_features:]
        d_topic_assignment[m] = np.asarray(d_topic_assignment[m])[arg_sorted]

    obsm_key = adata.uns['{}_params'.format(topics)]['x'] if 'x' in adata.uns['{}_params'.format(topics)] else None
    
    adata.uns['modalities'] = list()

    topic_counter = 0
    for m in modalities:
        v = adata.X[:, d_topic_assignment[m]] if obsm_key is None else adata.obsm[obsm_key][:, d_topic_assignment[m]]
        
        v = v.toarray() if issparse(v) else v
            
        if log_norm:
            v = anndata.AnnData(v)
            
            scp.pp.normalize_total(v, target_sum=10000)
            scp.pp.log1p(v)
            
            v = v.X

        adata.obsm['modality{}'.format(m)] = v
        adata.uns['modalities'].append('modality{}'.format(m))
        if verbose:
            print('[modality{}]\tModality generated.'.format(m))
        adata.uns['vars_{}'.format(m)] = list(np.asarray(adata.var.index)[list(d_topic_assignment[m])])

    if verbose:
        print('{} topic-based modalities generated.'.format(len(modalities)))

    return adata if copy else None
