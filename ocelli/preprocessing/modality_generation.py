import numpy as np
import pandas as pd
import anndata
from scipy.sparse import issparse
import scanpy as scp
import sys
import os


def modality_generation(adata: anndata.AnnData,
                        topics: str = 'X_lda',
                        n_features: int = 100,
                        log_norm: bool = True,
                        weights: str = 'weights',
                        verbose: bool = False,
                        copy: bool = False):
    """Generates modalities for unimodal data

    Generates topic-based modalities. Beforehand, compute topics using :class:`ocelli.pp.LDA`.

    In the first step, features (e.g., genes) are grouped based on LDA's topic-feature distribution values.
    For example, a gene distributed over three topics with parameters `[0.5, 0.25, 0.25]` will be assigned to the first topic.
    Next, features are filtered - only the `n_features` highest-scoring features per group are kept.
    The resulting groups of features form new modalities. Modalities with no features are ignored and not saved.
    Modalities are saved as :class:'numpy.ndarray' arrays in `adata.obsm[modality*]`,
    where `*` denotes an id of a topic.

    Parameters
    ----------
    adata
        The annotated data matrix.
    topics
        `adata.varm` key storing topic embedding (:class:`numpy.ndarray` of shape `(n_vars, n_topics)`). (default: `X_lda`)
    n_features
        The maximum number of variables per generated modality. (default: 100)
    log_norm
        If ``True``, generated modalities are log-normalized. (default: `True`)
    weights
        Topic-based weights are saved to `adata.obsm[weights]`.
        Matrix is row-normalized so that each row sums to 1. (default: `weights`)
    verbose
        Print progress notifications. (default: `False`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: `False`)

    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields: 
        `adata.uns["modalities"]` (list of generated modalities), 
        `adata.obsm[modality*]` (generated modalities, `*` denotes modality's topic id),
        `adata.uns[vars_*]` (feature names used when generating modalities, `*` denotes modality's topic id), 
        and `adata.obsm[weights]` (topic-based weights).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
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

    obsm_key = adata.uns['{}_params'.format(topics)]['x']
    adata.uns['modalities'] = list()

    topic_counter = 0
    for m in modalities:
        v = adata.X[:, d_topic_assignment[m]] if obsm_key is None else adata.obsm[obsm_key][:, d_topic_assignment[m]]
        
        if issparse(v):
            v = v.toarray()
            
        if log_norm:
            x = anndata.AnnData(v)
            
            old_stdout = sys.stdout # prevent any prints from scanpy
            sys.stdout = open(os.devnull, "w")
            
            scp.pp.normalize_total(x, target_sum=10000)
            scp.pp.log1p(x)
            
            sys.stdout = old_stdout
            
            v = x.X

        adata.obsm['modality{}'.format(m)] = v
        adata.uns['modalities'].append('modality{}'.format(m))
        if verbose:
            print('[modality{}]\tModality generated.'.format(m))
        adata.uns['vars_{}'.format(m)] = list(np.asarray(adata.var.index)[list(d_topic_assignment[m])])

    if verbose:
        print('{} topic-based modalities generated.'.format(len(modalities)))

    return adata if copy else None
