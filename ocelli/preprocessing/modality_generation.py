import numpy as np
import anndata
from scipy.sparse import issparse
import scanpy as scp
import sys
import os


def modality_generation(adata: anndata.AnnData,
                        topic_key: str = 'lda',
                        n_top_vars: int = 100,
                        top_vars_key: str = 'top_vars',
                        norm_log: bool = True,
                        verbose: bool = False,
                        copy: bool = False):
    """Modality generation for unimodal data

    Modalities can be generated automatically using topic modeling components,
    which are stored in ``adata.varm[topic_key]`` in an array of shape
    ``(n_vars, n_topics)``.

    Firstly, variables (e.g. genes) are grouped into topics based
    on the highest scores in the LDA components array.
    For example, a gene with scores ``[0.5, 0.25, 0.25]`` will be assigned to the first topic.
    Next, variables are filtered - only ``n_top_vars`` variables with the highest scores in each topic are saved.
    For example, if ``n_top_vars = 100``, at most 100 variables
    from each topic are saved. If fewer than ``n_top_vars`` variables
    are assigned to a topic, none get filtered out.
    The resulting groups of variables form the newly-generated modalities
    (modalities with zero variables are ignored and not saved).
    Modalities are saved as :class:'numpy.ndarray' arrays in ``adata.obsm[modality*]``,
    where ``*`` denotes an id of a topic.

    Parameters
    ----------
    adata
        The annotated data matrix.
    topic_key
        ``adata.varm`` key storing topic components
        (:class:`numpy.ndarray` of shape ``(n_vars, n_topics)``). (default: `lda`)
    n_top_vars
        The maximum number of top variables considered for each topic.
        These are variables with highest LDA component scores. (default: 100)
    top_vars_key
        Top topic variables are saved to ``adata.uns[top_vars_key]``. (default: `top_vars`)
    verbose
        Print progress notifications. (default: ``False``)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.uns[modalities]`` (:class:`list` with ``adata.obsm`` keys storing generated modalities,
        ``adata.obsm[modality*]`` (:class:`numpy.ndarray` arrays of shape ``(n_obs, n_var)``; ``*`` denotes a topic id),
        ``adata.uns[top_vars_key]`` (:class:`dict` storing ids of top variables from all topics).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    if topic_key not in list(adata.varm.keys()):
        raise (KeyError('No topic modeling components found. Run ocelli.pp.LDA.'))

    n_topics = adata.varm[topic_key].shape[1]
    
    topic_assignment = np.argmax(adata.varm[topic_key], axis=1)
    
    d_topic_assignment = dict()
    
    for i, t in enumerate(topic_assignment):
        if t in d_topic_assignment:
            d_topic_assignment[t].append(i)
        else:
            d_topic_assignment[t] = [i]
            
    modalities = np.unique(list(d_topic_assignment.keys()))
    
    for m in modalities:
        arg_sorted = np.argsort(adata.varm[topic_key][d_topic_assignment[m], m])[-n_top_vars:]
        d_topic_assignment[m] = np.asarray(d_topic_assignment[m])[arg_sorted]

    adata.uns[top_vars_key] = d_topic_assignment

    obsm_key = adata.uns['{}_params'.format(topic_key)]['output_key']
    adata.uns['modalities'] = list()
    
    obsm_key = None

    topic_counter = 0
    for m in modalities:
        v = adata.X[:, d_topic_assignment[m]] if obsm_key is None else adata.obsm[obsm_key][:, d_topic_assignment[m]]
        
        if issparse(v):
            v = v.toarray()
            
        if norm_log:
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
            print('Modality {} --> saved to adata.obsm[modality{}].'.format(m, m))

    if verbose:
        print('{} topic-based modalities generated.'.format(len(modalities)))

    return adata if copy else None
