import numpy as np
import anndata
from scipy.sparse import issparse

def modality_generation(adata: anndata.AnnData,
                        lda_key: str = 'lda',
                        n_top_vars: int = 100,
                        top_vars_key: str = 'top_vars',
                        verbose: bool = False,
                        copy: bool = False):
    """Modality generation for unimodal data

    Modalities can be generated automatically using topic modeling components,
    which are stored in ``adata.varm[lda_key]`` in an array of shape
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
    lda_key
        ``adata.varm[lda_key]`` stores LDA components
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

    if lda_key not in list(adata.varm.keys()):
        raise (KeyError('No topic modeling components found. Run ocelli.pp.LDA.'))

    n_topics = adata.uns['{}_params'.format(lda_key)]['n_components']
    D = {i: [] for i in range(n_topics)}

    for i, t in enumerate(np.argmax(adata.varm[lda_key], axis=1)):
        D[t].append(i)

    for i in range(n_topics):
        arg_sorted = np.argsort(adata.varm[lda_key][D[i], i])[-n_top_vars:]
        D[i] = np.asarray(D[i])[arg_sorted]

    adata.uns[top_vars_key] = D

    obsm_key = adata.uns['{}_params'.format(lda_key)]['output_key']
    adata.uns['modalities'] = list()

    topic_counter = 0
    for i in range(n_topics):
        v = adata.X[:, D[i]] if obsm_key is None else adata.obsm[obsm_key][:, D[i]]
        if issparse(v):
            v = v.toarray()

        if v.shape[1] > 0:
            topic_counter += 1
            adata.obsm['modality{}'.format(i)] = v
            adata.uns['modalities'].append('modality{}'.format(i))
            if verbose:
                print('Modality {}: saved to adata.obsm[modality{}].'.format(i, i))
        else:
            if verbose:
                print('Modality {}: skipped, no genes selected.'.format(i))

    if verbose:
        print('{} topic-based modalities generated.'.format(topic_counter))

    return adata if copy else None
