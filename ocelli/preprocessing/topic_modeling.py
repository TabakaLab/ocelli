from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import cpu_count
import numpy as np
from anndata import AnnData


def LDA(adata: AnnData,
                                x_key = None,
                                lda_key: str = 'lda',
                                n_topics: int = 10,
                                max_iter: int = 20,
                                verbose: int = 0,
                                n_jobs: int = -1,
                                copy: bool = False):
    """Latent Dirichlet Allocation

    Latent Dirichlet Allocation (LDA) is generative statistical model
    performing a topic modeling procedure.
    :class:`sklearn.decomposition.LatentDirichletAllocation` implementation is adapted here.

    Parameters
    ----------
    adata
        The annotated data matrix.
    x_key
        ``adata.obsm[x_key]`` stores a matrix for topic modeling (e.g. a gene expression matrix; matrix elements can not be negative).
        If :obj:`None`, ``adata.X`` is used. (default: :obj:`None`)
    n_topics
        The number of LDA topics. (default: 10)
    max_iter
        The number of LDA iterations. (default: 20)
    verbose
        The LDA verbosity level. (default: 0)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[lda_key]`` (:class:`numpy.ndarray` of shape ``(n_cells, n_topics)`` storing LDA topic components),
        ``adata.varm[lda_key]`` (:class:`numpy.ndarray` of shape ``(n_vars, n_topics)`` storing LDA topic scores),
        ``adata.uns[lda_key_params]`` (:class:`dict` storing LDA parameters).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

    lda = LatentDirichletAllocation(n_components=n_topics, n_jobs=n_jobs, doc_topic_prior=50 / n_topics,
                                    topic_word_prior=0.1, max_iter=max_iter, verbose=verbose)

    if x_key is None:
        adata.obsm[lda_key] = lda.fit_transform(adata.X)
    else:
        if x_key not in list(adata.obsm.keys()):
            raise (KeyError('No matrix found for LDA.'))

        adata.obsm[lda_key] = lda.fit_transform(adata.obsm[x_key])

    adata.varm[lda_key] = lda.components_.T

    adata.uns['{}_params'.format(lda_key)] = {'id': lda_key,
                                              'n_topics': n_topics,
                                              'n_jobs': n_jobs,
                                              'doc_topic_prior': 50 / n_topics,
                                              'topic_word_prior': 0.1,
                                              'max_iter': max_iter,
                                              'verbose': verbose,
                                              'x_key': x_key,
                                              'random_state': lda.random_state_}

    print('{} topics calculated.'.format(n_topics))

    return adata if copy else None


def generate_views(adata: AnnData,
                   lda_key: str = 'lda',
                   n_top_vars: int = 100,
                   top_vars_key: str = 'top_vars',
                   copy: bool = False):
    """Automatic view generation from topics

    Views can be generated automatically using LDA topic components,
    which are stored in ``adata.varm[lda_key]`` in an array of shape
    ``(n_vars, n_topics)``.

    Firstly, variables (e.g. genes) are grouped into topics based
    on the highest scores in the LDA components array.
    For example, a gene with scores ``[0.5, 0.25, 0.25]`` will be assigned to the first topic.
    Next, variables are filtered - only ``n_top_vars`` variables with the highest scores in each topic are saved.
    For example, if ``n_top_vars = 100``, at most 100 variables
    from each topic  will be saved. If fewer than ``n_top_vars`` variables
    are assigned to a topic, none get filtered out.
    The resulting groups of variables form the newly-generated views (views with zero variables are ignored and not saved).
    Views are then normalized, logarithmized, and saved as :class:'numpy.ndarray' arrays in ``adata.obsm["view*"]``,
    where ``*`` denotes an id of a topic.

    Parameters
    ----------
    adata
        The annotated data matrix.
    lda_key
        ``adata.varm[lda_key]`` stores LDA topic scores
        (:class:`numpy.ndarray` of shape ``(n_vars, n_topics)``). (default: ``lda``)
    n_top_vars
        The maximum number of top variables considered for each topic.
        These are variables with highest scores. (default: 100)
    top_vars_key
        Top topic variables are saved to ``adata.uns[top_vars_key]``. (default: ``top_vars_key``)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.uns[view_keys]`` (:class:`list` with ``adata.obsm`` keys storing generated views,
        ``adata.obsm[view*]`` (:class:`numpy.ndarray` of shape ``(n_cells, n_view*_vars)``; ``*`` denotes a topic id),
        ``adata.uns[top_vars_key]`` (:class:`dict` storing ids of top variables from all topics).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    if lda_key not in list(adata.varm.keys()):
        raise (KeyError('No LDA components found. Run oci.pp.latent_dirichlet_allocation.'))

    n_topics = adata.uns['{}_params'.format(lda_key)]['n_topics']
    D = {i: [] for i in range(n_topics)}

    for i, t in enumerate(np.argmax(adata.varm[lda_key], axis=1)):
        D[t].append(i)

    for i in range(n_topics):
        arg_sorted = np.argsort(adata.varm[lda_key][D[i], i])[-n_top_vars:]
        D[i] = np.asarray(D[i])[arg_sorted]

    adata.uns[top_vars_key] = D

    x_key = adata.uns['{}_params'.format(lda_key)]['x_key']
    adata.uns['view_keys'] = list()

    topic_counter = 0
    for i in range(n_topics):
        if x_key is None:
            try:
                v = adata.X[:, D[i]].toarray()
            except AttributeError:
                v = adata.X[:, D[i]]
        else:
            try:
                v = adata.obsm[x_key][:, D[i]].toarray()
            except AttributeError:
                v = adata.obsm[x_key][:, D[i]]

        v_sum = v.sum(axis=1, keepdims=True)

        for j in range(v.shape[0]):
            if v_sum[j] != 0:
                v[j] = v[j] / v_sum[j]

        v = np.log1p(v)

        if v.shape[1] > 0:
            topic_counter += 1
            adata.obsm['view{}'.format(i)] = v
            adata.uns['view_keys'].append('view{}'.format(i))
        else:
            print('View {} skipped - no genes selected.'.format(i))

    print('{} topic-based views generated and normalized.'.format(topic_counter))

    return adata if copy else None
