from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import cpu_count
import anndata as ad
import pandas as pd


def LDA(adata: ad.AnnData,
        x: str = None,
        out: str = 'X_lda',
        n_components: int = 10,
        max_iter: int = 30,
        doc_topic_prior = None,
        topic_word_prior: float = 0.1,
        learning_method: str = 'batch',
        learning_decay: float = 0.7,
        learning_offset: float = 10.0,
        batch_size: int = 128,
        evaluate_every: int = -1,
        total_samples: int = 1000000,
        perp_tol: float = 0.1,
        mean_change_tol: float = 0.001,
        max_doc_update_iter: int = 100,
        verbose: int = 0,
        random_state = None,
        n_jobs: int = -1,
        copy: bool = False):
    """Latent Dirichlet Allocation

    Latent Dirichlet Allocation (LDA) is a generative probabilistic model for topic modeling.
    This function is a wrapper of :class:`sklearn.decomposition.LatentDirichletAllocation`.

    Parameters
    ----------
    adata
        The annotated data matrix.
    x
        `adata.obsm` key storing a matrix with non-negative values. If :obj:`None`, `adata.X` is modeled. (default: :obj:`None`)
    out
        LDA output key (`adata.obsm` and `adata.varm`). (default: `X_lda`)
    n_components
        Number of topics. (default: 10)
    max_iter
        The maximum number of passes over the training data (aka epochs). (default: 30)
    doc_topic_prior
        Prior of document topic distribution `theta`. If the value is None,
        defaults to `50 / n_components`. (default: :obj:`None`)
    topic_word_prior
        Prior of topic word distribution `beta`. (default: 0.1)
    learning_method
        Method used to update `_component`.
        In general, if the data size is large, the online update will be much
        faster than the batch update. Valid options: 
        `batch` (Batch variational Bayes method. Use all training data in
        each EM update. Old `components_` will be overwritten in each iteration.) 
        and `online` (Online variational Bayes method. In each EM update, use
        mini-batch of training data to update the `components_`
        variable incrementally. The learning rate is controlled by the
        `learning_decay` and the `learning_offset` parameters.) (default: `batch`)
    learning_decay
        It is a parameter that control learning rate in the online learning method. 
        The value should be set between (0.5, 1.0] to guarantee asymptotic convergence.
        In the literature, this is called kappa. (default: 0.7)
    learning_offset
        A (positive) parameter that downweights early iterations in online learning.
        It should be greater than 1.0. In the literature, this is called `tau_0`. (default: 10.)
    batch_size
        Number of documents to use in each EM iteration. Only used in online learning. (default: 128)
    evaluate_every
        How often to evaluate perplexity. Set it to 0 or negative number to not evaluate perplexity in training at all.
        Evaluating perplexity can help you check convergence in training process, but it will also increase total training time.
        Evaluating perplexity in every iteration might increase training time up to two-fold. (default: -1)
    total_samples
        Total number of documents. Only used in the partial_fit method. (default: 1000000)
    perp_tol
        Perplexity tolerance in batch learning. Only used when ``evaluate_every`` is greater than 0. (default: 0.1)
    mean_change_tol
        Stopping tolerance for updating document topic distribution in E-step. (default: 0.001)
    max_doc_update_iter
        Max number of iterations for updating document topic distribution in the E-step. (default: 100)
    verbose
        Verbosity level. Valid options: 0, 1, 2. (default: 0)
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: `False`)   

    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields: 
        `adata.obsm[out]` (LDA observation-topic distribution),
        `adata.varm[out]` (LDA feature-topic distribution),
        and `adata.uns[out_params]` (LDA model parameters).
    :class:`anndata.AnnData`
        When `copy=True` is set, a copy of `adata` with those fields is returned.
    """

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])
    
    doc_topic_prior = 50/n_components if doc_topic_prior is None else doc_topic_prior

    lda = LatentDirichletAllocation(n_components=n_components, 
                                    doc_topic_prior=doc_topic_prior,
                                    topic_word_prior=topic_word_prior,
                                    learning_method=learning_method,
                                    learning_decay=learning_decay,
                                    learning_offset=learning_offset,
                                    max_iter=max_iter,
                                    batch_size=batch_size,
                                    evaluate_every=evaluate_every,
                                    total_samples=total_samples,
                                    perp_tol=perp_tol,
                                    mean_change_tol=mean_change_tol,
                                    max_doc_update_iter=max_doc_update_iter,
                                    verbose=verbose,
                                    random_state=random_state,
                                    n_jobs=n_jobs)

    X_lda = lda.fit_transform(adata.X if x is None else adata.obsm[x])
        
    adata.obsm[out] = pd.DataFrame(X_lda, 
                                   index=list(adata.obs.index), 
                                   columns=['Topic {}'.format(i) for i in range(n_components)])
    adata.varm[out] = lda.components_.T
    adata.uns['{}_params'.format(out)] = {'n_components': n_components, 
                                          'doc_topic_prior': doc_topic_prior,
                                          'topic_word_prior': topic_word_prior,
                                          'learning_method': learning_method,
                                          'learning_decay': learning_decay,
                                          'learning_offset': learning_offset,
                                          'max_iter': max_iter,
                                          'batch_size': batch_size,
                                          'evaluate_every': evaluate_every,
                                          'total_samples': total_samples,
                                          'perp_tol': perp_tol,
                                          'mean_change_tol': mean_change_tol,
                                          'max_doc_update_iter': max_doc_update_iter,
                                          'verbose': verbose,
                                          'random_state': random_state,
                                          'n_jobs': n_jobs,
                                          'out': out,
                                          'x': x}

    return adata if copy else None
