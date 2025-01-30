from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import cpu_count
import anndata as ad
import pandas as pd


def lda(adata: ad.AnnData,
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
    """
    Latent Dirichlet Allocation

    This function performs Latent Dirichlet Allocation (LDA) using 
    the :class:`sklearn.decomposition.LatentDirichletAllocation` implementation. 
    LDA is a generative probabilistic model used for topic modeling and 
    dimensionality reduction in single-cell data analysis.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param x: The key in `adata.obsm` storing the input data matrix with non-negative values. 
        If `None`, `adata.X` is used as input. (default: `None`)
    :type x: str or None

    :param out: The key to store the LDA output in `adata.obsm` and `adata.varm`. (default: `X_lda`)
    :type out: str

    :param n_components: Number of topics to model. (default: 10)
    :type n_components: int

    :param max_iter: Maximum number of iterations over the dataset. (default: 30)
    :type max_iter: int

    :param doc_topic_prior: Prior of the document-topic distribution. Defaults to `50 / n_components` if `None`. (default: `None`)
    :type doc_topic_prior: float or None

    :param topic_word_prior: Prior of the topic-word distribution. (default: 0.1)
    :type topic_word_prior: float

    :param learning_method: Method used for LDA updates. Options: `batch` or `online`. (default: `batch`)
    :type learning_method: str

    :param learning_decay: Learning rate for the online method. Must be in (0.5, 1.0]. (default: 0.7)
    :type learning_decay: float

    :param learning_offset: Downweighting parameter for early iterations in online learning. Must be >1.0. (default: 10.0)
    :type learning_offset: float

    :param batch_size: Number of documents per EM update in online learning. (default: 128)
    :type batch_size: int

    :param evaluate_every: Frequency of perplexity evaluation. Set to <=0 to disable. (default: -1)
    :type evaluate_every: int

    :param total_samples: Total number of documents. Used only in the partial_fit method. (default: 1,000,000)
    :type total_samples: int

    :param perp_tol: Perplexity tolerance for batch learning. Used only if `evaluate_every` > 0. (default: 0.1)
    :type perp_tol: float

    :param mean_change_tol: Stopping tolerance for updating document-topic distribution. (default: 0.001)
    :type mean_change_tol: float

    :param max_doc_update_iter: Maximum iterations for document-topic updates during the E-step. (default: 100)
    :type max_doc_update_iter: int

    :param verbose: Verbosity level for logging. Options: 0, 1, 2. (default: 0)
    :type verbose: int

    :param random_state: Random seed for reproducibility. (default: `None`)
    :type random_state: int or None

    :param n_jobs: Number of parallel jobs. -1 uses all CPUs. (default: -1)
    :type n_jobs: int

    :param copy: If `True`, returns a copy of the `AnnData` object with LDA results. If `False`, modifies the input object in-place. (default: `False`)
    :type copy: bool

    :returns: 
        - If `copy=False`: Updates the input `adata` with:
            - `adata.obsm[out]`: LDA observation-topic distribution.
            - `adata.varm[out]`: LDA feature-topic distribution.
            - `adata.uns[f"{out}_params"]`: Dictionary of LDA parameters.
        - If `copy=True`: Returns a modified copy of `adata` with the above updates.
    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Create example data
            adata = AnnData(X=np.random.rand(100, 50))

            # Apply LDA
            oci.pp.LDA(adata, n_components=20)
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

    adata.obsm[out] = lda.fit_transform(adata.X if x is None else adata.obsm[x])
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
