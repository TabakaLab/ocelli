import anndata as ad
import numpy as np


def neighbors_graph(adata: ad.AnnData,
                    x: str,
                    n_edges: int = 10,
                    out: str = 'graph',      
                    verbose: bool = False,
                    copy: bool = False):
    """
    Nearest neighbors-based graph construction

    Constructs a nearest neighbors-based graph from a precomputed nearest neighbors search. 
    Each graph node (cell) is connected to `n_edges` nearest neighbors in the embedding space.

    .. note::
        Before using this function, you must run `ocelli.pp.neighbors` to compute the nearest neighbors 
        for the specified embedding.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param x: Key in `adata.obsm` corresponding to the embedding for which nearest neighbors were computed.
    :type x: str

    :param n_edges: Number of edges (connections) per graph node. (default: 10)
    :type n_edges: int

    :param out: Key in `adata.obsm` where the constructed graph is saved. (default: `'graph'`)
    :type out: str

    :param verbose: Whether to print progress notifications. (default: `False`)
    :type verbose: bool

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the constructed graph stored in `adata.obsm[out]`.
        - If `copy=True`: Returns a modified copy of `adata` with the constructed graph.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm['embedding'] = np.random.rand(100, 10)

            # Compute nearest neighbors
            oci.pp.neighbors(adata, x=['embedding'], n_neighbors=20)

            # Construct nearest neighbors-based graph
            oci.tl.neighbors_graph(adata, x='embedding', n_edges=10, verbose=True)
    """
    
    if 'neighbors_{}'.format(x) not in adata.obsm:
        raise(KeyError('No nearest neighbors found in adata.obsm[neighbors_{}]. Run ocelli.pp.neighbors.'.format(x)))
        
    adata.obsm[out] = np.asarray(adata.obsm['neighbors_{}'.format(x)][:, :n_edges])

    if verbose:
        print('Nearest neighbors-based graph constructed.')
    
    return adata if copy else None
