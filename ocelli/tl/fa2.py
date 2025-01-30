import anndata as ad
import numpy as np
import pandas as pd
import pkg_resources
from multiprocessing import cpu_count
import os


def fa2(adata: ad.AnnData,
        graph: str = 'graph',
        n_components: int = 2,
        n_iter: int = 4000,
        linlogmode: bool = False,
        gravity: float = 1.,
        flags: str = '',
        out: str = 'X_fa2',
        random_state = None,
        n_jobs: int = -1,
        copy=False):
    """
    ForceAtlas2 dimensionality reduction

    ForceAtlas2 generates 2D or 3D graph embeddings by simulating node interactions based on attractive 
    and repulsive forces. This function wraps the Klarman Cell Observatory's Java implementation of ForceAtlas2.

    .. note::
        Before using this function, you must construct a graph by running 
        `ocelli.tl.neighbors_graph` or `ocelli.tl.transitions_graph`.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param graph: Key in `adata.obsm` storing the graph to visualize. The graph should be an array of cell indices. (default: `'graph'`)
    :type graph: str

    :param n_components: Dimensionality of the ForceAtlas2 embedding. Valid options: 2 (2D) or 3 (3D). (default: 2)
    :type n_components: int

    :param n_iter: Number of iterations for the ForceAtlas2 algorithm. Higher values refine the embedding. (default: 4000)
    :type n_iter: int

    :param linlogmode: Whether to switch to the lin-log mode, which tightens clusters. (default: `False`)
    :type linlogmode: bool

    :param gravity: Controls the attraction of nodes to the graph center. (default: 1.0)
    :type gravity: float

    :param flags: Additional ForceAtlas2 command-line flags. Refer to the official documentation for details. (default: `''`)
    :type flags: str

    :param out: Key in `adata.obsm` where the ForceAtlas2 embedding is saved. (default: `'X_fa2'`)
    :type out: str

    :param random_state: Seed for reproducibility. If `None`, no seed is set. (default: `None`)
    :type random_state: int or None

    :param n_jobs: Number of parallel jobs to use. If `-1`, all CPUs are used. (default: `-1`)
    :type n_jobs: int

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the ForceAtlas2 embedding stored in `adata.obsm[out]`.
        - If `copy=True`: Returns a modified copy of `adata` with the ForceAtlas2 embedding.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm['graph'] = np.random.randint(0, 100, size=(100, 10))

            # Construct ForceAtlas2 embedding
            oci.tl.fa2(
                adata,
                graph='graph',
                n_components=2,
                n_iter=4000,
                linlogmode=True,
                gravity=1.5,
                random_state=42,
                verbose=True
            )
    """
    
    if graph not in list(adata.obsm.keys()):
        raise(KeyError('No graph found. Construct a graph first.'))
        
    if n_components not in [2, 3]:
        raise(ValueError('Wrong number of dimensions. Valid options: 2, 3.'))
        
    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

    graph_path = 'graph.csv'
    df = pd.DataFrame(adata.obsm[graph], columns=[str(i) for i in range(adata.obsm[graph].shape[1])])
    df.to_csv(graph_path, sep=';', header=False)
    
    classpath = ('{}:{}'.format(pkg_resources.resource_filename('ocelli', 'forceatlas2/forceatlas2.jar'), 
                                pkg_resources.resource_filename('ocelli', 'forceatlas2/gephi-toolkit-0.9.2-all.jar')))

    output_name = 'fa2'
    linlogmode_command = '--linLogMode true' if linlogmode else ''
    gravity_command = '--gravity {}'.format(gravity)
    dim_command = '--2d' if n_components == 2 else ''
    thread_command = '--nthreads {}'.format(n_jobs)
    random_command = '--seed {}'.format(random_state) if random_state is not None else ''
    
    command = ['java -Djava.awt.headless=true -Xmx8g -cp', 
               classpath, 
               'kco.forceatlas2.Main', 
               '--input', 
               graph_path, 
               '--nsteps',
               n_iter, 
               '--output', 
               output_name,
               linlogmode_command,
               gravity_command,
               dim_command,
               thread_command,
               random_command,
               flags]
    
    os.system(' '.join(map(str, command)))

    adata.obsm[out] = np.asarray(pd.read_csv('{}.txt'.format(output_name),
                                             sep='\t').sort_values(by='id').reset_index(drop=True).drop('id', axis=1))

    if os.path.exists('{}.txt'.format(output_name)):
        os.remove('{}.txt'.format(output_name))
    if os.path.exists('{}.distances.txt'.format(output_name)):
        os.remove('{}.distances.txt'.format(output_name))
    if os.path.exists(graph_path):
        os.remove(graph_path)

    return adata if copy else None
