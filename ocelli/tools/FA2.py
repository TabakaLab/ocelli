import anndata as ad
import numpy as np
import pandas as pd
import pkg_resources
from multiprocessing import cpu_count
import os


def FA2(adata: ad.AnnData,
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
    """ForceAtlas2

    ForceAtlas2 generates 2D and 3D representations of graphs.
    This function is a wrapper of Klarman Cell Observatory's Gephi implementation of ForceAtlas2.
    
    Before running this function, 
    create a graph using :obj:`ocelli.tl.neighbors_graph` or :obj:`ocelli.tl.transitions_graph`.

    Parameters
    ----------
    adata
        The annotated data matrix.
    graph
        `adata.obsm` key storing the graph to be visualized. 
        The graph is represented as an array of cell indices. (default: `graph`)
    n_components
        ForceAtlas2 embedding dimensionality. Valid options: 2, 3. (default: 2)
    n_iter
        Number of ForceAtlas2 iterations. (default: 4000)
    linlogmode
        Switch ForceAtlas' model from lin-lin to lin-log (tribute to Andreas Noack). Makes clusters more tight. (default: `False`)
    gravity
        Attracts nodes to the center. (default: 1.0)
    flags
        Optional. Additional ForceAtlas2 command line flags as described in https://github.com/klarman-cell-observatory/forceatlas2.
    out
        ForceAtlas2 embedding is saved to `adata.obsm[out]`. (default: `X_fa2`)
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
        `adata.obsm[out]` (ForceAtlas2 embedding).
    :class:`anndata.AnnData`
        When `copy=True` is set, a copy of `adata` with those fields is returned.
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
