import anndata
import numpy as np
import pandas as pd
import pkg_resources
from multiprocessing import cpu_count


def FA2(adata: anndata.AnnData,
        n_components: int = 2,
        graph_key: str = 'graph',
        n_steps: int = 1000,
        random_state = None,
        n_jobs: int = -1,
        flags: str = '',
        output_key: str = 'X_fa2',
        copy=False):
    """Graph dimension reduction using ForceAtlas2

    2D and 3D representations of graphs using force-directed layout algorithm ForceAtlas2.

    This funstion is a wrapper for Klarman Cell Observatory's Gephi implementation of ForceAtlas2.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_components
        Defines whether ForceAtlas2 data reduction should be 2- or 3-dimensional. Valid options: 2, 3. (default: 2)
    graph_key
        ``adata.obsm[graph_key]`` stores the graph to be visualized. (default: `graph`)
    n_steps
        The number of ForceAtlas2 iterations. (default: 1000)
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    flags
        Optionally, additional ForceAtlas2 command line flags as described in https://github.com/klarman-cell-observatory/forceatlas2.
    output_key
        ``adata.uns[output_key]`` will store the ForceAtlas2 embedding. (default: `X_fa2`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[output_key]`` (:class:`numpy.ndarray` of shape ``(n_obs, n_components)`` storing 
        the ForceAtlas2 data representation).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """
    if graph_key not in list(adata.obsm.keys()):
        raise(KeyError('No graph found. Construct a graph first.'))

    graph_path = 'graph.csv'
    df = pd.DataFrame(adata.obsm[graph_key], columns=[str(i) for i in range(adata.obsm[graph_key].shape[1])])
    df.to_csv(graph_path, sep=';', header=False)
    
    classpath = (
            pkg_resources.resource_filename('ocelli', 'forceatlas2/forceatlas2.jar')
            + ":"
            + pkg_resources.resource_filename('ocelli', 'forceatlas2/gephi-toolkit-0.9.2-all.jar')
    )

    output_name = 'fa2'
    command = ['java', 
               '-Djava.awt.headless=true',
               '-Xmx8g',
               '-cp',
               classpath, 
               'kco.forceatlas2.Main', 
               '--input', 
               graph_path, 
               '--nsteps',
               n_steps, 
               '--output', 
               output_name,
               flags]
    
    if n_components == 2:
        command.append('--2d')
    elif n_components == 3:
        pass
    else:
        raise(ValueError('Wrong n_components value. Valid options: 2, 3.'))
        
    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])
    command += ['--nthreads', n_jobs]
    
    if random_state is not None:
        command += ['--seed', random_state]
    
    os.system(' '.join(map(str, command)))

    adata.obsm[output_key] = np.asarray(
        pd.read_csv('{}.txt'.format(output_name),
                    sep='\t').sort_values(by='id').reset_index(drop=True).drop('id', axis=1))

    if os.path.exists('{}.txt'.format(output_name)):
        os.remove('{}.txt'.format(output_name))
    if os.path.exists('{}.distances.txt'.format(output_name)):
        os.remove('{}.distances.txt'.format(output_name))
    if os.path.exists(graph_path):
        os.remove(graph_path)

    return adata if copy else None
