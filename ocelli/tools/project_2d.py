from anndata import AnnData
import os
import numpy as np
import pandas as pd
import pkg_resources
import umap
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from multiprocessing import cpu_count
import ray


def FA2(adata: AnnData,
                graph_key: str = 'graph',
                n_steps: int = 1000,
                is2d: bool = True,
                x_fa2_key: str = 'x_fa2',
                copy=False):
    """Force-directed graph layout

    2D and 3D plotting of graphs using ForceAtlas2.

    This funstion is a wrapper for Klarman Cell Observatory's Gephi implementation of ForceAtlas2.

    Parameters
    ----------
    adata
        The annotated data matrix.
    graph_key
        ``adata.obsm[graph_key]`` stores the graph to be visualized. (default: ``graph``)
    n_steps
        The number of ForceAtlas2 iterations. (default: 1000)
    is2d
        Defines whether ForceAtlas2 visualization should be 2- or 3-dimensional. (default: ``True``)
    x_fa2_key
        ``adata.uns[x_fa2_key]`` will store the ForceAtlas2 embedding. (default: ``x_fa2``)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[x_fa2_key]`` (:class:`numpy.ndarray` of shape ``(n_cells, 2)`` or ``(n_cells, 3)`` storing 
        the ForceAtlas2 embedding).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    if graph_key not in list(adata.obsm.keys()):
        raise (KeyError('No graph found. Construct a graph first.'))

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
               output_name]
    if is2d:
        command.append('--2d')
    
    os.system(' '.join(map(str, command)))

    adata.obsm[x_fa2_key] = np.asarray(
        pd.read_csv('{}.txt'.format(output_name),
                    sep='\t').sort_values(by='id').reset_index(drop=True).drop('id', axis=1))

    if os.path.exists('{}.txt'.format(output_name)):
        os.remove('{}.txt'.format(output_name))
    if os.path.exists('{}.distances.txt'.format(output_name)):
        os.remove('{}.distances.txt'.format(output_name))
    if os.path.exists(graph_path):
        os.remove(graph_path)

    return adata if copy else None


def UMAP(adata: AnnData,
         n_components: bool = True,
         x_fa2_key: str = 'x_fa2',
         copy=False):
 

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(adata.X)
    
    return adata if copy else None


def project_2d(adata: AnnData,
               x_key: str,
               projection_key: str = 'projection',
               alpha: int = 0,
               beta: int = 0,
               copy: bool = False):
    """2D projection of 3D embedding

    Projecting 3D embedding onto a 2D plane may result
    in a better visualization when compared to generating a 2D plot.
    This function can be used when 3D embedding is first generated.

    3D data is firstly projected onto a 3D plane,
    which goes through the origin point. The orientation
    of the plane is defined by its normal vector.
    A normal vector is a unit vector controlled
    by a spherical coordinate system angles: ``alpha`` and ``beta``.
    Subsequently, an orthonormal (orthogonal with unit norms) base
    of the 3D plane is found. Then all 3D points are embedded
    into a 2D space by finding their 2D coordinates in the new 2D base.
    Projection does not stretch original data,
    as base vectors have unit norms.

    Parameters
    ----------
    adata
        The annotated data matrix.
    x_key
        ``adata.obsm[x_key]`` stores a 3D embedding, that will be projected onto a plane.
    projection_key
        A 2D projection is saved to ``adata.obsm[projection_key]``. (default: ``projection``)
    alpha
        The first of polar coordinates' angles which define a projection
        plane's normal vector. ``beta`` is the second one. Use degrees, not radians.
    beta
        The second of polar coordinates' angles which define a projection
        plane's normal vector. ``alpha`` is the first one. Use degrees, not radians.
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[projection_key]`` (:class:`numpy.ndarray` of shape ``(n_cells, 2)`` storing
        a 2D embedding projection.
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """
    
    if alpha % 90 == 0:
        alpha += 5

    if beta % 90 == 0:
        beta += 5

    alpha = alpha * ((2 * np.pi) / 360)
    beta = beta * ((2 * np.pi) / 360)

    n = np.asarray([np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta), np.sin(alpha)])

    plane_3d = np.asarray([x - (n*np.dot(n, x)) for x in adata.obsm[x_key]])

    v1 = plane_3d[0]
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.linalg.solve(np.stack((n, v1, np.random.randint(100, size=3))), np.asarray([0, 0, 1]))
    v2 = v2 / np.linalg.norm(v2)

    plane_2d = np.asarray([np.linalg.solve(np.column_stack([v1[:2], v2[:2]]), p[:2]) for p in plane_3d])

    adata.obsm[projection_key] = plane_2d

    return adata if copy else None
