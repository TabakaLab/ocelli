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


def gaussian_kernel(x, y, epsilon_x, epsilon_y):
    """Adjusted Gaussian kernel"""
    epsilons_mul = epsilon_x * epsilon_y
    if epsilons_mul == 0:
        return 0
    else:
        try:
            x, y = x.toarray(), y.toarray()
        except AttributeError:
            pass

        x, y = x.flatten(), y.flatten()

        return np.exp(-1 * np.power(np.linalg.norm(x - y), 2) / epsilons_mul)


class MultiViewDiffMaps():
    """The multi-view diffusion maps class"""
    def __init__(self,
                 n_jobs=cpu_count()):
        if not ray.is_initialized():
            ray.init(num_cpus=n_jobs)
        self.n_jobs = n_jobs
        
    @staticmethod
    @ray.remote
    def __affinity_worker(views, nn, epsilons, weights, split, view_id, split_id):
        affinities = list()
        for cell0 in split[split_id]:
            for cell1 in nn[view_id][cell0]:
                if cell0 in nn[view_id][cell1] and cell0 < cell1:
                    pass
                else:
                    affinity = gaussian_kernel(views[view_id][cell0],
                                               views[view_id][cell1],
                                               epsilons[view_id][cell0],
                                               epsilons[view_id][cell1])
                    
                    mean_weight = (weights[cell0, view_id] + weights[cell1, view_id]) / 2
                    affinities.append([cell0, cell1, affinity * mean_weight])
                    affinities.append([cell1, cell0, affinity * mean_weight])
                    
        return np.asarray(affinities)
    
    def fit_transform(self, 
                      views, 
                      n_comps=10,
                      nn=None, 
                      epsilons=None, 
                      weights=None,
                      normalize_single_views=True,
                      eigval_times_eigvec=True):
        
        n_views = len(views)
        n_cells = views[0].shape[0]

        split = np.array_split(range(n_cells), self.n_jobs)
        
        views_ref = ray.put(views)
        nn_ref = ray.put(nn)
        epsilons_ref = ray.put(epsilons)
        split_ref = ray.put(split)
        weights_ref = ray.put(weights)
        
        for view_id, _ in enumerate(views):
            affinities = [self.__affinity_worker.remote(views_ref, 
                                                        nn_ref, 
                                                        epsilons_ref,
                                                        weights_ref,
                                                        split_ref, 
                                                        view_id, 
                                                        i) 
                          for i in range(self.n_jobs)]

            affinities = ray.get(affinities)
            affinities = np.vstack(affinities)
            affinity_view = coo_matrix((affinities[:, 2], (affinities[:, 0], affinities[:, 1])), 
                                         shape=(n_cells, n_cells)).tocsr()
            
            if normalize_single_views:
                diag_vals = np.asarray([1 / val if val != 0 else 0 for val in affinity_view.sum(axis=1).A1])
                affinity_view = diags(diag_vals) @ affinity_view
            
            if view_id == 0:
                affinity_matrix = affinity_view
            else:
                affinity_matrix += affinity_view
                
        diag_vals = np.asarray([1 / val if val != 0 else 1 for val in affinity_matrix.sum(axis=1).A1])
        affinity_matrix = diags(diag_vals) @ affinity_matrix
        
        affinity_matrix = affinity_matrix + affinity_matrix.T
        
        eigvals, eigvecs = eigsh(affinity_matrix, k=n_comps + 11, which='LA', maxiter=100000)
        eigvals, eigvecs = np.flip(eigvals)[1:n_comps + 1], np.flip(eigvecs, axis=1)[:, 1:n_comps + 1]
        rep = eigvecs * eigvals if eigval_times_eigvec else eigvecs
    
        return rep


def MVDM(adata: AnnData,
                              n_comps: int = 10,
                              view_keys = None,
                              weights_key: str = 'weights',
                              neighbors_key: str = 'neighbors',
                              epsilons_key: str = 'epsilons',
                              x_key: str = 'x_mvdm',
                              normalize_single_views: bool = True,
                              eigval_times_eigvec: bool = True,
                              n_jobs: int = -1,
                              copy: bool = False):
    """Multi-view diffusion maps

    Algorithm calculates multi-view diffusion maps cell embeddings using
    pre-calculated multi-view weights.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_comps
        The number of multi-view diffusion maps components. (default: 10)
    view_keys
        If :obj:`None`, view keys are loaded from ``adata.uns['key_views']``. Otherwise,
        ``view_keys`` should be a :class:`list` of ``adata.obsm`` keys,
        where views are stored. (default: :obj:`None`)
    weights_key
        ``adata.obsm[weights_key]`` stores multi-view weights. (default: ``weights``)
    neighbors_key
        ``adata.uns[neighbors_key]`` stores the nearest neighbor indices 
        (:class:`numpy.ndarray` of shape ``(n_views, n_cells, n_neighbors)``).
        (default: ``neighbors``)
    epsilons_key
        ``adata.uns[epsilons_key]`` stores epsilons used for adjusted Gaussian kernel 
        (:class:`numpy.ndarray` of shape ``(n_views, n_cells, n_neighbors)``).
        (default: ``epsilons``)
    x_key
        The multi-view diffusion maps embedding are saved to ``adata.obsm[x_key]``. (default: ``x_mvdm``)
    normalize_single_views
        If ``True``, single-view kernel matrices are normalized. (default: ``True``)
    eigval_times_eigvec
        If ``True``, the multi-view diffusion maps embedding is calculated by multiplying
        eigenvectors by eigenvalues. Otherwise, the embedding consists of unmultiplied eigenvectors. (default: ``True``)
    n_jobs
        The number of parallel jobs. If the number is larger than the number of CPUs, it is changed to -1.
        -1 means all processors are used. (default: -1)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[x_key]`` (:class:`numpy.ndarray`).
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    if view_keys is None:
        if 'view_keys' not in list(adata.uns.keys()):
            raise(NameError('No view keys found in adata.uns["view_keys"].'))
        view_keys = adata.uns['view_keys']
 
    if len(view_keys) == 0:
        raise(NameError('No view keys found in adata.uns["view_keys"].'))

    if weights_key not in adata.obsm:
        raise(KeyError('No weights found in adata.uns["{}"]. Run oci.tl.weights.'.format(weights_key)))

    if neighbors_key not in adata.uns:
        raise (KeyError(
            'No nearest neighbors found in adata.uns["{}"]. Run oci.pp.neighbors.'.format(neighbors_key)))
        
    if epsilons_key not in adata.uns:
        raise(KeyError('No epsilons found in adata.uns["{}"]. Run oci.pp.neighbors.'.format(epsilons_key)))

    n_jobs = cpu_count() if n_jobs == -1 else min([n_jobs, cpu_count()])

    adata.obsm[x_key] = MultiViewDiffMaps(n_jobs).fit_transform(views = [adata.obsm[key] for key in view_keys],
                                                                n_comps = n_comps,
                                                                nn = adata.uns[neighbors_key],
                                                                epsilons = adata.uns[epsilons_key],
                                                                weights = np.asarray(adata.obsm[weights_key]),
                                                                normalize_single_views = normalize_single_views,
                                                                eigval_times_eigvec = eigval_times_eigvec)

    print('{} multi-view diffusion maps components calculated.'.format(n_comps))
    
    return adata if copy else None




def FA2(adata: AnnData,
                graph_key: str = 'graph',
                n_steps: int = 1000,
                is2d: bool = True,
                x_fa2_key: str = 'x_fa2',
                copy=False):
    """Force-directed graph layout

    2D and 3D plotting of graphs using ForceAtlas2.

    Klarman Cell Observatory Java and Gephi implementation is used.

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
