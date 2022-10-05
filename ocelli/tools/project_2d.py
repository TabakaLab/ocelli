import anndata
import numpy as np


def project_2d(adata: anndata.AnnData,
               x3d_key: str,
               output_key: str = 'projection',
               alpha: int = 0,
               beta: int = 0,
               random_state = None,
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
    x3d_key
        ``adata.obsm[x3d_key]`` stores a 3D embedding to be projected onto a plane.
    output_key
        A 2D projection is saved to ``adata.obsm[output_key]``. (default: ``projection``)
    alpha
        The first of polar coordinates' angles which define a projection
        plane's normal vector. ``beta`` is the second one. Use degrees, not radians. (default: 0)
    beta
        The second of polar coordinates' angles which define a projection
        plane's normal vector. ``alpha`` is the first one. Use degrees, not radians. (default: 0)
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: ``False``)

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obsm[output_key]`` (:class:`numpy.ndarray` of shape ``(n_obs, 2)`` storing
        a 2D embedding projection.
    :class:`anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """
    
    if random_state is not None:
        np.random.seed(random_state)
    
    if alpha % 90 == 0:
        alpha += 1

    if beta % 90 == 0:
        beta += 1

    alpha = alpha * ((2 * np.pi) / 360)
    beta = beta * ((2 * np.pi) / 360)

    n = np.asarray([np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta), np.sin(alpha)])

    plane_3d = np.asarray([x - (n*np.dot(n, x)) for x in adata.obsm[x3d_key]])

    v1 = plane_3d[0]
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.linalg.solve(np.stack((n, v1, np.random.randint(100, size=3))), np.asarray([0, 0, 1]))
    v2 = v2 / np.linalg.norm(v2)

    plane_2d = np.asarray([np.linalg.solve(np.column_stack([v1[:2], v2[:2]]), p[:2]) for p in plane_3d])

    adata.obsm[output_key] = plane_2d

    return adata if copy else None
