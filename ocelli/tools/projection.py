import anndata as ad
import numpy as np


def projection(adata: ad.AnnData,
               x: str,
               phi: int = 0,
               theta: int = 0,
               out: str = 'X_proj',
               random_state = None,
               copy: bool = False):
    """2D projection of 3D data

    Projecting 3D embedding onto a 2D plane may result
    in a better visualization when compared to generating a 2D plot.

    3D data is first projected onto a plane embedded in 3D,
    which goes through the origin point. The orientation
    of the plane is defined by its normal vector.
    A normal vector is a unit vector controlled
    by a spherical coordinate system angles: ``phi`` and ``theta``.
    Subsequently, an orthonormal (orthogonal with the unit norm) base
    of the 3D plane is found. Then, all 3D points are embedded
    into a 2D space by finding their linear combinations in the new 2D base.
    Projection does not stretch original data,
    as base vectors have unit norms.

    Parameters
    ----------
    adata
        The annotated data matrix.
    x
        `adata.obsm[x]` stores 3D data.
    phi
        Ranges from 0 to 360 (degrees). The first polar coordinate which together define 
        projection plane's normal vector. (default: 0)
    theta
        Ranges from 0 to 180 (degrees). The second polar coordinate which together define 
        projection plane's normal vector. (default: 0) 
    out
        2D projection is saved to `adata.obsm['out']`. (default: `X_proj`)
    random_state
        Pass an :obj:`int` for reproducible results across multiple function calls. (default: :obj:`None`)
    copy
        Return a copy of :class:`anndata.AnnData`. (default: `False`)

    Returns
    -------
    :obj:`None`
        By default (`copy=False`), updates `adata` with the following fields:
        `adata.obsm[out]` (2D projection).
    :class:`anndata.AnnData`
        When `copy=True` is set, a copy of `adata` with those fields is returned.
    """

    
    if random_state is not None:
        np.random.seed(random_state)
        
    if phi < 0 or phi > 360:
        raise(ValueError('phi from 0 to 360 degrees.'))
        
    if theta < 0 or theta > 180:
        raise(ValueError('theta from 0 to 180 degrees.'))
        
    if phi % 90 == 0:
        phi += 1
    
    if theta % 90 == 0:
        theta += 1
        
    phi = phi * ((2 * np.pi) / 360)
    theta = theta * ((2 * np.pi) / 360)

    n = np.asarray([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

    plane_3d = np.asarray([v - (n*np.dot(n, v)) for v in adata.obsm[x]])

    v1 = plane_3d[0]
    v1 /=  np.linalg.norm(v1)
    v2 = np.linalg.solve(np.stack((n, v1, np.random.randint(100, size=3))), np.asarray([0, 0, 1]))
    v2 /= np.linalg.norm(v2)

    plane_2d = np.asarray([np.linalg.solve(np.column_stack([v1[:2], v2[:2]]), p[:2]) for p in plane_3d])

    adata.obsm[out] = plane_2d

    return adata if copy else None
