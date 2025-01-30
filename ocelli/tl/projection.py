import anndata as ad
import numpy as np


def projection(adata: ad.AnnData,
               x: str,
               phi: int = 20,
               theta: int = 20,
               out: str = 'X_proj',
               random_state = None,
               copy: bool = False):
    """
    Graphical projection of 3D data to 2D

    Projects 3D data onto a 2D plane defined by a normal vector. 
    This method enables better visualization of 3D embeddings by 
    projecting the data while preserving its structure.

    The 2D projection is determined by the orientation of the projection plane, 
    which is defined using spherical coordinates (`phi` and `theta`). 
    The normal vector of the plane is calculated based on these angles. 
    Data is then projected onto the 2D plane using orthogonal base vectors.

    :param adata: The annotated data matrix.
    :type adata: anndata.AnnData

    :param x: Key in `adata.obsm` storing the 3D data to be projected.
    :type x: str

    :param phi: First spherical angle in degrees, ranging from 0 to 360. (default: 20)
    :type phi: int

    :param theta: Second spherical angle in degrees, ranging from 0 to 180. (default: 20)
    :type theta: int

    :param out: Key in `adata.obsm` where the resulting 2D projection will be stored. (default: `'X_proj'`)
    :type out: str

    :param random_state: Seed for reproducibility. If `None`, no seed is set. (default: `None`)
    :type random_state: int or None

    :param copy: Whether to return a copy of `adata`. If `False`, updates are made in-place. (default: `False`)
    :type copy: bool

    :returns:
        - If `copy=False`: Updates `adata` with the 2D projection stored in `adata.obsm[out]`.
        - If `copy=True`: Returns a modified copy of `adata` with the 2D projection.

    :rtype: anndata.AnnData or None

    :example:

        .. code-block:: python

            import ocelli as oci
            from anndata import AnnData
            import numpy as np

            # Example data
            adata = AnnData(X=np.random.rand(100, 50))
            adata.obsm['embedding_3d'] = np.random.rand(100, 3)

            # Project 3D data to 2D
            oci.tl.projection(adata, x='embedding_3d', phi=45, theta=60, out='X_projected')
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
