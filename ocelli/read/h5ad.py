import anndata as ad


def h5ad(filename):
    """
    Load an `h5ad` file

    This function reads an AnnData object from the specified `.h5ad` file. The `.h5ad` format 
    is widely used for storing single-cell data, enabling efficient handling of large datasets.

    :param filename: The path to the `.h5ad` file containing the AnnData object to be loaded.
    :type filename: str

    :returns: An AnnData object loaded from the specified file.
    :rtype: anndata.AnnData

    :example:

        .. code-block:: python

            import ocelli as oci

            # Load the AnnData object
            adata = oci.h5ad("path/to/file.h5ad")
    """
    
    return ad.read_h5ad(filename)
