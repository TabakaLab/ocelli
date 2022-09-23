import anndata
import numpy as np
import matplotlib.pyplot as plt


def weights(adata: anndata.AnnData,
            weights_key: str = 'weights',
            celltype_key: str = 'celltype',
            showmeans: bool = False, 
            showmedians: bool = True, 
            showextrema: bool = False):
    """Multimodal weights violin plots
    
    Basic violin plots of multimodal weights. 
    A seperate violin plot is generated for each modality and celltype. 
    Used best when the numbers of modalities and cell types are not large.
    
    Returns a :class:`tuple` of :class:`matplotlib` figure and axes.
    They can be further customized, or saved.
    
    Parameters
    ----------
    adata
        The annotated data matrix.
    weights_key
        ``adata.obsm[weights_key]`` stores multimodal weights. (default: ``weights``)
    celltype_key
        ``adata.obs[celltype_key]`` stores celltypes. For each celltype 
        a seperate violin plot is generated. If ``celltype_key`` is not found, 
        violin plots for all cells are generated. (default: ``celltype``)
    showmeans
        If ``True``, will toggle rendering of the means. (default: ``False``)
    showmedians
        If ``True``, will toggle rendering of the medians. (default: ``True``)
    showextrema
        If ``True``, will toggle rendering of the extrema. (default: ``False``)

    Returns
    -------
    :class:`tuple`
        :class:`matplotlib.figure.Figure` and :class:`numpy.ndarray` storing :class:`matplotlib` figure and axes.
    """
    
    if celltype_key not in list(adata.obs.keys()):
        modalities = list(adata.obsm['weights'].columns)
        fig, ax = plt.subplots(nrows=len(modalities), ncols=1)
        fig.supylabel('modalities', size=6)
        fig.suptitle('weights', size=6)

        for i, m in enumerate(modalities):
            ax[i].violinplot(adata.obsm['weights'][m], 
                             showmeans=showmeans, 
                             showmedians=showmedians, 
                             showextrema=showextrema)
            ax[i].set_ylabel(m, size=6)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].set_xticks([])
            ax[i].tick_params(axis='both', which='major', labelsize=6, length=1)
            ax[i].set_yticks([0, 0.5, 1])
            ax[i].set_yticklabels([0, 0.5, 1])
            ax[i].set_ylim([0,1])
        plt.tight_layout()
    else:
        modalities = list(adata.obsm['weights'].columns)
        celltypes = list(np.unique(adata.obs[celltype_key]))

        fig, ax = plt.subplots(nrows=len(modalities), ncols=len(celltypes))
        fig.supylabel('modalities', size=6)
        fig.suptitle('celltypes', size=6)

        for i, m in enumerate(modalities):
            for j, celltype in enumerate(celltypes):
                ax[i][j].violinplot(adata[adata.obs[celltype_key] == celltype].obsm[weights_key][m], 
                                    showmeans=showmeans, 
                                    showmedians=showmedians, 
                                    showextrema=showextrema)
                if i == 0:
                    ax[i][j].set_title(celltypes[j], size=6)
                if j == 0:
                    ax[i][j].set_ylabel(m, size=6)
                ax[i][j].spines['right'].set_visible(False)
                ax[i][j].spines['top'].set_visible(False)
                ax[i][j].spines['bottom'].set_visible(False)
                ax[i][j].set_xticks([])
                ax[i][j].tick_params(axis='both', which='major', labelsize=6, length=1)
                ax[i][j].set_yticks([0, 0.5, 1])
                if j == 0:
                    ax[i][j].set_yticklabels([0, 0.5, 1])
                else:
                    ax[i][j].set_yticklabels([])
                ax[i][j].set_ylim([0,1])
        plt.tight_layout()
    
    return fig, ax
