import anndata
import numpy as np
import matplotlib.pyplot as plt


def weights(adata: anndata.AnnData,
            weights_key: str = 'weights',
            grouping_key = None,
            showmeans: bool = False, 
            showmedians: bool = True, 
            showextrema: bool = False,
            fontsize: int = 6,
            color = None):
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
    grouping_key
        ``adata.obs[grouping_key]`` stores celltypes. For each celltype 
        a seperate violin plot is generated. If ``grouping_key`` is not found, 
        violin plots for all cells are generated. (default: ``None``)
    showmeans
        If ``True``, will toggle rendering of the means. (default: ``False``)
    showmedians
        If ``True``, will toggle rendering of the medians. (default: ``True``)
    showextrema
        If ``True``, will toggle rendering of the extrema. (default: ``False``)
    fontsize
        Plot fontsize. (default: 6)
    color
        Color of violin bodies. (default: ``None``)
    Returns
    -------
    :class:`tuple`
        :class:`matplotlib.figure.Figure` and :class:`numpy.ndarray` storing :class:`matplotlib` figure and axes.
    """
    
    modalities = list(adata.obsm[weights_key].columns)
    
    groups = ['none'] if grouping_key is None else list(np.unique(adata.obs[grouping_key]))
    
    nrows, ncols = len(modalities), len(groups)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.supylabel('Modalities', size=fontsize)
    
    if grouping_key is None:
        for i, m in enumerate(modalities):
            plot = ax[i].violinplot(adata.obsm[weights_key][m], 
                             showmeans=showmeans, 
                             showmedians=showmedians, 
                             showextrema=showextrema)
            if color is not None:
                for pc in plot['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
            ax[i].set_ylabel(m, size=fontsize)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].set_xticks([])
            ax[i].tick_params(axis='both', which='major', labelsize=fontsize, length=1)
            ax[i].set_yticks([0, 0.5, 1])
            ax[i].set_yticklabels([0, 0.5, 1])
            ax[i].set_ylim([0,1])
    else:
        for i, m in enumerate(modalities):
            for j, g in enumerate(groups):
                plot = ax[i][j].violinplot(adata[adata.obs[grouping_key] == g].obsm[weights_key][m], 
                                    showmeans=showmeans, 
                                    showmedians=showmedians, 
                                    showextrema=showextrema)
                if color is not None:
                    for pc in plot['bodies']:
                        pc.set_facecolor(color)
                        pc.set_edgecolor('white')
                        pc.set_alpha(0.7)

                if i == 0:
                    ax[i][j].set_title(groups[j], size=fontsize)
                if j == 0:
                    ax[i][j].set_ylabel(m, size=fontsize)
                ax[i][j].spines['right'].set_visible(False)
                ax[i][j].spines['top'].set_visible(False)
                ax[i][j].spines['bottom'].set_visible(False)
                ax[i][j].set_xticks([])
                ax[i][j].tick_params(axis='both', which='major', labelsize=fontsize, length=1)
                ax[i][j].set_yticks([0, 0.5, 1])
                if j == 0:
                    ax[i][j].set_yticklabels([0, 0.5, 1])
                else:
                    ax[i][j].set_yticklabels([])
                ax[i][j].set_ylim([0,1])
    
    plt.tight_layout()
    
    return fig, ax