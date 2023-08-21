import scanpy as sc
import scvelo as scv
import pandas as pd


adata = scv.datasets.pancreas()

sc.pp.filter_cells(adata, min_genes=20)
scv.pp.filter_genes(adata, min_shared_cells=20)

adata.obs = pd.read_csv('pancreas_celltypes.csv', index_col='index')
adata.var = adata.var[[]]
adata.uns = {}
adata.obsm = {}
adata.obsp = {}

adata.write('pancreas_rna.h5ad', compression='gzip', compression_opts=9)
