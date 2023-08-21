import scanpy as sc
import pandas as pd


adata = sc.read_mtx('matrix.mtx.gz').T

df_obs = pd.read_csv('barcodes.tsv', sep='\t', header=None)
df_obs['celltype'] = pd.read_csv('human_bone_marrow_celltypes.csv')['celltype']
df_obs.columns = ['barcode', 'celltype']
df_obs = df_obs.set_index('barcode')

adata.obs = df_obs

df_var = pd.read_csv('features.tsv', sep='\t', header=None)
df_var = df_var[[0, 2]]
df_var.columns = ['', 'modality']
df_var = df_var.set_index('')

adata.var = df_var

protein = adata[:, adata.var.modality == 'Antibody Capture']
protein.write('human_bone_marrow_prot.h5ad', compression='gzip', compression_opts=9)

chromatin = adata[:, adata.var.modality == 'Gene Expression']
chromatin.write('human_bone_marrow_atac.h5ad', compression='gzip', compression_opts=9)
