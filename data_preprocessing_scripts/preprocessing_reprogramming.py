import anndata as ad
import pandas as pd
import scanpy as sc
import ocelli as oci
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz
from tqdm import tqdm


SEED = 17

adata = ad.read_h5ad('reprogramming_raw.h5ad')

df_obs = pd.DataFrame(index=list(adata.obs.index))
df_obs['day'] = [el.split('_')[0][1:] for el in adata.obs['origin']]
df_obs['origin'] = [el.split('_')[1] for el in adata.obs['origin']]
df_obs['doublet_score'] = list(adata.obs['doublet_score'])
adata.obs = df_obs

df_var = pd.read_csv('var-genes.csv', names=['gene'], index_col='gene', header=0)
adata.var = df_var

adata = adata[adata.obs['origin'].isin(['Dox', 'serum']), :]

adata = adata[adata.obs['doublet_score'] < 0.3]

filtered_timestamps = ['8', '8.25', '8.5', '8.75', '9', '9.5',  
                       '10', '10.5', '11', '11.5', '12', '12.5',
                       '13', '13.5', '14', '14.5', '15', '15.5',
                       '16', '16.5', '17', '17.5', '18']

adata = adata[adata.obs['day'].isin(filtered_timestamps), :]

sc.pp.filter_cells(adata, min_counts=2000)
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.downsample_counts(adata, counts_per_cell=15000, random_state=SEED)

adata.obs['day'] = [float(t) for t in adata.obs['day']]
adata.obs = adata.obs[['origin', 'day']]
adata.var = adata.var[[]]
adata.layers = {}

adata.write('reprogramming_RNAseq.h5ad', compression='gzip', compression_opts=9)

paths = ['serum_8.0_8.25.h5ad', 'serum_8.25_8.5.h5ad',  'serum_8.5_8.75.h5ad', 
         'serum_8.75_9.0.h5ad', 'serum_9.0_9.5.h5ad', 'serum_9.5_10.0.h5ad', 
         'serum_10.0_10.5.h5ad', 'serum_10.5_11.0.h5ad', 'serum_11.0_11.5.h5ad', 
         'serum_11.5_12.0.h5ad', 'serum_12.0_12.5.h5ad', 'serum_12.5_13.0.h5ad', 
         'serum_13.0_13.5.h5ad', 'serum_13.5_14.0.h5ad', 'serum_14.0_14.5.h5ad', 
         'serum_14.5_15.0.h5ad', 'serum_15.0_15.5.h5ad', 'serum_15.5_16.0.h5ad', 
         'serum_16.0_16.5.h5ad', 'serum_16.5_17.0.h5ad', 'serum_17.0_17.5.h5ad', 
         'serum_17.5_18.0.h5ad']

def filter_cells(adata, x):
    obs_in, var_in = list(), list()
    
    for el in adata.obs.index:
        if el in x.obs.index:
            obs_in.append(el)
    for el in adata.obs.index:
        if el in x.var.index:
            var_in.append(el)
            
    return x[obs_in, var_in]
        
barcode_map = dict()
for i, barcode in enumerate(adata.obs.index):
    barcode_map[barcode] = i
    
M = coo_matrix(([], ([], [])), shape=(adata.shape[0], adata.shape[0])).tocsr()
    
for path in tqdm(paths):
    x = ad.read_h5ad(join(folder_path, 'tmaps/{}'.format(path)))    
    x = filter_cells(adata, x)
    
    obs = [barcode_map[el] for el in x.obs.index]
    var = [barcode_map[el] for el in x.var.index]
    
    M[np.ix_(obs, var)] = csr_matrix(x.X)

save_npz('reprogramming_WOT.npz', M)
