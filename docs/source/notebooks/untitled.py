import anndata
import scvelo as scv
import pandas as pd
import numpy as np
import scanpy as scp
from scipy.sparse import issparse, csr_matrix, coo_matrix
from tqdm import tqdm
from os.path import join
import ocelli as oci

data_folder = '../../../../experiments/reprogramming/data'
output_folder = '../../../../experiments/reprogramming/output'

SEED = 17

adata = anndata.read_h5ad(join(output_folder, 'R1.h5ad'))

oci.pp.LDA(adata, n_components=20, output_key='lda', verbose=2, max_iter=30, random_state=SEED)
oci.pp.modality_generation(adata, topic_key='lda', norm_log=True, verbose=True)
adata.write(join(output_folder, 'R2.h5ad'))

paths = ['serum_0.0_0.5.h5ad', 'serum_0.5_1.0.h5ad', 'serum_1.0_1.5.h5ad', 
         'serum_1.5_2.0.h5ad', 'serum_2.0_2.5.h5ad', 'serum_2.5_3.0.h5ad', 
         'serum_3.0_3.5.h5ad', 'serum_3.5_4.0.h5ad', 'serum_4.0_4.5.h5ad', 
         'serum_4.5_5.0.h5ad', 'serum_5.0_5.5.h5ad', 'serum_5.5_6.0.h5ad', 
         'serum_6.0_6.5.h5ad', 'serum_6.5_7.0.h5ad', 'serum_7.0_7.5.h5ad', 
         'serum_7.5_8.0.h5ad', 'serum_8.0_8.25.h5ad', 'serum_8.25_8.5.h5ad', 
         'serum_8.5_8.75.h5ad', 'serum_8.75_9.0.h5ad', 'serum_9.0_9.5.h5ad', 
         'serum_9.5_10.0.h5ad', 'serum_10.0_10.5.h5ad', 'serum_10.5_11.0.h5ad', 
         'serum_11.0_11.5.h5ad', 'serum_11.5_12.0.h5ad', 'serum_12.0_12.5.h5ad', 
         'serum_12.5_13.0.h5ad', 'serum_13.0_13.5.h5ad', 'serum_13.5_14.0.h5ad', 
         'serum_14.0_14.5.h5ad', 'serum_14.5_15.0.h5ad', 'serum_15.0_15.5.h5ad',
         'serum_15.5_16.0.h5ad', 'serum_16.0_16.5.h5ad', 'serum_16.5_17.0.h5ad', 
         'serum_17.0_17.5.h5ad', 'serum_17.5_18.0.h5ad']

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
    x = anndata.read_h5ad(join(data_folder, 'tmaps/{}'.format(path)))    
    x = filter_cells(adata, x)
    
    obs = [barcode_map[el] for el in x.obs.index]
    var = [barcode_map[el] for el in x.var.index]
    
    M[np.ix_(obs, var)] = csr_matrix(x.X)

adata.uns['velocity_graph'] = M

adata.write(join(output_folder, 'R3.h5ad'))

oci.pp.neighbors(adata, n_neighbors=20, verbose=True)
oci.tl.MDM(adata, weights_key='lda', n_components=20, random_state=SEED, verbose=True)
adata.write(join(output_folder, 'R4.h5ad'))

oci.pp.neighbors(adata, modalities=['X_mdm'], neighbors_key='neighbors_mdm', n_neighbors=100, verbose=True)
oci.tl.vel_graph(adata, n=10, neighbors_key='neighbors_mdm', verbose=True, use_timestamps=True)
oci.tl.FA2(adata, n_components=2, n_steps=10000, random_state=SEED, output_key='X_fa2')
adata.write(join(output_folder, 'R5.h5ad'))