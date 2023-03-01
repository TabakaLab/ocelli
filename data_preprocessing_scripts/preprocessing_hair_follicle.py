import scanpy as sc
import scrublet as scr
import numpy as np
import pandas as pd
import ocelli as oci
from tqdm import tqdm

SEED = 17

# preprocess RNA-seq

rna_seq = sc.read_loom('skin_XCISH.loom')
rna_seq.var_names_make_unique()
rna_seq.obs.index = [index.split(':')[1].replace(',', '.') for index in rna_seq.obs.index]

celltypes = pd.read_csv('GSM4156597_skin_celltype.txt', sep='\t')
celltypes = celltypes[celltypes.celltype.isin(['TAC-1', 'TAC-2', 'IRS', 'Medulla', 'Hair Shaft-cuticle.cortex'])].reset_index(drop=True)

rna_seq = rna_seq[celltypes['rna.bc']]
rna_seq.obs['celltype'] = list(celltypes['celltype'])

# remove genes associated with RNA contamination
rna_seq = rna_seq[:, ~rna_seq.var.index.isin(['Malat1', 'Gm42418', 'AY036118'])]

# remove soublets
D, N = 884736, rna_seq.shape[0]
n_collisions = int(N - D + D * (((D - 1) / D)**N))

scrublet = scr.Scrublet(rna_seq.X, expected_doublet_rate=n_collisions / N, random_state=SEED)
doublet_scores, predicted_doublets = scrublet.scrub_doublets()

doublet_mask = [False if i in np.argpartition(doublet_scores, -n_collisions)[-n_collisions:] 
                else True for i in range(rna_seq.shape[0])]

rna_seq.obs['doublet_scores'] = doublet_scores
rna_seq = rna_seq[doublet_mask]

# filter cells and genes
sc.pp.filter_cells(rna_seq, min_genes=50)
sc.pp.filter_genes(rna_seq, min_counts=50)

# remove genes highly correlated with the z-scores of proliferation gene signature
# step 1: log-normalize count matrix and compute z-scores
rna_seq_sig = rna_seq.copy()
sc.pp.normalize_total(rna_seq_sig, target_sum=10000)
sc.pp.log1p(rna_seq_sig)

rna_vars = list(rna_seq_sig.var.index)

def get_indices(signature, var_names):
    signature_indices = list()
    for gene in signature:
        if gene in var_names:
            signature_indices.append(var_names.index(gene))
    return signature_indices

signature = list(pd.read_csv('signature-proliferation.csv', index_col=0).index)
oci.tl.mean_z_scores(rna_seq_sig, 
                     markers=get_indices(signature, rna_vars),
                     out='proliferation', 
                     vmin=-5, 
                     vmax=5)
# step 2: compute correlations for all genes
df = list()
for gene in tqdm(rna_seq.var.index):
    df.append([gene, np.corrcoef(np.asarray(rna_seq_sig.obs['proliferation']), 
                                 rna_seq[:, gene].X.toarray().flatten())[0, 1]])
df = pd.DataFrame(df)
df.columns = ['gene', 'pearson']
# step 3: remove genes with the highest correlations
discarded_genes = list(df[df.pearson > 0.18]['gene'])
rna_seq = rna_seq[:, ~rna_seq.var.index.isin(discarded_genes)]

rna_seq.write('hair_follicle_RNAseq.h5ad', compression='gzip', compression_opts=9)

# preprocess ATAC-seq

# fragments_matrix.mtx.gz is generated using the preprocessing_hair_follicle_atacseq.R script
atac_seq = sc.read_mtx('fragments_matrix.mtx.gz')
atac_seq = atac_seq.T
atac_seq.obs = pd.read_csv('CellIDS.txt', index_col='colnames(mat)')
atac_seq.var = pd.read_csv('genes.txt', index_col='rownames(mat)')

rna_seq_indices = rna_seq.obs.index
celltypes.index = list(celltypes['rna.bc'])
celltypes.loc[rna_seq_indices]['atac.bc']
atac_seq_barcodes = celltypes.loc[rna_seq.obs.index]['atac.bc']
atac_seq = atac_seq[atac_seq_barcodes]
atac_seq.var.index = [str(el) for el in atac_seq.var.index]

# filter cells and genes
sc.pp.filter_genes(atac_seq, min_counts=50)
atac_seq = atac_seq[:, ~atac_seq.var.index.isin(discarded_genes)]

atac_seq.write('hair_follicle_ATACseq.h5ad', compression='gzip', compression_opts=9)
