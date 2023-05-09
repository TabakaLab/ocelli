library(Signac)
library(Seurat)
library(ComplexHeatmap)
library(GenomeInfoDb)
library(EnsDb.Mmusculus.v79)
library(data.table)
require(data.table)
require(Matrix)

gene.coords <- genes(EnsDb.Mmusculus.v79, filter = ~ gene_biotype == "protein_coding")
seqlevelsStyle(gene.coords) <- 'UCSC'
genebody.coords <- keepStandardChromosomes(gene.coords, pruning.mode = 'coarse')
genebodyandpromoter.coords <- Extend(x = gene.coords, upstream = 2000, downstream = 0)

frag_file <- "GSM4156597_skin.late.anagen.atac.fragments.srt.proc.bed.gz"

fragments <- CreateFragmentObject(frag_file)
mat<-FeatureMatrix(fragments = fragments, features = genebodyandpromoter.coords, cells = NULL)
gene.key <- genebodyandpromoter.coords$gene_name
names(gene.key) <- GRangesToString(grange = genebodyandpromoter.coords)

rownames(mat) <- gene.key[rownames(mat)]

mat <- as(mat, "sparseMatrix")     
writeMM(mat,file="fragments_matrix.mtx")
fwrite(as.data.frame(colnames(mat)),file="CellIDS.txt.gz",compress = "auto")
fwrite(as.data.frame(rownames(mat)),file="genes.txt.gz",compress = "auto")
