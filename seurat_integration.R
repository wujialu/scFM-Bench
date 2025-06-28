library(Seurat)
library(SeuratDisk)
library(Matrix)
library(reticulate)

# 从命令行传入数据集路径
args <- commandArgs(trailingOnly = TRUE)
dataset_name <- args[1]
batch_col <- args[2]

# set env
use_python("/data2/zhuyiheng/.conda/envs/singlecell/bin/python")  
use_condaenv("/data2/zhuyiheng/.conda/envs/singlecell")  # 使用 conda 环境
sc <- import("scanpy")
np <- import("numpy")

# load Anndata and transform to Seurat object
input_file <- paste0("data/datasets/", dataset_name, ".h5ad")
adata <- sc$read_h5ad(input_file)
if (!is.null(adata$raw)) {
  adata$layers["counts"] <- adata$raw$X
}
sc$pp$filter_cells(adata, min_genes = 25)
sc$pp$filter_genes(adata, min_cells = 10)
counts_mat <- py_to_r(adata$layers["counts"])
# row: genes, col: cells
seurat_obj <- CreateSeuratObject(counts = t(counts_mat), meta.data = py_to_r(adata$obs))
rownames(seurat_obj) <- row.names(adata$var)

#* filter cells and genes using Seurat
# seurat_obj <- subset(seurat_obj, subset = nFeature_RNA >= 25)
# cat("After filter cells:", dim(seurat_obj), "\n")
# seurat_obj <- seurat_obj[rowSums(seurat_obj@assays$RNA@counts > 0) >= 10, ]
# cat("After filter genes:", dim(seurat_obj), "\n")

# data process and integration
seurat_obj[["RNA"]] <- split(seurat_obj[["RNA"]], f = seurat_obj@meta.data[[batch_col]])
seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj)
obj <- IntegrateLayers(
  object = seurat_obj, method = CCAIntegration,
  orig.reduction = "pca", new.reduction = "integrated.cca",
  verbose = FALSE
)

# save integrated cell embeddings
cca_mat <- obj@reductions$integrated.cca@cell.embeddings
cca_mat <- as.matrix(cca_mat)
cat("Integrated CCA matrix shape: ", dim(cca_mat), "\n")
output_dir <- paste0("output/", dataset_name, "/Seurat_cca")
output_file <- paste0(output_dir, "/cell_emb.npy")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
np$save(output_file, cca_mat)

# Rscript seurat_integration.R pancreas_scib batch
# Rscript seurat_integration.R Immune_all_human_scib batch
# Rscript seurat_integration.R HLCA_core dataset
# Rscript seurat_integration.R Tabula_Sapiens_all tissue_in_publication