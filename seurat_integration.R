library(Seurat)
library(SeuratDisk)
library(Matrix)
library(reticulate)
library(future)

options(future.globals.maxSize = 120 * 1024^3)
# plan("multicore", workers = 4)

# 从命令行传入数据集路径
args <- commandArgs(trailingOnly = TRUE)
dataset_name <- args[1]
batch_col <- args[2]

# set env
use_python("/home/wujialu/.conda/envs/singlecell/bin/python")  
use_condaenv("/home/wujialu/.conda/envs/singlecell")  # 使用 conda 环境
data_dir <- "/mnt/nvme/extra_data/wujialu/scFM-Bench/data/datasets/"
sc <- import("scanpy")
np <- import("numpy")

# load Anndata and transform to Seurat object
input_file <- paste0(data_dir, dataset_name, ".h5ad")
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
cat("Calling IntegrateLayers...\n")
obj <- IntegrateLayers(
  object = seurat_obj, method = CCAIntegration,
  orig.reduction = "pca", new.reduction = "integrated.cca",
  verbose = FALSE
)
cat("Integration completed.\n")

# 保存 reference 对象
output_dir <- paste0("output/", dataset_name, "/X/Seurat_cca")
saveRDS(obj, file = paste0(output_dir, "/reference.rds"))

# save integrated cell embeddings
cca_mat <- obj@reductions$integrated.cca@cell.embeddings
cca_mat <- as.matrix(cca_mat)
cat("Integrated CCA matrix shape: ", dim(cca_mat), "\n")
output_file <- paste0(output_dir, "/cell_emb.npy")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
np$save(output_file, cca_mat)