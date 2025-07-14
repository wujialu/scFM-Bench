library(Seurat)
library(SeuratDisk)
library(Matrix)
library(reticulate)
library(future)

options(future.globals.maxSize = 40 * 1024^3)
plan("multicore", workers = 4)

# 从命令行传入数据集路径
args <- commandArgs(trailingOnly = TRUE)
ref_dataset <- args[1]
dataset_name <- args[2]

# set env
use_python("/home/wujialu/.conda/envs/singlecell/bin/python")  
use_condaenv("/home/wujialu/.conda/envs/singlecell")  # 使用 conda 环境
data_dir <- "/mnt/nvme/extra_data/wujialu/scFM-Bench/data/datasets/"
sc <- import("scanpy")
np <- import("numpy")

# Load the CCA-integrated reference object you created previously
# This object already contains the 'integrated.cca' reduction
ref_obj <- readRDS(paste0("output/", ref_dataset, "/X/Seurat_cca/reference.rds"))

input_file <- paste0(data_dir, dataset_name, ".h5ad")
adata <- sc$read_h5ad(input_file)
if (!is.null(adata$raw)) {
  adata$layers["counts"] <- adata$raw$X
}
sc$pp$filter_cells(adata, min_genes = 25)
sc$pp$filter_genes(adata, min_cells = 10)
counts_mat <- py_to_r(adata$layers["counts"])
# row: genes, col: cells
query_obj <- CreateSeuratObject(counts = t(counts_mat), meta.data = py_to_r(adata$obs))
rownames(query_obj) <- row.names(adata$var)
query_obj <- NormalizeData(query_obj, normalization.method = "LogNormalize", scale.factor = 10000)

# --- Step 1: Identify the features to use for mapping ---

# The mapping should be based on the "variable features" that were used to build the reference integration.
# First, get these features from your reference object.
ref_variable_features <- VariableFeatures(ref_obj)

# Now, find the intersection: which of these variable features also exist in your query object?
# This creates the list of common genes that can actually be used for comparison.
features_to_use <- intersect(ref_variable_features, rownames(query_obj))

# Check how many features you are left with.
# If this number is too low (e.g., < 200), the mapping quality might be poor.
cat("Number of reference variable features:", length(ref_variable_features), "\n")
cat("Number of features to use for mapping (intersection):", length(features_to_use), "\n")


# --- Step 2: Use the 'features' argument in FindTransferAnchors ---

# Now, run the function, explicitly telling it which genes to use.
anchors <- FindTransferAnchors(
  reference = ref_obj,
  query = query_obj,
  features = features_to_use, # <-- THIS IS THE CRITICAL FIX
  normalization.method = "LogNormalize",
  reference.reduction = "pca",
  dims = 1:30
)

# --- Step 3: Proceed with MapQuery as before ---

query_obj <- MapQuery(
  anchorset = anchors,
  reference = ref_obj,
  query = query_obj,
  reference.reduction = "pca",
  reduction.model = "integrated.cca"
)

# --- Step 4: Extract and save the mapped query cell embeddings ---

# The mapped embeddings are in the 'ref.cca' reduction of the query object
query_embedding <- query_obj@reductions$ref.cca@cell.embeddings
query_embedding <- as.matrix(query_embedding)

cat("Mapped query embedding matrix shape: ", dim(query_embedding), "\n")

# Define the output directory and file path for the query embedding
# The `dataset_name` variable should correspond to your query dataset name
output_dir <- paste0("output/", dataset_name, "/X/Seurat_cca_mapped")
output_file <- paste0(output_dir, "/cell_emb.npy")

# Create the directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save the matrix as a .npy file
np$save(output_file, query_embedding)

cat("Successfully saved mapped query embedding to:", output_file, "\n")