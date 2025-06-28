import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
import numpy as np
import pandas as pd
import os
from OnClass.OnClassModel import OnClassModel
from utils import read_ontology_file, make_folder, read_data_file, read_exclude_data, parse_pkl, MapLabel2CL, MyDataset, seed_everything
from config import ontology_data_dir, scrna_data_dir, result_dir
from torch.utils.data import DataLoader
import torch
import random
from sklearn.metrics import f1_score
import json
import itertools
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score

    
if len(sys.argv) <= 2:
	device = sys.argv[1]
	source_dname = "HLCA_core"
	dnames = ["Tabula_Sapiens_all"] 
else:
	device = sys.argv[1]
	source_dname = sys.argv[2]
	dnames = sys.argv[3]
	dnames = dnames.split(',')

niter = 5 # 5-fold CV
batch_correct = True
minibatch_size=128
max_iter = 50
train = False
test_ratio = 0.8

# in-distribution evaluation
dot_product = True
celltype_embed = "onclass"
refine = False
unseen_ratio_ls = [0]


def main(dname, model_to_params, model_list, iter, unseen_ratio=0):
	pred_logits = {}
	pred_labels = {}
	for model in model_list:
		output_dir = make_folder(result_dir+f'/Raw')

		params = model_to_params[source_dname][model]
		lr, l2 = float(params["lr"]), float(params["l2"])

		if dname == source_dname:
			cross_dataset = False
			target_labels = None
		else:
			cross_dataset = True
			if dname == "multi_tissue_tumor_part":
				target_labels = {'CL:0000057', 'CL:0000084', 'CL:0000097', 'CL:0000236', 'CL:0000784'}
			else: # overlapped of remained cell types between HLCA and Tabula (#class=14)
				target_labels = {'CL:0000860', 'CL:0002543', 'CL:0000875', 'CL:0000077', 'CL:0002062', 'CL:0000786', 'CL:0000186', 
								'CL:0002138', 'CL:0002063', 'CL:0000097', 'CL:0000158', 'CL:0002399', 'CL:0002144', 'CL:0000037'}
		emb_dir = f"../output/{dname}"
		if model == "xTrimoGene":
			emb_file = f"{model}/mapping_01B-resolution_singlecell_cell_embedding_t4.5_resolution.npy"
		elif model == "scVI":
			if cross_dataset:
				emb_file = f"{model}_surgery/cell_emb.npy"
			else:
				emb_file = f"{model}/cell_emb.npy"
		else:
			emb_file = f"{model}/cell_emb.npy"
		
		cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file(dname, ontology_data_dir)
		OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file, device=device)
		feature_file, filter_key, drop_key, label_key, batch_key, label_file, gene_file = read_data_file(dname, scrna_data_dir, model=model)

		if feature_file.endswith('.pkl'):
			feature, label, genes = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
		elif feature_file.endswith('.h5ad'):
			# nlp_mapping: map cell ontology class according to the text similarity
			# exclude_non_leaf_ontology: remove parent node if child node exist, 防止类别之间重合
			feature, genes, label, _, _ = read_exclude_data(feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
					tissue_key = None, filter_key = filter_key, AnnData_label_key=label_key,
					nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp,
					emb_file = os.path.join(emb_dir, emb_file) if emb_file is not None else None,
					target_labels = target_labels)
			
		seed_everything(iter)
		folder = make_folder(output_dir +'/'+ source_dname + '/' + f"lr_{lr}_l2_{l2}_testset_{test_ratio}" + '/' + str(iter) + '/' + str(unseen_ratio) + '/')
		model_path = folder + 'model'

		if emb_file is None:
			cor_test_feature = OnClass_train_obj.ProcessTestFeature(feature, genes, use_pretrain = model_path, log_transform = True)	
		else:
			cor_test_feature = feature

		print (f'initialize test model. Load the model from {model_path}...')
		OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file, device=device)
		OnClass_test_obj.BuildModel(ngene = cor_test_feature.shape[1], use_pretrain = model_path, dot_product=dot_product)
		# create dataloader
		test_Y = MapLabel2CL(label, OnClass_test_obj.co2i)
		test_dataset = MyDataset(cor_test_feature, test_Y)
		test_loader = DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False, num_workers=10)
		pred_Y_seen, pred_Y_seen_logits, pred_Y_all, pred_label = OnClass_test_obj.Predict(test_loader, use_normalize = False, unseen_ratio = unseen_ratio, refine=refine)
		# np.save(folder+ 'pred_Y_seen.npy',pred_Y_seen)
		# np.save(folder+ 'pred_Y_all.npy',pred_Y_all)

		print ("Shape of pred_Y_seen", np.shape(pred_Y_seen))
		print ("Shape of pred_Y_all", np.shape(pred_Y_all))

		pred_labels[model] = pred_label
		pred_logits[model] = pred_Y_seen_logits
		
	return test_Y, pred_labels, pred_logits


if __name__ == "__main__":
	params_file = "./results_gridsearch/model_to_params.json"
	with open(params_file, "r") as f:
		model_to_params = json.load(f)
	
	model_list = ["Geneformer", "scGPT", "UCE", "LangCell", "xTrimoGene"]
	result = []
	for dname in dnames:
		for iter in range(niter):
			y_true, pred_labels, pred_logits = main(dname, model_to_params, model_list, iter)

			#! full ensemble 
			#* Setting 1: aggregate logits
			for i, model in enumerate(model_list):
				if i == 0:
					all_logits = pred_logits[model]
				else:
					all_logits += pred_logits[model]
			y_pred = softmax(all_logits, axis=1).argmax(-1)
			
			accuracy = accuracy_score(y_true, y_pred)
			macro_f1 = f1_score(y_true, y_pred, average="macro")
			result.append(
				{
					"iter": iter,
					"model1": "all_logits",
					"model2": "all_logits",
					"Accuracy@1": accuracy,
					"Macro F1": macro_f1
				}
			)

			#* Setting 2: majority voting
			from scipy.stats import mode
			for i, model in enumerate(model_list):
				if i == 0:
					y_pred = pred_labels[model].reshape(-1,1) # [n_samples, 1]
				else:
					y_pred = np.concatenate((y_pred, pred_labels[model].reshape(-1,1)), axis=1) # [n_samples, n_models]

			y_pred, _ = mode(y_pred, axis=1, keepdims=False) # [n_samples, ]

			accuracy = accuracy_score(y_true, y_pred)
			macro_f1 = f1_score(y_true, y_pred, average="macro")
			result.append(
				{
					"iter": iter,
					"model1": "all_voting",
					"model2": "all_voting",
					"Accuracy@1": accuracy,
					"Macro F1": macro_f1
				}
			)

			#! pairwise ensemble
			for (model1, model2) in list(itertools.combinations(model_list, 2)):
				y_pred = softmax(pred_logits[model1] + pred_logits[model2], axis=1).argmax(-1)

				accuracy = accuracy_score(y_true, y_pred)
				macro_f1 = f1_score(y_true, y_pred, average="macro")

				result.append(
					{
						"iter": iter,
						"model1": model1,
						"model2": model2,
						"Accuracy@1": accuracy,
						"Macro F1": macro_f1
					}
				)
		result_df = pd.DataFrame.from_dict(result)
		result_df.to_csv(f"../figures/cell_annotation/model_ensemble/model_complement_transfer_{source_dname}_to_{dname}.csv",index=False)