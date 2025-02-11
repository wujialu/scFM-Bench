import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
import numpy as np
import pandas as pd
import os
from OnClass.OnClassModel import OnClassModel
from utils import read_ontology_file, read_data, make_folder, read_data_file, read_data, parse_pkl, SplitTrainTest, MapLabel2CL, evaluate, MyDataset
from config import ontology_data_dir, scrna_data_dir, result_dir, optuna_result_dir
from torch.utils.data import DataLoader
import torch
import random
from scipy.special import logsumexp
import json


def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if len(sys.argv) <= 2:
	device = sys.argv[1]
	model = None
	dnames = ["Diabetic_Kidney_Disease"] 
else:
	device = sys.argv[1]
	model = sys.argv[2]
	dnames = sys.argv[3]
	dnames = dnames.split(",")
 
    
niter = 5 
batch_correct = True
minibatch_size = 128
max_iter = 50
train = True

# OOD evaluation
dot_product = True 
celltype_embed = "onclass"
refine = False
test_ratio = 0.2
unseen_ratio_ls = [0.1,0.3,0.5,0.7,0.9]

# in-distribution evaluation
# dot_product = True 
# refine = False
# unseen_ratio_ls = [0]

if model is not None:
	output_dir = make_folder(result_dir+f'/{model}/{celltype_embed}_dot_product')
	if model == "xTrimoGene":
		emb_file = f"{model}/mapping_01B-resolution_singlecell_cell_embedding_t4.5_resolution.npy"
	else:
		emb_file = f"{model}/cell_emb.npy"
else:
    output_dir = make_folder(result_dir+f'/Raw')

def main(model_to_params):
	for dname in dnames:
		params = model_to_params[dname][model]
		lr, l2 = float(params["lr"]), float(params["l2"])
		
		emb_dir = f"../output/{dname}"
		cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file("cl", ontology_data_dir)	
	
		OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file, device=device)
		feature_file, filter_key, drop_key, label_key, label_file, gene_file = read_data_file(dname, scrna_data_dir)

		if feature_file.endswith('.pkl'):
			feature, label, genes = parse_pkl(feature_file, label_file, gene_file, exclude_non_leaf_ontology = True, cell_ontology_file = cell_type_network_file)
		elif feature_file.endswith('.h5ad'):
			# nlp_mapping: map cell ontology class according to the text similarity
			# exclude_non_leaf_ontology: remove parent node if child node exist, 防止类别之间重合
			feature, genes, label, _, _, _ = read_data(feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
					exclude_non_leaf_ontology = True, tissue_key = None, filter_key = filter_key, AnnData_label_key=label_key,
					nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp,
					emb_file = os.path.join(emb_dir, emb_file) if emb_file is not None else None)
	
		for iter in range(niter):
			seed_everything(iter)
			for unseen_ratio in unseen_ratio_ls:
				folder = make_folder(output_dir +'/'+ dname + '/' + f"lr_{lr}_l2_{l2}_testset_{test_ratio}" + '/' + str(iter) + '/' + str(unseen_ratio) + '/')
				train_feature, train_label, test_feature, test_label, unseen_label = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = iter, nfold_sample=test_ratio)
				train_genes = genes
				test_genes = genes
				if celltype_embed == "DAGFormer":
					OnClass_train_obj.CreateOntoGraph(train_label)
				OnClass_train_obj.EmbedCellTypes(train_label)
				nseen = OnClass_train_obj.nseen
				co2i, i2co = OnClass_train_obj.co2i.copy(), OnClass_train_obj.i2co.copy()
				# co2i["unseen"] = OnClass_train_obj.nco
				# i2co[OnClass_train_obj.nco] = "unseen"
				model_path = folder + 'model'
				print (f'generate pretrain model. Save the model to {model_path}...')
				if emb_file is None:
					cor_train_feature, cor_test_feature, cor_train_genes, cor_test_genes = \
						OnClass_train_obj.ProcessTrainFeature(train_feature, train_label, train_genes, test_feature = test_feature, test_genes = test_genes, 
															batch_correct = batch_correct, log_transform = True)		
					nhidden = [1000]
				else:
					cor_train_feature, cor_test_feature = train_feature, test_feature
					cor_train_genes, cor_test_genes = None, None
					OnClass_train_obj.genes = None
					# nhidden = [1000]
					nhidden = [512, 1024]
		
				print("Shape of train feature:", cor_train_feature.shape)
				print("Shape of test feature:", cor_test_feature.shape)	
				# print(np.shape(cor_train_genes), np.shape(cor_test_genes)))
		
				# split train/valid dataset
				nx = np.shape(cor_train_feature)[0]
				ntrain = int(nx*0.9)
				permutation = list(np.random.permutation(nx))
				train_ind = permutation[:ntrain]
				valid_ind = permutation[ntrain:]
		
				# create dataloader
				train_Y = MapLabel2CL(train_label, co2i)
				test_Y = MapLabel2CL(test_label, co2i)
				train_dataset = MyDataset(cor_train_feature[train_ind, :], train_Y[train_ind])
				valid_dataset = MyDataset(cor_train_feature[valid_ind, :], train_Y[valid_ind])
				train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True, num_workers=10)
				valid_loader = DataLoader(valid_dataset, batch_size=minibatch_size, shuffle=False, num_workers=10)
	
				OnClass_train_obj.BuildModel(ngene = cor_train_feature.shape[1], nhidden=nhidden, lr=lr, l2=l2, dot_product=dot_product)
				OnClass_train_obj.train_feature_mean = np.mean(cor_train_feature, axis = 0) # used if batch_correct=True
				
				if train or not os.path.exists(model_path + '.pt'):
					best_valid_loss = 1e6
					patience = 0
					for epoch in range(max_iter):
						train_epoch_loss, valid_epoch_loss = OnClass_train_obj.Train(train_loader, valid_loader)
						print(f"Epoch: {epoch}, train_loss: {train_epoch_loss}, val_loss: {valid_epoch_loss}")
						if valid_epoch_loss < best_valid_loss:
							best_valid_loss = valid_epoch_loss
							OnClass_train_obj.save_model(model_path=model_path)
							patience = 0
						else:
							patience += 1
						if patience >= 5:
							break
		
				print (f'initialize test model. Load the model from {model_path}...')
				OnClass_test_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file, device=device)
				OnClass_test_obj.BuildModel(ngene = cor_train_feature.shape[1], use_pretrain = model_path, dot_product=dot_product)
				if emb_file is None:
					cor_test_feature = OnClass_train_obj.ProcessTestFeature(cor_test_feature, cor_test_genes, use_pretrain = model_path, 
																			batch_correct = batch_correct, log_transform = False)	
				test_dataset = MyDataset(cor_test_feature, test_Y)
				test_loader = DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False, num_workers=10)
				pred_Y_seen, pred_Y_seen_logits, pred_Y_all, pred_Y_all_new, seen_ind, seen_ind_before_refine, unseen_conf, unseen_conf_before_refine = OnClass_test_obj.Predict(test_loader, use_normalize = False, unseen_ratio = unseen_ratio, refine=refine)

				#! save predictions
				Y_truth_co = np.array([i2co[y] for y in test_Y])
				Y_pred_co = np.array([i2co[y] for y in np.argmax(pred_Y_seen, axis=1)])
				Y_pred_all_co = np.array([i2co[y] for y in np.argmax(pred_Y_all, axis=1)])
				pred_df = pd.DataFrame()
				pred_df["y_true"] = Y_truth_co
				pred_df["y_pred"] = Y_pred_co
				pred_df["y_pred_all"] = Y_pred_all_co
				pred_df["softmax_score"] = np.max(pred_Y_seen, axis=1)
				pred_df["energy_score"] = - logsumexp(pred_Y_seen_logits, axis=1)
				pred_df["negative_energy"] = - pred_df["energy_score"]
				pred_df["unseen"] = unseen_label
				pred_df.to_csv(os.path.join(folder, "pred_label.csv"), index=False)

				print ("Shape of pred_Y_seen", np.shape(pred_Y_seen))
				print ("Shape of pred_Y_all", np.shape(pred_Y_all))

				#! save metrics
				# onto_net = OnClass_train_obj.ontology_dict
				# unseen_l_str = OnClass_train_obj.unseen_co
				# # unseen_l_str = ["unseen"]
				# unseen_l = MapLabel2CL(unseen_l_str, co2i)
				# test_Y_ind = np.sort(np.array(list(set(test_Y) |  set(train_Y))))
				
				# v1
				# res_v = evaluate(pred_Y_all, test_Y, unseen_l, nseen, Y_net = onto_net, write_screen = True, prefix = 'OnClass', i2co=i2co,
				#         		 Y_ind = test_Y_ind) 
				# res_v["AUROC(OOD)_with_refine"] = auroc_ood_with_refine
				# res_v["AUPRC(OOD)_with_refine"] = auprc_ood_with_refine
				# res_v["AUROC(OOD)_wo_refine"] = auroc_odd_wo_refine
				# res_v["AUPRC(OOD)_wo_refine"] = auprc_odd_wo_refine
				# df = pd.DataFrame.from_dict(res_v)
				# df.to_csv(os.path.join(folder, "metrics.csv"), index=False)

				# v2
				# pred_Y_all[seen_ind, :nseen] = pred_Y_seen[seen_ind]
				# res_v = evaluate(pred_Y_all, test_Y, unseen_l, nseen, Y_net = onto_net, write_screen = True, prefix = 'OnClass', i2co=i2co,
				# 				 Y_ind = test_Y_ind) 
				# df = pd.DataFrame.from_dict(res_v)
				# df.to_csv(os.path.join(folder, "metrics_origin.csv"), index=False)

				# # v3
				# res_v = evaluate(pred_Y_all_new, test_Y, unseen_l, nseen, Y_net = onto_net, write_screen = True, prefix = 'OnClass', i2co=i2co,
				# 				 Y_ind = test_Y_ind) 
				# df = pd.DataFrame.from_dict(res_v)
				# df.to_csv(os.path.join(folder, "metrics_unseen_conf_before_refine.csv"), index=False)
				
				# # v4 (slightly better than v3)
				# pred_Y_all_new[seen_ind_before_refine, :nseen] = pred_Y_seen[seen_ind_before_refine]
				# res_v = evaluate(pred_Y_all_new, test_Y, unseen_l, nseen, Y_net = onto_net, write_screen = True, prefix = 'OnClass', i2co=i2co,
				# 				 Y_ind = test_Y_ind) 
				# df = pd.DataFrame.from_dict(res_v)
				# df.to_csv(os.path.join(folder, "metrics_origin_unseen_conf_before_refine.csv"), index=False)

if __name__ == "__main__":
	params_file = optuna_result_dir + "/model_to_params.json"
	with open(params_file, "r") as f:
		model_to_params = json.load(f)
	main(model_to_params)