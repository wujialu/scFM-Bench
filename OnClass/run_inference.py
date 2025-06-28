import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
import numpy as np
import pandas as pd
import os
from OnClass.OnClassModel import OnClassModel
from utils import read_ontology_file, make_folder, read_data_file, read_exclude_data, parse_pkl, MapLabel2CL, MyDataset, calculate_subtype_acc, seed_everything
from config import ontology_data_dir, scrna_data_dir, result_dir, optuna_result_dir
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import json

    
if len(sys.argv) <= 2:
	device = sys.argv[1]
	model = "scVI"
	source_dname = "HLCA_core"
	dnames = ["HLCA_core"] 
else:
	device = sys.argv[1]
	model = sys.argv[2]
	source_dname = sys.argv[3]
	dnames = sys.argv[4]
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

if model is not None:
	output_dir = make_folder(result_dir+f'/{model}/{celltype_embed}_dot_product')
else:
	output_dir = make_folder(result_dir+f'/Raw')

def main(model_to_params):
	for dname in dnames:
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
		metrics = []
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
	
		for iter in range(niter):
			seed_everything(iter)
			for unseen_ratio in unseen_ratio_ls:
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

				pred_label_co = np.array([OnClass_test_obj.i2co[y] for y in pred_label])
				acc = calculate_subtype_acc(cell_type_network_file, label, pred_label_co)

				# save results
				pred_df = pd.DataFrame()
				pred_df["y_true"] = label
				pred_df["y_pred"] = pred_label_co

				if cross_dataset:
					pred_df.to_csv(os.path.join(make_folder(os.path.join(folder, dname)), "pred_label.csv"), index=False)
					macro_f1 = f1_score(test_Y, pred_label, average="macro")
					metrics = {"iter": iter,
							   "unseen_ratio": unseen_ratio,
							   "Accuracy@1": acc,
							   "Macro F1": macro_f1}
					metrics_df = pd.DataFrame(metrics.items()).set_index(0).T
					metrics_df.to_csv(os.path.join(folder, dname, f"metrics.csv"), index=False)
				else:
					pred_df.to_csv(os.path.join(folder, "pred_label_nonleaf.csv"), index=False)
					metrics_df = pd.read_csv(os.path.join(folder, "metrics.csv"))
					metrics_df["Accuracy@1(nonleaf)"] = acc
					metrics_df.to_csv(os.path.join(folder, "metrics.csv"), index=False)
	
				# nseen = OnClass_test_obj.nseen
				# onto_net = OnClass_test_obj.ontology_dict
				# unseen_l_str = OnClass_test_obj.unseen_co
				# unseen_l = MapLabel2CL(unseen_l_str, OnClass_test_obj.co2i)

				# res = evaluate(pred_Y_all, test_Y, unseen_l, nseen, Y_net = onto_net, write_screen = True, prefix = 'OnClass',
				#       write_to_file=folder, i2co=OnClass_test_obj.i2co) # Y_ind = test_Y_ind

		# if cross_dataset:
		# 	metrics_df = pd.DataFrame(metrics)
		# 	metrics_df.to_csv(Path(folder).parents[1] + f"/metrics_{dname}.csv", index=False)

if __name__ == "__main__":
	params_file = optuna_result_dir + "/model_to_params.json"
	with open(params_file, "r") as f:
		model_to_params = json.load(f)
	main(model_to_params)