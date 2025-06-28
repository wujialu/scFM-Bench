import os
import sys
import networkx
from utils import calculate_lcad, calculate_lcad_continuous
import parallel
import numpy as np
import pandas as pd
from config import ontology_data_dir, result_dir
import json

# create cell ontology graph
cell_type_network_file = os.path.join(ontology_data_dir, 'cl.ontology.new')
distance_df = pd.read_csv("../data/OnClass_data_public/Ontology_data/cl.ontology.rwr.csv", index_col=0)
G = networkx.DiGraph()
fin = open(cell_type_network_file)
for line in fin:
	s,p = line.strip().split('\t')
	G.add_edge(p, s, weight=distance_df.loc[s,p])
fin.close()

# load prediction results
model_list = ["scVI", "Geneformer", "scGPT", "UCE", "LangCell", "xTrimoGene", "Harmony", "Seurat_cca"]
unseen_ratio_ls = [0]
# unseen_ratio_ls = [0.1, 0.3, 0.5, 0.7, 0.9]
test_ratio = 0.8
celltype_embed = "onclass"
lcad_continuous = False

source_dname = sys.argv[1]
dnames = sys.argv[2]
dnames = dnames.split(',')

def main(model_to_params):
    for dname in dnames:
        if dname != source_dname:
            cross_dataset = True
        else:
            cross_dataset = False

        for model in model_list:
            params = model_to_params[source_dname][model]
            lr, l2 = float(params["lr"]), float(params["l2"])
            output_dir = result_dir + f'/{model}/{celltype_embed}_dot_product/{source_dname}/lr_{lr}_l2_{l2}_testset_{test_ratio}'
            for unseen_ratio in unseen_ratio_ls:
                result_df = pd.DataFrame()

                for iter in range(5):
                    if cross_dataset:
                        pred_file = os.path.join(output_dir, str(iter), str(unseen_ratio), dname, "pred_label.csv")
                        prev_pred_file = os.path.join(output_dir, str(iter), str(unseen_ratio), dname, "pred_label_nonleaf.csv")
                        if os.path.isfile(prev_pred_file):
                            os.rename(prev_pred_file, pred_file)
                            break
                                        
                        metrics_file = os.path.join(output_dir, str(iter), str(unseen_ratio), dname, "metrics.csv")

                    else:
                        pred_file = os.path.join(output_dir, str(iter), str(unseen_ratio), "pred_label.csv")
                        metrics_file = os.path.join(output_dir, str(iter), str(unseen_ratio), "metrics.csv")

                    pred_df = pd.read_csv(pred_file)
                    metrics_df = pd.read_csv(metrics_file)

                    if unseen_ratio == 0:
                        if "LCAD" not in pred_df.columns:
                            incorrect_index = pred_df[pred_df["y_true"]!=pred_df["y_pred"]].index
                            
                            if lcad_continuous:
                                tasks = [(G, distance_df, node_gt, node_pred) for node_gt, node_pred in zip(pred_df.y_true[incorrect_index], pred_df.y_pred[incorrect_index])]
                                results = parallel.map(calculate_lcad_continuous, tasks, n_CPU=12, progress=True)
                                metrics_df["LCAD_continuous"] = np.mean(results)
                                pred_df.loc[incorrect_index, "LCAD_continuous"] = results
                            else:
                                tasks = [(G, node_gt, node_pred) for node_gt, node_pred in zip(pred_df.y_true[incorrect_index], pred_df.y_pred[incorrect_index])]
                                results = parallel.map(calculate_lcad, tasks, n_CPU=10, progress=True)
                                metrics_df["LCAD"] = np.mean(results)
                                pred_df.loc[incorrect_index, "LCAD"] = results

                            pred_df = pred_df.fillna(0)
                            pred_df.to_csv(pred_file, index=False)
                    
                        else:
                            metrics_df["LCAD"] = np.mean(pred_df[pred_df["LCAD"]!=0]["LCAD"])

                    result_df = pd.concat([result_df, metrics_df])

                result_df = result_df.reset_index(drop=True)
                result_df.loc["mean", :] = result_df.mean(0)
                result_df.loc["std", :] = result_df.std(0)
                if cross_dataset:
                    result_df.to_csv(os.path.join(output_dir, f"all_metrics_transfer_{dname}.csv"), index=False)
                else:
                    result_df.to_csv(os.path.join(output_dir, f"all_metrics_unseen_{unseen_ratio}.csv"), index=False)

if __name__ == "__main__":
	params_file = "./results_gridsearch/model_to_params.json"
	with open(params_file, "r") as f:
		model_to_params = json.load(f)
	main(model_to_params)