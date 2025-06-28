import os
import sys
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from OnClass.OnClassModel import OnClassModel
from utils import read_ontology_file, read_data, make_folder, read_data_file, read_data, SplitTrainTest, seed_everything, read_exclude_data, evaluate, MapLabel2CL, calculate_subtype_acc
from config import ontology_data_dir, scrna_data_dir, result_dir
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix

source_dname = sys.argv[1]
target_dname = sys.argv[2]
model = "HVG"
output_dir = make_folder(result_dir+'/LR_OvR')
emb_file = f"{model}/cell_emb.npy"

# load ontology files
cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file("cl", ontology_data_dir)	
OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)

# load source dataset
feature_file, filter_key, drop_key, label_key, batch_key, label_file, gene_file = read_data_file(source_dname, scrna_data_dir)
feature, genes, label, _, _, remained_terms = read_data(
    feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
    exclude_non_leaf_ontology = True, tissue_key = None, filter_key = filter_key, AnnData_label_key=label_key,
    nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp,
    emb_file = os.path.join(f"../output/{source_dname}", emb_file))

nonleaf_feature, _, nonleaf_label, _, _ = read_exclude_data(
    feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
    tissue_key = None, filter_key = filter_key, AnnData_label_key=label_key,
    nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp,
    emb_file = os.path.join(f"../output/{source_dname}", emb_file),
    target_labels = None)

# load target dataset
feature_file, filter_key, drop_key, label_key, batch_key, label_file, gene_file = read_data_file(target_dname, scrna_data_dir)
target_labels = {'CL:0000860', 'CL:0002543', 'CL:0000875', 'CL:0000077', 'CL:0002062', 'CL:0000786', 'CL:0000186', 
                'CL:0002138', 'CL:0002063', 'CL:0000097', 'CL:0000158', 'CL:0002399', 'CL:0002144', 'CL:0000037'}
target_dataset_feature, _, target_dataset_label, _, _ = read_exclude_data(
    feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
    tissue_key = None, filter_key = filter_key, AnnData_label_key=label_key,
    nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp,
    emb_file = os.path.join(f"../output/{target_dname}", emb_file),
    target_labels = target_labels)

niter = 5 
unseen_ratio_ls = [0.,0.1,0.3,0.5,0.7,0.9]
for iter in range(niter):
    seed_everything(iter)
    for unseen_ratio in unseen_ratio_ls:
        if unseen_ratio == 0.:
            test_ratio = 0.8
        else:
            test_ratio = 0.2
        folder = make_folder(output_dir +'/'+ source_dname + '/' + f"testset_{test_ratio}" + '/' + str(iter) + '/' + str(unseen_ratio) + '/')
        train_feature, train_label, test_feature, test_label, unseen_label = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = iter, nfold_sample=test_ratio)
        OnClass_train_obj.EmbedCellTypes(train_label)
        co2i, i2co = OnClass_train_obj.co2i.copy(), OnClass_train_obj.i2co.copy()
        train_Y = MapLabel2CL(train_label, co2i)  
        test_Y = MapLabel2CL(test_label, co2i)
        unseen_l_str = OnClass_train_obj.unseen_co
        unseen_l = MapLabel2CL(unseen_l_str, co2i)

        # 训练模型
        #! liblinear is toooo slow
        # model = make_pipeline(StandardScaler(), LogisticRegression(multi_class='ovr', solver='liblinear'))
        model = make_pipeline(StandardScaler(), LogisticRegression(multi_class='ovr', solver='saga', n_jobs=-1))
        model.fit(train_feature, train_label)

        # predict on test set
        pred_Y_seen = model.predict_proba(test_feature)  # shape: [n_samples, n_classes]
        max_probs = np.max(pred_Y_seen, axis=1)    # 最大类别的概率 --> confidence score
        pred_label = model.classes_[np.argmax(pred_Y_seen, axis=1)]  # 原始预测

        pred_df = pd.DataFrame()
        pred_df["y_true"] = test_label
        pred_df["y_pred"] = pred_label
        pred_df["y_prob"] = max_probs
        pred_df["unseen"] = unseen_label
        pred_df.to_csv(os.path.join(folder, "pred_label.csv"), index=False)
        
        pred_Y_all = np.zeros((pred_Y_seen.shape[0], OnClass_train_obj.nco)) 
        pred_Y_all[:, :OnClass_train_obj.nseen] = pred_Y_seen
        res_v = evaluate(pred_Y_all, test_Y, unseen_l, OnClass_train_obj.nseen, Y_net = OnClass_train_obj.ontology_dict, 
                         write_screen = True, prefix = 'LR_OvR', i2co = i2co, train_Y = train_Y)
        df = pd.DataFrame(res_v.items()).set_index(0).T
        df.to_csv(os.path.join(folder, "metrics.csv"), index=False)

        if unseen_ratio == 0.:
            print("========== Predict on non-leaf nodes ==========")
            pred_Y_seen = model.predict_proba(nonleaf_feature)  # shape: [n_samples, n_classes]
            max_probs = np.max(pred_Y_seen, axis=1)    # 最大类别的概率 --> confidence score
            pred_label = model.classes_[np.argmax(pred_Y_seen, axis=1)]  # 原始预测

            pred_df = pd.DataFrame()
            pred_df["y_true"] = nonleaf_label
            pred_df["y_pred"] = pred_label
            pred_df["y_prob"] = max_probs
            pred_df.to_csv(os.path.join(folder, "pred_label_nonleaf.csv"), index=False)
            
            acc = calculate_subtype_acc(cell_type_network_file, nonleaf_label, pred_label)
            metrics_df = pd.read_csv(os.path.join(folder, "metrics.csv"))
            metrics_df["Accuracy@1(nonleaf)"] = acc
            metrics_df.to_csv(os.path.join(folder, "metrics.csv"), index=False)

            print("========== Predict on target dataset ==========")
            pred_Y_seen = model.predict_proba(target_dataset_feature)  # shape: [n_samples, n_classes]
            max_probs = np.max(pred_Y_seen, axis=1)    # 最大类别的概率 --> confidence score
            pred_label = model.classes_[np.argmax(pred_Y_seen, axis=1)]  # 原始预测

            pred_df = pd.DataFrame()
            pred_df["y_true"] = target_dataset_label
            pred_df["y_pred"] = pred_label
            pred_df["y_prob"] = max_probs
            pred_df.to_csv(os.path.join(make_folder(os.path.join(folder, target_dname)), "pred_label.csv"), index=False)

            acc = calculate_subtype_acc(cell_type_network_file, target_dataset_label, pred_label)
            macro_f1 = f1_score(target_dataset_label, pred_label, average="macro")
            metrics = {"iter": iter,
                       "unseen_ratio": unseen_ratio,
                       "Accuracy@1": acc,
                       "Macro F1": macro_f1}
            metrics_df = pd.DataFrame(metrics.items()).set_index(0).T
            metrics_df.to_csv(os.path.join(folder, target_dname, f"metrics.csv"), index=False)