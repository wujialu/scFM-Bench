import os
import sys
import scanpy as sc
import pandas as pd
import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
from cuml.linear_model import LogisticRegression
from cuml.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from OnClass.OnClassModel import OnClassModel
from utils import read_ontology_file, read_data, make_folder, read_data_file, read_data, SplitTrainTest, seed_everything, read_exclude_data, evaluate, MapLabel2CL, calculate_subtype_acc
from config import ontology_data_dir, scrna_data_dir, result_dir, cell_emb_dir
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix

#! Logistic Regression can not be transfered to new datasets (input genes are different)
dname = sys.argv[1]
model = "HVG"
output_dir = make_folder(result_dir+'/LR_OvR')
emb_file = f"{model}/cell_emb.npy"

# load ontology files
cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file = read_ontology_file("cl", ontology_data_dir)	
OnClass_train_obj = OnClassModel(cell_type_nlp_emb_file = cell_type_nlp_emb_file, cell_type_network_file = cell_type_network_file)

# load dataset
data_info_dict = read_data_file(dname, scrna_data_dir)
feature_file = data_info_dict['feature_file']
label_file = data_info_dict['label_file']
gene_file = data_info_dict['gene_file']
filter_key = data_info_dict['filter_key']
label_key = data_info_dict['label_key']
layer_key = data_info_dict['layer_key']
emb_dir = os.path.join(cell_emb_dir, dname, layer_key)
feature, genes, label, _, _, remained_terms = read_data(
    feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
    exclude_non_leaf_ontology = True, tissue_key = None, filter_key = filter_key, AnnData_label_key=label_key,
    nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp,
    emb_file = os.path.join(emb_dir, emb_file))

nonleaf_feature, _, nonleaf_label, _, _ = read_exclude_data(
    feature_file, cell_ontology_ids = OnClass_train_obj.cell_ontology_ids,
    tissue_key = None, filter_key = filter_key, AnnData_label_key=label_key,
    nlp_mapping = False, cl_obo_file = cl_obo_file, cell_ontology_file = cell_type_network_file, co2emb = OnClass_train_obj.co2vec_nlp,
    emb_file = os.path.join(emb_dir, emb_file),
    target_labels = None)

niter = 5 
unseen_ratio_ls = [0,0.1,0.3,0.5,0.7,0.9]
for iter in range(niter):
    seed_everything(iter)
    for unseen_ratio in unseen_ratio_ls:
        if unseen_ratio == 0.:
            test_ratio = 0.8
        else:
            test_ratio = 0.2
        folder = make_folder(output_dir +'/'+ dname + '/' + f"testset_{test_ratio}" + '/' + str(iter) + '/' + str(unseen_ratio) + '/')
        train_feature, train_label, test_feature, test_label, unseen_label = SplitTrainTest(feature, label, nfold_cls = unseen_ratio, random_state = iter, nfold_sample = test_ratio)
        OnClass_train_obj.EmbedCellTypes(train_label)
        co2i, i2co = OnClass_train_obj.co2i.copy(), OnClass_train_obj.i2co.copy()
        train_Y = MapLabel2CL(train_label, co2i)  
        test_Y = MapLabel2CL(test_label, co2i)
        unseen_l_str = OnClass_train_obj.unseen_co
        unseen_l = MapLabel2CL(unseen_l_str, co2i)

        # 训练模型
        #* sklearn
        # model = make_pipeline(StandardScaler(), LogisticRegression(multi_class='ovr', solver='liblinear'))
        # model = make_pipeline(StandardScaler(), LogisticRegression(multi_class='ovr', solver='saga', n_jobs=-1))
        #* cuml
        model = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2'))
        model.fit(train_feature, train_Y)

        # predict on test set
        pred_Y_seen = model.predict_proba(test_feature)  # shape: [n_samples, n_classes]
        max_probs = np.max(pred_Y_seen, axis=1)    # 最大类别的概率 --> confidence score
        # pred_label_co = model.classes_[np.argmax(pred_Y_seen, axis=-1)]  # 原始预测
        pred_label = np.argmax(pred_Y_seen, axis=-1)
        pred_label_co = np.array([OnClass_train_obj.i2co[y] for y in pred_label])

        pred_df = pd.DataFrame()
        pred_df["y_true"] = test_label
        pred_df["y_pred"] = pred_label_co
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
            # pred_label_co = model.classes_[np.argmax(pred_Y_seen, axis=-1)]  # 原始预测
            pred_label = np.argmax(pred_Y_seen, axis=-1)
            pred_label_co = np.array([OnClass_train_obj.i2co[y] for y in pred_label])

            pred_df = pd.DataFrame()
            pred_df["y_true"] = nonleaf_label
            pred_df["y_pred"] = pred_label_co
            pred_df["y_prob"] = max_probs
            pred_df.to_csv(os.path.join(folder, "pred_label_nonleaf.csv"), index=False)
            
            acc = calculate_subtype_acc(cell_type_network_file, nonleaf_label, pred_label_co)
            metrics_df = pd.read_csv(os.path.join(folder, "metrics.csv"))
            metrics_df["Accuracy@1(nonleaf)"] = acc
            metrics_df.to_csv(os.path.join(folder, "metrics.csv"), index=False)