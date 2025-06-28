import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
from scipy.sparse.csr import csr_matrix
from sklearn.preprocessing import normalize
from scipy import stats
import warnings
from torch.optim import optimizer
from OnClass.OnClass_utils import *
from OnClass.BilinearNN import *
from config import ontology_data_dir
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import to_undirected
import networkx as nx
from sklearn.mixture import GaussianMixture


class OnClassModel:
	"""
	PyTorch implementation of the OnClass model
	"""
	def __init__(self, cell_type_network_file='../../OnClass_data/cell_ontology/cl.ontology', cell_type_nlp_emb_file='../../OnClass_data/cell_ontology/cl.ontology.nlp.emb', 
              	 memory_saving_mode=False, device="cuda"):
		"""
		Initialize OnClass model with a given cell-type network and cell-type embedding file.
		Also, you may set the memory_saving_mode to True to get a model that uses less RAM.
		"""
		self.cell_type_nlp_emb_file = cell_type_nlp_emb_file
		self.cell_type_network_file = cell_type_network_file
		self.co2co_graph, self.co2co_nlp, self.co2vec_nlp, self.cell_ontology_ids, self.adj_list_dag, self.adj_list = read_cell_type_nlp_network(self.cell_type_nlp_emb_file, self.cell_type_network_file)
		self.mode = memory_saving_mode
		self.device = device
		self.onto_graph = None
  
	def CreateOntoGraph(self, train_Y_str):
		self.unseen_co, self.co2i, self.i2co, self.ontology_dict, self.ontology_mat = creat_cell_ontology_matrix(train_Y_str, self.co2co_graph, self.cell_ontology_ids, dfs_depth = 3)
  
		node_vec_sorted = torch.from_numpy(np.array([self.co2vec_nlp[co] for co in self.i2co.values()])).float()

		adj_list = [[self.co2i[x[0]], self.co2i[x[1]]] for x in self.adj_list]
		adj_list_dag = [[self.co2i[x[0]], self.co2i[x[1]]] for x in self.adj_list_dag]
		edge_index = torch.tensor(adj_list, dtype=torch.long).t()
		edge_index_dag = torch.tensor(adj_list_dag, dtype=torch.long).t()
		pyg_graph = Data(x=node_vec_sorted, edge_index=edge_index)
  
		# calculate node depth (for DAG-aware position embedding)
		graph_size=pyg_graph.num_nodes
		edge_index=edge_index_dag
		node_ids = np.arange(graph_size, dtype=int)
		node_order = np.zeros(graph_size, dtype=int)
		unevaluated_nodes = np.ones(graph_size, dtype=bool)
		# print(unevaluated_nodes,edge_index)
		# print(edge_index)
		if(edge_index.shape[0]==0):
			pe = torch.from_numpy(np.array(range(graph_size))).long()
			pyg_graph.abs_pe = pe
		else:
			parent_nodes = edge_index[0]
			child_nodes = edge_index[1]

			n = 0
			while unevaluated_nodes.any():
				# Find which parent nodes have not been evaluated
				unevaluated_mask = unevaluated_nodes[parent_nodes]
				if(unevaluated_mask.shape==()):
					unevaluated_mask=np.array([unevaluated_mask])
				# Find the child nodes of unevaluated parents
				unready_children = child_nodes[unevaluated_mask]

				# Mark nodes that have not yet been evaluated
				# and which are not in the list of children with unevaluated parent nodes
				nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_children)

				node_order[nodes_to_evaluate] = n
				unevaluated_nodes[nodes_to_evaluate] = False

				n += 1
			
			pe = torch.from_numpy(node_order).long()
			pyg_graph.abs_pe = pe

		pyg_graph.edge_index_dag = edge_index_dag
		data_new = Data(x=pyg_graph.x, edge_index=edge_index_dag)
		DG = to_networkx(data_new) 
		
		# Compute DAG transitive closures
		TC = nx.transitive_closure_dag(DG)

		# TC_copy = TC.copy()
		# for edge in TC_copy.edges():
		#     if(nx.shortest_path_length(DG,source=edge[0],target=edge[1])>1000):
		#         TC.remove_edge(edge[0], edge[1])
				
		# add k-hop-neighborhood
		# for node_idx in range(pyg_graph.num_nodes):
		#     sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
		#         node_idx, 
		#         3, 
		#         pyg_graph.edge_index,
		#         relabel_nodes=True, 
		#         num_nodes=pyg_graph.num_nodes
		#         )
		#     for node in sub_nodes:
		#         TC.add_edge(node_idx, node.item())
		
		data_new = from_networkx(TC)
		edge_index_dag = data_new.edge_index
  
		# DAG receptive fields
		pyg_graph.dag_rr_edge_index = to_undirected(edge_index_dag)
		self.onto_graph = Batch.from_data_list([pyg_graph])
  
	def EmbedCellTypes(self, train_Y_str, dim=5, emb_method=3, use_pretrain = None, write2file=None):
		"""
		Embed the cell ontology
		Parameters
		----------
		cell_type_network_file : each line should be cell_type_1\tcell_type_2\tscore for weighted network or cell_type_1\tcell_type_2 for unweighted network
		dim: `int`, optional (500)
			Dimension of the cell type embeddings
		emb_method: `int`, optional (3)
			dimensionality reduction method
		use_pretrain: `string`, optional (None)
			use pretrain file. This should be the numpy file of cell type embeddings. It can read the one set in write2file parameter.
		write2file: `string`, optional (None)
			write the cell type embeddings to this file path
		Returns
		-------
		co2emb, co2i, i2co
			returns three dicts, cell type name to embeddings, cell type name to cell type id and cell type id to embeddings.
		"""

		self.unseen_co, self.co2i, self.i2co, self.ontology_dict, self.ontology_mat = creat_cell_ontology_matrix(train_Y_str, self.co2co_graph, self.cell_ontology_ids, dfs_depth = 3)
		self.nco = len(self.i2co)
		Y_emb = emb_ontology(self.i2co, self.ontology_mat, dim = dim, mi=emb_method, co2co_nlp = self.co2co_nlp, unseen_l = self.unseen_co)
		print("Shape of Y_emb", Y_emb.shape)
		self.co2emb = np.column_stack((np.eye(self.nco), Y_emb)) # [2743, 2743+5]
		self.nunseen = len(self.unseen_co)
		self.nseen = self.nco - self.nunseen
		self.co2vec_nlp_mat = np.zeros((self.nco, len(self.co2vec_nlp[self.i2co[0]])))
		for i in range(self.nco):
			self.co2vec_nlp_mat[i,:] = self.co2vec_nlp[self.i2co[i]]
		return self.co2emb, self.co2i, self.i2co, self.ontology_mat

	def BuildModel(self, ngene, nhidden=[1000], lr=1e-4, l2=5e-3, use_pretrain=None, dot_product=True, class_weights=None):
		"""
		Initialize the model or use the pretrain model
		Parameters
		----------
		ngene: `int`
			Number of genes
		nhidden: `list`, optional ([1000])
			Gives the hidden dimensions of the model
		use_pretrain: `string`, optional (None)
			File name of the pretrained model
		Returns
		-------
		"""
		self.ngene = ngene
		self.use_pretrain = use_pretrain
		self.nhidden = nhidden
		#self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		if use_pretrain is not None:
			# Load all the OnClassModel state
			npzfile = np.load(use_pretrain+'.npz',allow_pickle=True)
			self.co2i = npzfile['co2i'].item()
			self.i2co = npzfile['i2co'].item()
			self.genes = npzfile['genes']
			self.co2emb = npzfile['co2emb']
			# self.ngene = len(self.genes),
			self.ontology_mat = npzfile['ontology_mat']
			self.nco = npzfile['nco']
			self.nseen = npzfile['nseen']
			self.co2vec_nlp_mat = npzfile['co2vec_nlp_mat']
			self.nhidden = npzfile['nhidden']
			self.ontology_dict = npzfile['ontology_dict'].item()
			self.train_feature_mean = npzfile['train_feature_mean']
			if os.path.isfile(use_pretrain+'_onto_graph.pt'):
				self.onto_graph = torch.load(use_pretrain+'_onto_graph.pt')

		self.model = BilinearNN(self.co2emb, self.nseen, self.ngene, use_pretrain = use_pretrain, 
                          		lr=lr, l2=l2, nhidden=self.nhidden, memory_saving_mode = self.mode,
                            	dot_product=dot_product, onto_graph=self.onto_graph, class_weights=class_weights)
		
		if use_pretrain is not None:
			# Load the actual PyTorch model parameters
			self.model.load_state_dict(torch.load(use_pretrain + '.pt', map_location="cpu"))
		self.model.to(self.device)
		return self.model

	def ProcessTrainFeature(self, train_feature, train_label, train_genes, test_feature = None, test_genes = None, batch_correct = False, log_transform = True):
		"""
		Process the gene expression matrix used to train the model, and optionally the test data.
		Parameters
		----------
		train_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
			gene expression matrix of cell types
		train_label: `numpy.ndarray`
			labels for the training features
		train_genes: `list`
			list of genes used during training
		test_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode), optional (None)
			gene expression matrix of cell types for the test set
		test_genes: `list`, optional (None)
			list of genes used in test set
		batch_correct: `bool`, optional (False)
			whether to correct for batch effect in data
		log_transform:`bool`, optional (True)
			whether to apply log transform to data
		Returns
		-------
		train_feature, test_feature, self.genes, self.genes
			returns the training feature gene expression matrix and the list of genese associated
			with it. If test_feature was not none, also returns the test features and their genes.
		"""
		
		if log_transform is False and np.max(train_feature) > 1000:
			warnings.warn("Max expression is"+str(np.max(train_feature))+'. Consider setting log transform = True\n')
		self.genes = train_genes
		# batch correction is currently not supported for memory_saving_mode
		if batch_correct and test_feature is not None and test_genes is not None and not self.mode:
			train_feature, test_feature, selected_train_genes = process_expression(train_feature, test_feature, train_genes, test_genes)
			self.genes = selected_train_genes
		elif log_transform:
			if self.mode:
				train_feature = csr_matrix.log1p(train_feature)
			else:
				train_feature = np.log1p(train_feature)

			if test_feature is not None:
				if self.mode:
					test_feature = csr_matrix.log1p(test_feature)
				else:
					test_feature = 	np.log1p(test_feature)
		self.train_feature = train_feature
		self.train_label = train_label
		if test_feature is not None:
			return train_feature, test_feature, self.genes, self.genes
		else:
			return train_feature, self.genes

	def ProcessTestFeature(self, test_feature, test_genes, use_pretrain = None, batch_correct = False, log_transform = True):
		"""
		Process the gene expression matrix used to test the model.
		Parameters
		----------
		test_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
			gene expression matrix of cell types for the test set
		test_genes: `list`
			list of genes used in test set
		use_pretrain: `string`, optional (None)
			name of the pretrained model
		batch_correct: `bool`, optional (False)
			whether to correct for batch effect in data
		log_transform:`bool`, optional (True)
			whether to apply log transform to data
		Returns
		-------
		gene_mapping or test_feature
			processes the test features and returns a data structure that encodes the gene
			expression matrix that should be used for testing. If the model is in memory saving
			mode, then the function will return a tuple of gene expression matrix and index array,
			otherwise, it will just return the matrix.
		"""		
		if log_transform is False and np.max(test_feature) > 1000:
			warnings.warn("Max expression is"+str(np.max(test_feature))+'. Consider setting log transform = True\n')
		
		if use_pretrain is not None:
			if log_transform:
				test_feature = np.log1p(test_feature)
			if batch_correct and not self.mode:
				test_feature = mean_normalization(self.train_feature_mean, test_feature)

		if self.mode:
			gene_mapping = get_gene_mapping(test_genes, self.genes)
			return test_feature, gene_mapping
		else:
			test_feature = map_genes(test_feature, test_genes, self.genes, memory_saving_mode=self.mode)
			return test_feature


	def Train(self, train_loader, valid_loader):
		"""
		Train the model or use the pretrain model
		Parameters
		----------
		train_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
			gene expression matrix of cell types. Should be a cell by gene expression matrix
		train_label: `numpy.ndarray`
			labels for the training features. Should be a cell by class number binary matrix.
		save_model: `string`, optional (None)
			name of file to save model to.
		max_iter: `int`, optional (15)
			number of epochs to train for.
		minibatch_size: `int`, optional (128)
			size of each minibatch.
		"""
		if self.use_pretrain:
			print ('Use pretrained model: ', self.use_pretrain)
			return
		
		# train_label = [self.co2i[tp] for tp in train_label]

		# if use_device:
		# 	print("Using", self.device, "for training")
		# 	self.model.to(self.device)

		# self.train_feature_mean = np.mean(train_feature, axis = 0)
		# self.model.read_training_data(train_feature, train_label)
		# self.model.optimize(max_iter = max_iter, mini_batch_size = minibatch_size, device=self.device)

		self.model.train()
		train_epoch_loss = self.model.train_epoch(train_loader, device=self.device)
		self.model.eval()
		valid_epoch_loss = self.model.eval_epoch(valid_loader, device=self.device)
		return train_epoch_loss, valid_epoch_loss
		
	def save_model(self, model_path):
		torch.save(self.model.state_dict(), model_path + '.pt')
		if self.onto_graph is not None:
			torch.save(self.onto_graph, model_path + "_onto_graph.pt")
		save_model_file = model_path + '.npz'
		np.savez(save_model_file, co2i = self.co2i, co2emb = self.co2emb, nhidden = self.nhidden, i2co = self.i2co, genes = self.genes, nco = self.nco, nseen = self.nseen,
				 ontology_mat = self.ontology_mat, co2vec_nlp_mat = self.co2vec_nlp_mat, ontology_dict = self.ontology_dict, train_feature_mean = self.train_feature_mean)
	
		
	def Predict(self, test_loader, use_normalize = False, refine = True, unseen_ratio = 0.1):
		"""
		Predict the label for new cells
		Parameters
		----------
		test_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
			gene expression matrix of cell types for the test set
		test_genes: `list`, optional (None)
			list of genes used in test set
		"""
		# if use_device:
		# 	 print("Using", self.device, "for predicting")
		# 	 self.model.to(self.device)

		# if test_genes is not None:
		# 	if not self.mode:
		# 		test_feature = map_genes(test_feature, test_genes, self.genes, memory_saving_mode=self.mode)
		# 	else:
		# 		mapping = get_gene_mapping(test_genes, self.genes)
		# else:
		# 	assert(np.shape(test_feature)[1] == self.ngene)

		self.model.eval()
		# with torch.no_grad():
		# 	if not self.mode:
		# 		test_Y_pred_seen = torch.nn.functional.softmax(self.model.forward(test_feature, training=False, device=self.device), dim=1)
		# 	else: # The test set will be in sparse matrix format
		# 		num_batches = 20
		# 		test_Y_pred_seen = torch.nn.functional.softmax(self._batch_predict(test_feature, num_batches, mapping=mapping), dim=1)
		# test_Y_pred_seen = test_Y_pred_seen.detach().cpu()

		test_Y_pred_seen, logits = self.model.test_epoch(test_loader, device=self.device)
  
		if refine:
			ratio = (self.nco*1./self.nseen)**2
			# network = create_propagate_networks_using_nlp(self.co2i, self.ontology_dict, self.ontology_mat, self.co2vec_nlp_mat)
			network = np.load(os.path.join(ontology_data_dir, "networks.npy"))
			test_Y_pred_all = extend_prediction_2unseen(test_Y_pred_seen, network, self.nseen, ratio = ratio, use_normalize = use_normalize)
			#! Q1: which confidence is better --> before refine is better
			unseen_confidence = np.max(test_Y_pred_all[:,self.nseen:], axis=1) - np.max(test_Y_pred_all[:,:self.nseen], axis=1)
			unseen_confidence_before_refine = 1 - np.max(test_Y_pred_seen, axis=1)
		else:
			test_Y_pred_all = np.zeros((test_Y_pred_seen.shape[0], self.nco)) # the last column is the "unseen" class
			test_Y_pred_all[:, :self.nseen] = test_Y_pred_seen
			unseen_confidence = 1 - np.max(test_Y_pred_seen, axis=1)
			unseen_confidence_before_refine = unseen_confidence
   
		if unseen_ratio == 0:
			return test_Y_pred_seen, logits, test_Y_pred_all, test_Y_pred_seen.argmax(-1)
		else: #! Q2: unseen ratio is acturally not know in real-world applications
			nexpected_unseen = int(np.shape(test_Y_pred_seen)[0] * unseen_ratio) + 1 
			unseen_ind = np.argpartition(unseen_confidence, -1 * nexpected_unseen)[-1 * nexpected_unseen:]
			seen_ind = np.argpartition(unseen_confidence, -1 * nexpected_unseen)[:-1 * nexpected_unseen]
			unseen_ind_before_refine = np.argpartition(unseen_confidence_before_refine, -1 * nexpected_unseen)[-1 * nexpected_unseen:]
			seen_ind_before_refine  = np.argpartition(unseen_confidence_before_refine, -1 * nexpected_unseen)[:-1 * nexpected_unseen]
			test_Y_pred_all_new = test_Y_pred_all.copy()

			if refine:
				test_Y_pred_all[unseen_ind, :self.nseen] -= 1000000
				test_Y_pred_all[seen_ind, self.nseen:] -= 1000000
				test_Y_pred_all[:, self.nseen:] = stats.zscore(test_Y_pred_all[:, self.nseen:], axis = 0)

				test_Y_pred_all_new[unseen_ind_before_refine, :self.nseen] -= 1000000
				test_Y_pred_all_new[seen_ind_before_refine, self.nseen:] -= 1000000
				test_Y_pred_all_new[:, self.nseen:] = stats.zscore(test_Y_pred_all_new[:, self.nseen:], axis = 0)
    
			else:
				test_Y_pred_all[unseen_ind, :self.nseen] -= 1000000
				test_Y_pred_all[seen_ind, self.nseen:] -= 1000000

			return test_Y_pred_seen, logits, test_Y_pred_all, test_Y_pred_all_new, seen_ind, seen_ind_before_refine, unseen_confidence, unseen_confidence_before_refine
		
	def _batch_predict(self, X, num_batches, mapping=None):
		"""
		Predicts the type of each cell in the test data, X, in batches.
		"""

		ns = X.shape[0]
		Y = np.zeros((ns, self.nseen))
		batch_size = int(X.shape[0] / num_batches)
		for k in range(0, num_batches):
			X_array = X[k * batch_size : (k+1) * batch_size,:].todense()
			# Remaps genes to match the test set
			X_array = X_array[:,mapping]
			Y[k * batch_size : (k+1) * batch_size,:] = self.model.forward(X_array, training=False, device=self.device)
			
		# handling the end case (last batch < batch_size)
		if ns % batch_size != 0:
			X_array = X[num_batches * batch_size : ns,:].todense()
			# Remaps genes to match the test set
			X_array = X_array[:,mapping]
			Y[num_batches * batch_size : ns,:] = self.model.forward(X_array, training=False, device=self.device)
		
		return Y
