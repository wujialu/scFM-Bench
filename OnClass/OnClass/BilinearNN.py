import numpy as np
import os

#Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from scipy.special import softmax
import time
import math
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from .dag_transformer import GraphTransformer

class BilinearNN(nn.Module):
	"""
	PyTorch implementation of the Bilinear Neural Network described in the OnClass paper
	"""
	def __init__(self, Y_emb, nseen, ngene, use_pretrain = None, nhidden=[1000], lr=1e-4, l2=0.005, memory_saving_mode = False,
                 dot_product = True, onto_graph = None, class_weights = None):
		super(BilinearNN, self).__init__()
		self.mode = memory_saving_mode
		self.ncls, self.ndim = np.shape(Y_emb)
		self.l2 = l2
		self.nseen = nseen
		self.ngene = ngene
		self.use_pretrain = use_pretrain
		self.seen_Y_emb = Y_emb[:nseen,:] # seen labels first
		self.nY = np.shape(self.seen_Y_emb)[0]
		self.seen_Y_emb = np.array(self.seen_Y_emb, dtype=np.float32)
		self.Y_emb = np.array(Y_emb, dtype=np.float32)

		self.nhidden = [self.ngene]
		self.nhidden.extend(nhidden)
		self.dot_product = dot_product

		self.onto_graph = onto_graph
		if self.onto_graph is not None:
			self.ndim = 128
			self.label_encoder = GraphTransformer(in_size=256, d_model=self.ndim, dim_feedforward=self.ndim*4, use_edge_attr=False, in_embed=False,
												  num_layers=2, num_heads=4, dropout=0.2, batch_norm=True, SAT=False)
			
		if self.dot_product:
			self.nhidden.append(self.ndim) # number of classes(one-hot) + class emb_dim
		else:
			self.nhidden.append(self.nseen)
  
		torch.manual_seed(3)
		self.__build()
		if class_weights is not None:
			self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
		else:
			self.loss_fn = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.l2)

	def forward(self, X, training=True, device="cuda"):
		"""
		Runs a forward iteration on the gene expression matrix, X
		"""
		X = np.array(X, dtype=np.float32)
		X = torch.from_numpy(X).to(device)

		if self.onto_graph is not None:
			seen_embedding = self.label_encoder(self.onto_graph.to(device))[:self.nseen, :].t()
		else:
			seen_embedding = torch.from_numpy(self.seen_Y_emb).t().to(device)

		for i in range(1,self.nlayer):
			X = self.W[str(i)](X)
			if i != self.nlayer - 1:
				X = F.relu(X)
				X = F.dropout(X, training=training)
    
		if self.dot_product:
			X = torch.matmul(X, seen_embedding)  # [n_cells, nseen_class]
		return X
	
	def __build(self, stddev=0.0001, seed=3):
		"""
		Builds the model's linear layers and initializes them.
		"""
		torch.manual_seed(3)
		W = {}
		self.nlayer = len(self.nhidden)
		for i in range(1,self.nlayer):
			W[str(i)] = nn.Linear(self.nhidden[i-1], self.nhidden[i]).requires_grad_()
			nn.init.xavier_uniform_((W[str(i)]).weight)
			nn.init.zeros_((W[str(i)]).bias)
		self.W = nn.ModuleDict(W)
		

	def read_training_data(self, train_X, train_Y, use_valid = False, valid_X = None, valid_Y = None, test_X = None, test_Y = None):
		"""
		Stores the training and optionally the test data in the model for ease of use. Also, the
		use_valid parameter allows for quick validation testing on the training data (where the
		validation set is 10% of the training data) 
		"""
		self.use_valid = use_valid
		if self.use_valid:
			if valid_X is None:
				np.random.seed(1)
				nx = np.shape(train_X)[0]
				ntrain = int(nx*0.9)
				permutation = list(np.random.permutation(nx))
				train_ind = permutation[:ntrain]
				valid_ind = permutation[ntrain:]
				self.train_X = train_X[train_ind, :]
				self.valid_X = train_X[valid_ind, :]
				self.train_Y = train_Y[train_ind]
				self.valid_Y = train_Y[valid_ind]
			else:
				self.train_X = train_X
				self.valid_X = valid_X
				self.train_Y = train_Y
				self.valid_Y = valid_Y
			self.valid_Y = self.one_hot_matrix(self.valid_Y, self.ncls)
		else:
			self.train_X = train_X
			self.train_Y = train_Y
		self.train_Y = self.one_hot_matrix(self.train_Y, self.nseen)
		self.nX = np.shape(self.train_X)[0]

		if test_X is not None:
			self.test_X = test_X
			self.test_Y = self.one_hot_matrix(test_Y, self.ncls)
		else:
			self.test_X = None
			self.test_Y = None
		if not self.mode:
			self.train_X = np.array(self.train_X, dtype=np.float32)
			self.train_Y = np.array(self.train_Y, dtype=np.float32) 


	def one_hot_matrix(self, labels, C):
		"""
		Returns the one-hot matrix representation of labels.
		"""
		return F.one_hot(torch.Tensor(labels).to(torch.int64), num_classes=C)

	def train_epoch(self, train_loader, device="cuda"):
		epoch_cost = []
		for minibatch in tqdm(train_loader):
			(minibatch_X, minibatch_Y) = minibatch
			if self.mode:
				minibatch_X = minibatch_X.todense()
			pred = self.forward(minibatch_X, device=device)

			# Sometimes our minibatch labels are tensors, and other times they are numpy
			# arrays. This allows the model to handle either
			if isinstance(minibatch_Y, np.ndarray):
				minibatch_Y = torch.from_numpy(minibatch_Y)
			
			# labels = torch.argmax(minibatch_Y, axis=1).to(device)
			loss = self.loss_fn(pred, minibatch_Y.to(device))

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			epoch_cost.append(loss.item())
		return np.mean(epoch_cost)
    
	def eval_epoch(self, valid_loader, device="cuda"):
		epoch_val_loss = []
		y_pred_all = []
		y_true_all = []
		for minibatch in tqdm(valid_loader):
			(minibatch_X, minibatch_Y) = minibatch
			if self.mode:
				minibatch_X = minibatch_X.todense()
			with torch.no_grad():
				pred = self.forward(minibatch_X, device=device)
			loss = self.loss_fn(pred, minibatch_Y.to(device))
			epoch_val_loss.append(loss.item())

		return np.mean(epoch_val_loss)

	def test_epoch(self, test_loader, device="cuda"):
		y_true = []
		y_pred = []
		for minibatch in tqdm(test_loader):
			(minibatch_X, minibatch_Y) = minibatch
			if self.mode:
				minibatch_X = minibatch_X.todense()
			with torch.no_grad():
				pred = self.forward(minibatch_X, device=device)
		# 	y_true.append(minibatch_Y.numpy())
			y_pred.append(pred.detach().cpu().numpy())
		# y_true = np.concatenate(y_true, axis=0)
		y_pred_logits = np.concatenate(y_pred, axis=0)
		y_pred = softmax(y_pred_logits, axis=1)
		# accuracy = np.mean(np.argmax(y_pred,axis=1) == np.argmax(y_true,axis=1))
		return y_pred, y_pred_logits

	def optimize(self, train_loader, max_iter = 15, mini_batch_size = 128, keep_prob = 1., device=None):
		"""
		Uses optimizer (default is PyTorch's cross entropy) to train linear layers
		"""
		self.keep_prob = keep_prob
		# seed = 3
		for epoch in range(max_iter):
			# seed = seed + 1
			# minibatches = self.random_mini_batches(mini_batch_size=mini_batch_size, seed=seed)
			# num_minibatches = int(self.nX / mini_batch_size)
			epoch_cost = self.train_epoch(train_loader, device=device)
			print("Epoch", epoch, "with loss", np.mean(epoch_cost))

	def random_mini_batches(self, mini_batch_size=32, seed=1):
		"""
		input -- X (training set), Y (true labels)
		output - minibatches
		"""
		ns = self.train_X.shape[0]
		mini_batches = []
		np.random.seed(seed)
		# shuffle (X, Y)
		permutation = list(np.random.permutation(ns))
		shuffled_X = self.train_X[permutation, :]
		shuffled_Y = self.train_Y[permutation, :]
		# partition (shuffled_X, shuffled_Y), minus the end case.
		num_complete_minibatches = int(math.floor(ns/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
		for k in range(0, num_complete_minibatches):
			mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
			mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
    	# handling the end case (last mini-batch < mini_batch_size)
		if ns % mini_batch_size != 0:
			mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : ns,:]
			mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : ns,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
		return mini_batches

	def predict_prob(self, X, Y):
		"""
		Prints out information such as auroc, and accuracy. Useful for finding getting info on 
		model at each epoch of training.
		"""
		label = Y
		p = self.forward(X)
		p = softmax(p, axis=1)
		accuracy = np.mean(np.argmax(p,axis=1) == np.argmax(label,axis=1))
		[nsample, nclass] = np.shape(Y)
		class_auc_macro = np.full(nclass, np.nan)
		class_auprc_macro =  np.full(nclass, np.nan)
		for i in range(self.nseen):
			if len(np.unique(Y[:,i]))==2:
				class_auc_macro[i] = roc_auc_score(Y[:,i], p[:,i])
				class_auprc_macro[i] = average_precision_score(Y[:,i], p[:,i])
		seen_auroc = np.nanmedian(class_auc_macro)
		seen_auprc = np.nanmedian(class_auprc_macro)
		class_auc_macro = np.full(nclass, np.nan)
		class_auprc_macro =  np.full(nclass, np.nan)
		for i in range(self.nseen, self.ncls):
			if i >= np.shape(Y)[1] or i  >= np.shape(p)[1]:
				break
			if len(np.unique(Y[:,i]))==2:
				class_auc_macro[i] = roc_auc_score(Y[:,i], p[:,i])
				class_auprc_macro[i] = average_precision_score(Y[:,i], p[:,i])
		unseen_auroc = np.nanmedian(class_auc_macro)
		unseen_auprc = np.nanmedian(class_auprc_macro)

		return accuracy, p, seen_auroc, seen_auprc, unseen_auroc, unseen_auprc
