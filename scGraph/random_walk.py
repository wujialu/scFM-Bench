import torch
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import collections

def renorm(X):
	Y = X.copy()
	Y = Y.astype(float)
	ngene,nsample = Y.shape
	s = np.sum(Y, axis=0)
	#print s.shape()
	for i in range(nsample):
		if s[i]==0:
			s[i] = 1
			if i < ngene:
				Y[i,i] = 1
			else:
				for j in range(ngene):
					Y[j,i] = 1. / ngene
		Y[:,i] = Y[:,i]/s[i]
	return Y

def RandomWalkRestart(A, rst_prob, delta = 1e-4, reset=None, max_iter=50,use_torch=False,return_torch=False):
	if use_torch:
		device = torch.device("cuda:0")
	nnode = A.shape[0]
	#print nnode
	if reset is None:
		reset = np.eye(nnode)
	nsample,nnode = reset.shape
	#print nsample,nnode
	P = renorm(A)
	P = P.T
	norm_reset = renorm(reset.T)
	norm_reset = norm_reset.T
	if use_torch:
		norm_reset = torch.from_numpy(norm_reset).float().to(device)
		P = torch.from_numpy(P).float().to(device)
	Q = norm_reset

	for i in range(1,max_iter):
		#Q = gnp.garray(Q)
		#P = gnp.garray(P)
		if use_torch:
			Q_new = rst_prob*norm_reset + (1-rst_prob) * torch.mm(Q, P)#.as_numpy_array()
			delta = torch.norm(Q-Q_new, 2)
		else:
			Q_new = rst_prob*norm_reset + (1-rst_prob) * np.dot(Q, P)#.as_numpy_array()
			delta = np.linalg.norm(Q-Q_new, 'fro')
		Q = Q_new
		#print (i,Q)
		sys.stdout.flush()
		if delta < 1e-4:
			break
	if use_torch and not return_torch:
		Q = Q.cpu().numpy()
	return Q


def read_cell_type_nlp_network(nlp_emb_file, cell_type_network_file):
	cell_ontology_ids = set()
	fin = open(cell_type_network_file)
	co2co_graph = {}
	for line in fin:
		w = line.strip().split('\t')
		if w[0] not in co2co_graph:
			co2co_graph[w[0]] = set()
		co2co_graph[w[0]].add(w[1])
		cell_ontology_ids.add(w[0])
		cell_ontology_ids.add(w[1])
	fin.close()

	i2l = {}
	l2i = {}
	for l in cell_ontology_ids:
		nl = len(i2l)
		l2i[l] = nl
		i2l[nl] = l
		
	fin = open(nlp_emb_file)
	co2vec_nlp = {}
	for line in fin:
		w = line.strip().split('\t')
		vec = []
		for i in range(1,len(w)):
			vec.append(float(w[i]))
		co2vec_nlp[w[0]] = np.array(vec)
	fin.close()
 
	co2co_nlp = collections.defaultdict(dict)
	nco = len(cell_ontology_ids)
	ontology_mat = np.zeros((nco,nco))
	for id1 in co2co_graph:
		for id2 in co2co_graph[id1]:
			sc = 1 - spatial.distance.cosine(co2vec_nlp[id1], co2vec_nlp[id2])
			co2co_nlp[id1][id2] = sc
			ontology_mat[l2i[id1], l2i[id2]] = 1
	return co2co_graph, co2co_nlp, co2vec_nlp, ontology_mat, l2i, i2l

def emb_ontology(i2l, co2co_nlp, ontology_mat, rst = 0.7):
	nco = len(i2l)
	network = np.zeros((nco, nco))
	for i in range(nco):
		c1 = i2l[i]
		for j in range(nco):
			# used the text description-based similarity to augment edges on the Cell Ontology graph.
			if ontology_mat[i,j] == 1:
				network[i,j] = co2co_nlp[c1][i2l[j]]
				network[j,i] = co2co_nlp[c1][i2l[j]]
	onto_net_rwr = RandomWalkRestart(network, rst)
	print(f"The shape of the RWR based cell ontology net: {onto_net_rwr.shape}")
	return onto_net_rwr