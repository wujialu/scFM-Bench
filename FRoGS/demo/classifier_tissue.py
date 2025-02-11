import os
import random
import numpy as np
import pandas as pd
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

model_ls = ["go", "archs4", "geneformer", "langcell", "scgpt", "uce", "xtrimogene"]
dim_ls = [256, 256, 512, 512, 512, 5120, 768]

tissue_gene = {}
with open('../data/tissue_demo/tissue_specific.txt') as fr:
    for line in fr:
        items = line.split('\n')[0].split(',')
        tissue_gene[items[0]] = set(items[1:])

#Filter out genes that appear in more than one tissue
common = set([])
for tissue1 in tissue_gene:
    for tissue2 in tissue_gene:
        if tissue1 != tissue2:
            common = common.union(tissue_gene[tissue1].intersection(tissue_gene[tissue2]))

for tissue in tissue_gene:
    tissue_gene[tissue] -= common
    
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

def get_data(model, dim):
    with open(f'../data/gene_vec_{model}_{dim}.csv', mode='r') as infile:
        reader = csv.reader(infile)
        gene_emb = {rows[0]:np.array(rows[1:], dtype=np.float32) for rows in reader}

    #Assign embeddings and labels to genes
    X = []
    y = []
    for i, tissue in enumerate(tissue_gene):
        for gid in tissue_gene[tissue]:
            # emb = np.zeros(256 + dim)
            # if gid in go_emb:
            #     emb[:256]=go_emb[gid]

            # if gid in archs4_emb:
            #     emb[256:]=archs4_emb[gid]
            emb = gene_emb[gid]
            X.append(emb)
            y.append(tissue)

    X = np.array(X)
    y = np.array(y)
    
    return X, y

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features 
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        # self.y = y
        self.label2id = {label:i for i, label in enumerate(set(y))}
        self.id2label = {v:k for k, v in self.label2id.items()}
        self.y = np.array([self.label2id[y] for y in y])
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)
    
def classify(X, y, output, method, clf_type="RF", device="cuda"):
    #Generate the t-SNE plot
    model = TSNE(n_components=2, random_state=0)
    tsne_pj = model.fit_transform(X)
    plt.clf()
    plt.figure(figsize=(4,4))
    plt.axis('equal')
    for tissue in tissue_gene:
        idx = np.where(y == tissue)[0]
        plt.scatter(tsne_pj[idx,0], tsne_pj[idx,1], label=tissue, s=3)
    plt.legend()
    plt.tight_layout()
    plt.title(method)
    plt.savefig(output, bbox_inches='tight')

    #Build the random forest classifer. Using 80% genes for training and 20% genes for test
    train_index = index[:int(0.8 * len(X))]
    test_index = index[int(0.8 * len(X)):]
    
    if clf_type == "RF":
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
        clf.fit(X[train_index], y[train_index])
        y_hat = clf.predict(X[test_index])
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean")
        clf = MLP(in_features=X.shape[1], hidden_features=256, out_features=len(tissue_gene), act_layer=nn.ReLU, drop=0.3)
        clf.to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=1e-4)
        # split train/val dataset
        train_index, val_index = train_index[:int(0.8 * len(train_index))], train_index[int(0.8 * len(train_index)):]
        train_dataset = MyDataset(X[train_index], y[train_index])
        valid_dataset = MyDataset(X[val_index], y[val_index])
        test_dataset = MyDataset(X[test_index], y[test_index])
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  
        valid_loader = DataLoader(valid_dataset, batch_size=8)  
        test_loader = DataLoader(test_dataset, batch_size=8)  
        
        best_valid_loss = 1e4
        for epoch in range(50):
            clf.train()
            train_loss = []
            for batch in tqdm(train_loader):
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
                logits = clf(batch_x)
                loss = criterion(logits, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())

            clf.eval()
            valid_loss = []
            for batch in tqdm(valid_loader):
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
                logits = clf(batch_x)
                loss = criterion(logits, batch_y)
                
                valid_loss.append(loss.item())
                
            valid_loss = np.mean(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                
                y_hat = []
                for batch in tqdm(test_loader):
                    batch_x, batch_y = batch
                    logits = clf(batch_x.float().to(device))
                    y_hat.append(F.softmax(logits, dim=1).detach().cpu().numpy())
                    
                y_hat = np.concatenate(y_hat).argmax(axis=1)
                    
            print(f"Epoch: {epoch}, Train loss: {np.mean(train_loss)}, Best valid loss: {best_valid_loss}")
        y_hat = np.array([train_dataset.id2label[y] for y in y_hat])
        
    print('Groundtruth', y[test_index])
    print('Predictions', y_hat)
    # acc = (y_hat==y[test_index]).sum()/len(y_hat)
    acc = accuracy_score(y[test_index], y_hat)
    macro_f1 = f1_score(y[test_index], y_hat, average="macro")
    print('Accuracy on test:', acc)
    return acc, macro_f1

#! get overlapping genes
for model, dim in zip(model_ls, dim_ls):
    with open(f'../data/gene_vec_{model}_{dim}.csv', mode='r') as infile:
        reader = csv.reader(infile)
        gene_emb = {rows[0]:np.array(rows[1:], dtype=np.float32) for rows in reader}

    #Filter out genes that do not appear in our embedding
    print('Number of genes:')
    
    for tissue in tissue_gene:
        # tissue_gene[tissue] = tissue_gene[tissue].intersection(set(go_emb.keys()).union(archs4_emb.keys()))
        tissue_gene[tissue] = tissue_gene[tissue].intersection(gene_emb.keys())
        print(tissue, len(tissue_gene[tissue]))
        

#! five runs with different random seeds
num_genes = 0
for tissue in tissue_gene:
    num_genes += len(tissue_gene[tissue])
    
results = []
device = "cuda:1"
clf_type = "MLP"
for iter in range(5):
    index = np.arange(num_genes)
    seed_everything(iter)
    np.random.shuffle(index)
    
    for model, dim in zip(model_ls, dim_ls):
        X, y = get_data(model, dim)
        acc, macro_f1 = classify(X, y, f"tsne_{model}.png", model, clf_type, device)
        results.append({"Model": model, "Accuracy": acc, "Macro F1": macro_f1})

    model = "One-hot"
    X = np.diag(np.ones(X.shape[0]))
    acc, macro_f1 = classify(X, y, "tsne_onehot.png", model, clf_type, device)
    results.append({"Model": model, "Accuracy": acc, "Macro F1": macro_f1})

df = pd.DataFrame(results)
df.to_csv(f"results/{clf_type}_tissue_specific_gene_classificaton.csv", index=False)
