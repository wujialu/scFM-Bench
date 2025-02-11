#%%
from torch.utils.data import TensorDataset,DataLoader
import torch
import pandas as pd 
from models import model
from utils import opt_utils,args_utils 

import importlib
import glob
import os
import numpy as np 
from data_loaders import domian_loaders
import itertools
import optuna
import matplotlib.pyplot as plt

def main(args):
    # get dataloaders for training
    train_loaders, feature_dim = domian_loaders.train_domian_loaders_l(args)
    args.feature_dim = feature_dim
    print("Load data from {} domains for training".format(len(train_loaders)))
    val_loaders = domian_loaders.val_domian_loaders_l(args)

    # i/o for log output
    os.makedirs(os.path.join(args.output, args.val_domain), exist_ok=True)
    f_loss_io = open(os.path.join(args.output, args.val_domain, f'{args.logs_name}_loss.txt'),'w')
    f_val_io = open(os.path.join(args.output, args.val_domain, f'{args.logs_name}_val.txt'),'w')
    [print(_,file=f_val_io,end='\t') if idx!=len(val_loaders)-1 else print(_,file=f_val_io,end='\n') for idx, _ in enumerate(val_loaders)]
    print(f"Acc\t    AUROC\tAUPRC",file=f_val_io,end='\n') 
    algorithm = model.VREx(args)

    # train
    best_metrics = {"auroc": 0.}
    opt = opt_utils.get_optimizer(algorithm, args)
    sch = opt_utils.get_scheduler(opt, args)
    early_stop = 0
    for epoch in range(args.max_epoch):
        count = 0 
        for single_train_minibatches in zip(*train_loaders): 
        # for single_train_minibatches in itertools.zip_longest(*train_loaders, fillvalue=None):
            count+=1
            algorithm.train()
            if args.gpu_id is not None:
                algorithm.cuda()
            minibatches_device = [(data) for data in single_train_minibatches if data is not None]      
            # back-propagation
            step_vals = algorithm.update(minibatches_device, opt, sch)
            print(step_vals,file=f_loss_io)
        algorithm.eval()
        # algorithm.cpu()

        # evaluate accuracy during training
        for idx,loader_idx in enumerate(val_loaders):
            acc, auroc, auprc, y_true, y_pred = opt_utils.evaluate(algorithm,val_loaders[loader_idx])
            if idx!=len(val_loaders)-1: # report val result for each val loader
                print (f'{acc:.4f}\t{auroc:.4f}\t{auprc:.4f}',file=f_val_io,end='\t')
            else :
                print (f'{acc:.4f}\t{auroc:.4f}\t{auprc:.4f}',file=f_val_io,end='\n')
            print(f'{auroc:.4f}',end='\t')
        f_val_io.flush()    
        print(f'epoch={epoch}',end='\n')    
        print(step_vals)
        
        # save pretrained model
        if auroc >  best_metrics["auroc"]:
            best_metrics["auroc"] = auroc
            opt_utils.save_checkpoint(f'{args.val_domain}/model_best.pkl', algorithm, args)
            y_pred_best = y_pred
            best_metrics["acc"] = acc
            best_metrics["auprc"] = auprc
            early_stop = 0
        else:
            early_stop += 1
        if early_stop > args.patience:
            break
            
    # save prediction results
    print(f'Best result:\n Acc: {best_metrics["acc"]:.4f}\n AUROC: {best_metrics["auroc"]:.4f}\n AUPRC: {best_metrics["auprc"]:.4f}', file=f_val_io)
    f_val_io.flush()
    pred_df = pd.DataFrame()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred_best
    pred_df.to_csv(os.path.join(args.output, args.val_domain, "predictions.csv"), index=False)
    f_val_io.close()
    f_loss_io.close()

    return best_metrics

def define_params(trial):
    args = args_utils.get_args()
    args.lr = trial.suggest_categorical('lr', [1e-4, 5e-4, 1e-3, 5e-3])
    args.dropout = trial.suggest_float("dropout", 0.1, 0.3, step=0.05)
    args.weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 1e-5])
    args.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    return args

def objective(trial):
    args = define_params(trial)

    args.output = os.path.join(args.output, args.input_features, f"bs_{args.batch_size}_lr_{args.lr}_drop_{args.dropout:.2f}_decay_{args.weight_decay}")
    os.makedirs(args.output,exist_ok=True)

    # get model
    args_utils.set_random_seed(args.seed)
    args.HVG_list, domains = opt_utils.generate_genelist(args)

    # optimize multiple domains
    best_auroc_ls = []
    for domain in domains:
        args.val_domain = domain
        best_metrics = main(args)
        best_auroc_ls.append(best_metrics["auroc"])
    return np.mean(best_auroc_ls)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  
    study.optimize(objective, n_trials=20)  
    best_trial = study.best_trial
    print('Best trial values: {}'.format(best_trial.values))
    print("Best trial parameters: {}".format(best_trial.params))

    args = args_utils.get_args()
    df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete','duration','number'], axis=1)
    df.to_csv(f"./optuna_results/{args.input_features}/optuna_all_trials.csv", index=False)

    # fig = optuna.visualization.plot_pareto_front(study, target_names=["Tissue 1", "Tissue 2", "Tissue 3", "Tissue 4"])
    # plt.savefig(f"./optuna_results/{args.input_features}/optuna_pareto_front.png", dpi=300)

# %%
