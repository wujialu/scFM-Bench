# OnClass
The source code is from https://github.com/wangshenguiuc/OnClass.


## Optimize hyperparameters
### Run the optuna-based optimization script
```
# cd to the OnClass folder
bash scripts/optimize_params.sh
```
Please change the `dataset` variable in `optimize_params.sh`.

### Gather results and save the best params 
```
python get_best_params.py
```
The best params will be saved as `model_to_params.json` in `optuna_result_dir`.

## Intra-dataset and cross-dataset validation 
```
bash scripts/run_5cv_hlca.sh
bash scripts/run_5cv_tabula.sh
```

## Identifying novel cell types under different unseen ratios
```
bash scripts/run_5cv_hlca_unseen.sh
bash scripts/run_5cv_tabula_unseen.sh
```

## Model ensemble
```
bash scripts/run_model_ensemble.sh
```
