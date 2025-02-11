import os
import pandas as pd
import json
from config import optuna_result_dir

model_ls = ["scVI", "Geneformer", "LangCell", "scGPT", "xTrimoGene", "UCE"]
dnames = ["HLCA_core", "Tabula_Sapiens_all"]
results = []
for dname in dnames:
    for model in model_ls:
        result_dir = optuna_result_dir + f"/{model}/onclass_dot_product/{dname}"
        for subdir in os.listdir(result_dir):
            # print(subdir)
            lr = subdir.split("_")[1]
            l2 = subdir.split("_")[3]
            metrics = pd.read_csv(os.path.join(result_dir, subdir, "0/0/metrics.csv"))
            value = metrics.loc[0, "Accuracy@1"]
            results.append(
                {
                    "dname": dname,
                    "model": model if model != "xTrimoGene" else "scFoundation",
                    "lr": lr,
                    "l2": l2,
                    "value": value
                }
            )
result_df = pd.DataFrame.from_dict(results)

model_to_params = {}
for dname in result_df["dname"].unique():
    subset = result_df[result_df["dname"]==dname]
    max_value_df = subset.loc[subset.groupby("model")["value"].idxmax()].reset_index(drop=True)
    model_to_params[dname] = max_value_df.drop("dname",axis=1).set_index("model").to_dict(orient="index")

with open(optuna_result_dir + "/model_to_params.json", "w") as f:
    json.dump(model_to_params, f, indent=4,)