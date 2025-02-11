import subprocess
import optuna
import re
import os

def run_script(h_dim, z_dim, ep, la1, mb, drug, software):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    command = [
        "python", "utils/SCAD_train_binarized_5folds-pub.py",
        "-e", "FX", "-d", drug, "-g", "_norm", "-s", "42",
        "-h_dim", str(h_dim), "-z_dim", str(z_dim), "-ep", str(ep),
        "-la1", str(la1), "-mbS", str(mb), "-mbT", str(mb), "-emb", "1", "--software", software
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

def parse_output(output):
    match = re.search(r"Average Test AUC = (\d+\.\d+)", output)
    if match:
        return float(match.group(1))
    return 0.0

def objective(trial, drug):
    h_dim = trial.suggest_int("h_dim", 512, 2049)
    z_dim = trial.suggest_int("z_dim", 128, 513)
    ep = trial.suggest_int("ep", 20, 101)
    la1 = trial.suggest_float("la1", 0.0, 5.0)
    mb = trial.suggest_categorical("mb", [8, 16, 32])

    output = run_script(h_dim, z_dim, ep, la1, mb, drug)
    score = parse_output(output)

    return score

drugs = ["NVP-TAE684", "Sorafenib", "Etoposide", "PLX4720_451Lu"]
softwares = ["langcell", "geneformer", "scgpt", "uce"]
results = {}

for software in softwares:
    for drug in drugs:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, drug), n_trials=50)
        results[drug] = {
            "best_params": study.best_params,
            "best_score": study.best_value
        }

    os.makedirs(f"./output/{software}", exist_ok=True)
    with open(f"best_scad_params_{software}.txt", "w") as f:
        for drug, result in results.items():
            f.write(f"Drug: {drug}\n")
            f.write(f"Best Parameters: {result['best_params']}\n")
            f.write(f"Best Score: {result['best_score']}\n\n")