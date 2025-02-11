import os
repo_dir = os.path.dirname(__file__)
data_dir = os.path.join(repo_dir, "../data/")
print(repo_dir, data_dir)

# scrna_data_dir = data_dir + 'OnClass_data_public/scRNA_data/'
scrna_data_dir = data_dir
ontology_data_dir = data_dir + 'OnClass_data_public/Ontology_data/'
intermediate_dir = data_dir + 'OnClass_data_public/Intermediate_files/'

optuna_result_dir = repo_dir + '/results_gridsearch'
result_dir = repo_dir + '/results_best_params'
figure_dir = repo_dir + '/result/SingleCell/OnClass/Reproduce/Figure/'
model_dir = data_dir + 'OnClass_data_public/Pretrained_model/'


MEMORY_SAVING_MODE = True