#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install scanpy matplotlib numpy pandas pybiomart scikit-misc scvi-tools optuna')


# In[9]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[10]:


import scvi
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from pybiomart import Server
import scipy.sparse
import warnings
import optuna
from sklearn.metrics import silhouette_score
from scvi.model import SCVI
import joblib


# In[11]:


arg = sc.read("/kaggle/input/data-science-mini-project-qced-data/arg_qc_data.h5ad")
pb_batch1 = sc.read("/kaggle/input/data-science-mini-project-qced-data/pb_batch1_qc_data.h5ad")
pb_batch2 = sc.read("/kaggle/input/data-science-mini-project-qced-data/pb_batch2_qc_data.h5ad")


# ## Step 1: Train scvi on all datasets

# In[12]:


# Combine the datasets into one AnnData object
adata_combined = arg.concatenate(pb_batch1, pb_batch2, batch_key="batch")

# Ensure "Unknown" is a valid category
adata_combined.obs["celltype"] = adata_combined.obs["celltype"].cat.add_categories("Unknown")
# Replace NaNs with "Unknown" safely
adata_combined.obs["celltype"] = adata_combined.obs["celltype"].fillna("Unknown")


# In[13]:


# Store HVGs for each batch
batch_hvgs = []

for batch in adata_combined.obs["batch"].unique():
    adata_batch = adata_combined[adata_combined.obs["batch"] == batch].copy()
    sc.pp.highly_variable_genes(adata_batch, flavor="seurat_v3", n_top_genes=2000)
    batch_hvgs.append(set(adata_batch.var_names[adata_batch.var["highly_variable"]]))

# Union of HVGs from all datasets
combined_hvgs = set.union(*batch_hvgs)

adata_combined = adata_combined[:, list(combined_hvgs)].copy()


# In[14]:


adata_combined.write("/kaggle/working/adata_combined.h5ad")


# In[15]:


# Setup anndata (once)
scvi.model.SCVI.setup_anndata(adata_combined, batch_key="batch")

def objective(trial):
    # Architecture & model parameters
    n_latent = trial.suggest_categorical("n_latent", [5, 10, 20, 30, 40, 50])
    n_hidden = trial.suggest_categorical("n_hidden", [64, 128, 256, 512])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3]) # Choose from a fixed list of options
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05) # Choose from a fixed list of options
    latent_distribution = trial.suggest_categorical("latent_distribution", ["normal", "ln"])

    dispersion = "gene-batch"
    likelihood = "nb"


    # Training parameters
    max_epochs = trial.suggest_int("max_epochs", 100, 500, step=50)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Initialize SCVI
    model = SCVI(
        adata_combined,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        dispersion=dispersion,
        gene_likelihood=likelihood,
        latent_distribution = latent_distribution
    )

    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        plan_kwargs={"lr": learning_rate},
        early_stopping=True
    )

    # Latent representation
    latent = model.get_latent_representation()

    try:
        # Cell type silhouette (higher is better)
        sil_celltype = silhouette_score(latent, adata_combined.obs["celltype"])

        # Batch silhouette (lower is better → we subtract it)
        sil_batch = silhouette_score(latent, adata_combined.obs["batch"])

        score = sil_celltype - sil_batch

    except Exception:
        score = -999  # Very bad score in case of failure

    return score


# In[16]:


all_trials = []

def logging_callback(study, trial):
    entry = {
        "trial": trial.number,
        "score": trial.value,
        **trial.params
    }
    all_trials.append(entry)


# In[17]:


sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", study_name="scvi_dual_optimization",sampler=sampler)
study.optimize(objective, n_trials=20, callbacks=[logging_callback])


# In[18]:


# Convert the collected trials into a DataFrame and save
df_trials = pd.DataFrame(all_trials)
df_trials.to_csv("/kaggle/working/optuna_scvi_trials.csv", index=False)


# In[19]:


df_trials.sort_values("score", ascending=False, inplace=True)
df_trials.reset_index(drop=True, inplace=True)

# Preview the top results
df_trials.head()


# In[20]:


#print the best parameters and metric values using optuna
study.best_params


# In[21]:


study.best_value


# In[22]:


#visualize plot optimisation using optuna
optuna.visualization.plot_optimization_history(study)


# In[23]:


optuna.visualization.plot_param_importances(study)


# In[24]:


# Save the full study
joblib.dump(study, "/kaggle/working/optuna_study_scvi.pkl")


# In[ ]:


kaggle/working/adata_combined.h5ad()

