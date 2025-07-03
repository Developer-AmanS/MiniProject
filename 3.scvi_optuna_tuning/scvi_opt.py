#!/usr/bin/env python
# scVI Optimization Script with NaN error fix

import scvi
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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


import os
import random
import numpy as np
import torch

# 设置固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


# 设置PyTorch稳定性选项
torch.set_float32_matmul_precision('medium')  # 使用更稳定的计算精度
torch.backends.cudnn.deterministic = True  # 确保结果可复现

# 设置警告过滤器
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

print("Loading datasets...")
arg = sc.read("arg_qc_data.h5ad")
pb_batch1 = sc.read("pb_batch1_qc_data.h5ad")
pb_batch2 = sc.read("pb_batch2_qc_data.h5ad")

print("Combining datasets...")
# 合并数据集为一个AnnData对象
adata_combined = arg.concatenate(pb_batch1, pb_batch2, batch_key="batch")

# 确保"Unknown"是一个有效的类别
adata_combined.obs["celltype"] = adata_combined.obs["celltype"].cat.add_categories("Unknown")
# 安全地将NaN替换为"Unknown"
adata_combined.obs["celltype"] = adata_combined.obs["celltype"].fillna("Unknown")

print("Computing highly variable genes...")
# 存储每个批次的高可变基因
batch_hvgs = []

for batch in adata_combined.obs["batch"].unique():
    adata_batch = adata_combined[adata_combined.obs["batch"] == batch].copy()
    sc.pp.highly_variable_genes(adata_batch, flavor="seurat_v3", n_top_genes=2000)
    batch_hvgs.append(set(adata_batch.var_names[adata_batch.var["highly_variable"]]))

# 所有数据集高可变基因的并集
combined_hvgs = set.union(*batch_hvgs)

# 仅保留高可变基因
adata_combined = adata_combined[:, list(combined_hvgs)].copy()

# 检查数据是否有NaN值并处理
print("Checking for NaN values in data...")
if scipy.sparse.issparse(adata_combined.X):
    if np.isnan(adata_combined.X.data).any():
        print("Found NaNs in sparse data matrix, replacing with zeros")
        adata_combined.X.data[np.isnan(adata_combined.X.data)] = 0
else:
    if np.isnan(adata_combined.X).any():
        print("Found NaNs in data matrix, replacing with zeros")
        adata_combined.X[np.isnan(adata_combined.X)] = 0

print("Saving combined dataset...")
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
adata_combined.write(f"{output_dir}/adata_combined.h5ad")

# 设置AnnData (一次性)
print("Setting up scVI...")
scvi.model.SCVI.setup_anndata(adata_combined, batch_key="batch")


def objective(trial):
    # 保持原始超参数搜索空间不变
    n_latent = trial.suggest_categorical("n_latent", [5, 10, 20, 30, 40, 50])
    n_hidden = trial.suggest_categorical("n_hidden", [64, 128, 256, 512])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05)

    # 这些是为每组试验固定的
    dispersion = trial.suggest_categorical("dispersion", [dispersion_value])
    likelihood = trial.suggest_categorical("gene_likelihood", [likelihood_value])

    latent_distribution = trial.suggest_categorical("latent_distribution", ["normal", "ln"])

    # 训练参数 - 保持原始范围，但添加稳定性改进
    max_epochs = trial.suggest_int("max_epochs", 100, 500, step=50)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # 使用异常处理来增强稳定性
    try:
        model = SCVI(
            adata_combined,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=likelihood,
            latent_distribution=latent_distribution
        )

        # 添加额外的训练参数以解决NaN问题
        model.train(
            max_epochs=max_epochs,
            batch_size=batch_size,
            plan_kwargs={
                "lr": learning_rate,
                "weight_decay": 1e-6,  # 添加轻微的权重衰减以提高稳定性
                "eps": 1e-8  # 为优化器增加小的epsilon值
            },
            early_stopping=True,
            check_val_every_n_epoch=5  # 适度减少验证检查频率
        )

        # 获取潜在表示
        latent = model.get_latent_representation()

        # 检查NaN值
        if np.isnan(latent).any():
            print("Warning: Latent representation contains NaN values, returning bad score")
            return -999

        try:
            # 细胞类型轮廓得分 (越高越好)
            sil_celltype = silhouette_score(latent, adata_combined.obs["celltype"])

            # 批次轮廓得分 (越低越好 → 我们将其减去)
            sil_batch = silhouette_score(latent, adata_combined.obs["batch"])

            score = sil_celltype - sil_batch

        except Exception as e:
            print(f"Error calculating score: {e}")
            score = -999  # 失败情况下的很差分数

    except Exception as e:
        print(f"Error in model training: {e}")
        score = -999  # 出现任何错误时的很差分数

    return score


# 定义要测试的特定组合
combination_pairs = [
    ("gene-batch", "zinb"),
    ("gene-cell", "nb")
]

# 创建列表用于存储所有试验结果
all_trials = []


# 用于记录试验结果的函数
def logging_callback(study, trial):
    entry = {
        "trial": trial.number,
        "score": trial.value,
        "dispersion": dispersion_value,
        "gene_likelihood": likelihood_value,
        **{k: v for k, v in trial.params.items() if k not in ["dispersion", "gene_likelihood"]}
    }
    all_trials.append(entry)

    # 每5次试验保存一次所有数据，以防程序中断
    if trial.number % 5 == 0 and trial.number > 0:
        temp_df = pd.DataFrame(all_trials)
        temp_df.to_csv(f"{output_dir}/optuna_scvi_trials_checkpoint.csv", index=False)


# 为每个组合运行20次试验
for dispersion_value, likelihood_value in combination_pairs:
    print(f"\nRunning trials for dispersion={dispersion_value}, likelihood={likelihood_value}")

    # 为此组合创建一个研究
    study_name = f"scvi_opt_{dispersion_value}_{likelihood_value}"
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize",
                                study_name=study_name,
                                sampler=sampler)

    # 运行20次试验，允许出错并继续
    try:
        study.optimize(objective, n_trials=20, callbacks=[logging_callback],
                       catch=(ValueError, RuntimeError))  # 捕获常见错误并继续
    except Exception as e:
        print(f"Optimization error: {e}")

    # 保存此组合的研究
    joblib.dump(study, f"{output_dir}/optuna_study_{dispersion_value}_{likelihood_value}.pkl")

    # 打印此组合的最佳参数
    if study.best_trial:
        print(f"Best score for {dispersion_value}/{likelihood_value}: {study.best_value}")
        print(f"Best parameters: {study.best_params}")
    else:
        print(f"No successful trials for {dispersion_value}/{likelihood_value}")

# 将所有收集的试验转换为DataFrame并保存
df_trials = pd.DataFrame(all_trials)
df_trials.to_csv(f"{output_dir}/optuna_scvi_all_trials.csv", index=False)

# 如果有成功的试验，获取前20个结果
if len(df_trials) > 0:
    df_trials.sort_values("score", ascending=False, inplace=True)
    df_trials.reset_index(drop=True, inplace=True)

    # 打印前10个总体结果
    print("\nTop 10 combinations across all trials:")
    print(df_trials.head(10))

    # 保存前20个结果
    df_trials.head(20).to_csv(f"{output_dir}/optuna_scvi_top_trials.csv", index=False)

    # 为最佳组合生成可视化
    if len(df_trials) > 0:
        top_dispersion = df_trials.iloc[0]["dispersion"]
        top_likelihood = df_trials.iloc[0]["gene_likelihood"]

        print(f"\nCreating visualizations for top combination: {top_dispersion}/{top_likelihood}")

        # 加载最佳研究
        best_study_path = f"{output_dir}/optuna_study_{top_dispersion}_{top_likelihood}.pkl"
        if os.path.exists(best_study_path):
            best_study = joblib.load(best_study_path)

            # 生成可视化图表
            try:
                # 创建图表目录
                plots_dir = f"{output_dir}/plots"
                os.makedirs(plots_dir, exist_ok=True)

                # 优化历史
                fig = optuna.visualization.plot_optimization_history(best_study)
                fig.write_image(f"{plots_dir}/optimization_history_{top_dispersion}_{top_likelihood}.png")

                # 参数重要性
                fig = optuna.visualization.plot_param_importances(best_study)
                fig.write_image(f"{plots_dir}/param_importance_{top_dispersion}_{top_likelihood}.png")

                # 平行坐标
                fig = optuna.visualization.plot_parallel_coordinate(best_study)
                fig.write_image(f"{plots_dir}/parallel_coordinate_{top_dispersion}_{top_likelihood}.png")

                print("Visualizations saved to plots directory")
            except Exception as e:
                print(f"Error generating visualizations: {e}")
        else:
            print(f"Best study file not found: {best_study_path}")
else:
    print("No successful trials found")

print("\nOptimization complete!")