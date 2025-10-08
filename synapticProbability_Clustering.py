import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway, chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
# animal1 = pd.read_csv("E:/Ryan Analysis/P8RA3M4_unrefined_scaled_independently_Ib.csv")
animal2 = pd.read_csv("E:/Ryan Analysis/P9RA3M4_Pooled_refined_scaled.csv")

# Label each animal
#animal1["Animal"] = "Animal1"
#animal2["Animal"] = "Animal2"

# Combine
df_all = pd.concat([animal2], ignore_index=True)

# Predictors
predictors = ["Brp", "Unc13A", "RBP"]

# Normalize predictors
scaler = StandardScaler()
df_all[predictors] = scaler.fit_transform(df_all[predictors])

# t-SNE for 2D visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(df_all[predictors])
df_all["TSNE_1"] = tsne_result[:, 0]
df_all["TSNE_2"] = tsne_result[:, 1]

# KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
df_all["KMeans_Cluster"] = kmeans.fit_predict(df_all[predictors])

# Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=10)
df_all["Agglomerative_Cluster"] = agg.fit_predict(df_all[predictors])

# Plot t-SNE clusters
for cluster_col in ["KMeans_Cluster", "Agglomerative_Cluster"]:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_all, x="TSNE_1", y="TSNE_2", hue=cluster_col, palette="Set1", s=70)
    plt.title(f"t-SNE colored by {cluster_col}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title=cluster_col)
    plt.tight_layout()
    plt.show()

# Fs_All by cluster (boxplots)
plt.figure(figsize=(10, 4))
for i, cluster_col in enumerate(["KMeans_Cluster", "Agglomerative_Cluster"], 1):
    plt.subplot(1, 2, i)
    sns.boxplot(data=df_all, x=cluster_col, y="Fs_All", palette="Set2")
    plt.title(f"Fs_All by {cluster_col}")
    plt.xlabel("Cluster")
    plt.ylabel("Fs_All")
plt.tight_layout()
plt.show()

# ANOVA test
def anova_test(label_col):
    groups = [group["Fs_All"].values for _, group in df_all.groupby(label_col)]
    f_stat, p_val = f_oneway(*groups)
    print(f"{label_col} ANOVA: F = {f_stat:.4f}, p = {p_val:.4e}")

anova_test("KMeans_Cluster")
anova_test("Agglomerative_Cluster")

# Cluster composition by animal (bar plots)
for cluster_col in ["KMeans_Cluster", "Agglomerative_Cluster"]:
    ct = pd.crosstab(df_all[cluster_col], df_all["Num"], normalize='index')
    ct.plot(kind='bar', stacked=True, colormap='Set2')
    plt.title(f"{cluster_col} composition by Animal")
    plt.xlabel("Cluster")
    plt.ylabel("Proportion from each Animal")
    plt.legend(title="Animal")
    plt.tight_layout()
    plt.show()

# Chi-squared test
def chi2_test(cluster_col):
    contingency = pd.crosstab(df_all[cluster_col], df_all["Num"])
    chi2, p, dof, _ = chi2_contingency(contingency)
    print(f"{cluster_col} vs Animal: chi² = {chi2:.2f}, p = {p:.4e}")

chi2_test("KMeans_Cluster")
chi2_test("Agglomerative_Cluster")
