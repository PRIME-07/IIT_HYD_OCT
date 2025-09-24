from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd

# --------------------
# 1. Load the 30D UMAP embeddings
# --------------------
csv = "trained_embeddings/org_img_cnn/image_embeddings_org_img.csv_umap_2d.csv"
df_30d = pd.read_csv(csv)

names = df_30d["name"]
labels = df_30d["label"]
features = df_30d.drop(columns=["name", "label"]).values

# --------------------
# 2. K-Means clustering
# --------------------
num_clusters = 2  # binary classification
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features)

# --------------------
# 3. Evaluation
# --------------------
ari = adjusted_rand_score(labels, cluster_labels)
nmi = normalized_mutual_info_score(labels, cluster_labels)

print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# --------------------
# 4. Save clustering results
# --------------------
df_30d["cluster"] = cluster_labels
df_30d.to_csv(f"{csv[:-4]}_clusters.csv", index=False)
print("✅ Saved clustering results to embeddings_umap_30d_clusters.csv")
