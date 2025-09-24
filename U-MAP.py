# pip install umap-learn matplotlib pandas scikit-learn

import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt

# --------------------
# 1. Load CSV
# --------------------
embeddings_csv = "trained_embeddings/image_embeddings_org_img.csv"
df = pd.read_csv(embeddings_csv)

# Separate metadata
names = df['name']
labels = df['label']

# Extract only embedding columns
embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
embeddings = df[embedding_cols].values  # Shape: (num_samples, 512)

# --------------------
# 2. UMAP - 2D for visualization
# --------------------
umap_2d = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42
).fit_transform(embeddings)

# Save 2D embeddings
umap_2d_df = pd.DataFrame({
    "name": names,
    "label": labels,
    "umap_1": umap_2d[:, 0],
    "umap_2": umap_2d[:, 1]
})
umap_2d_df.to_csv(f"{embeddings_csv}_umap_2d.csv", index=False)

# --------------------
# 3. UMAP - 30D for clustering
# --------------------
# umap_30d = umap.UMAP(
#     n_neighbors=15,
#     min_dist=0.1,
#     n_components=2,  # More dimensions to keep structure for clustering
#     metric='euclidean',
#     random_state=42
# ).fit_transform(embeddings)

# # Save 30D embeddings
# umap_30d_df = pd.DataFrame(umap_30d, columns=[f"umap_{i+1}" for i in range(2)])
# umap_30d_df.insert(0, "label", labels)
# umap_30d_df.insert(0, "name", names)
# umap_30d_df.to_csv(f"{embeddings_csv[:-4]}_umap_2d.csv", index=False)

# --------------------
# 4. Plot 2D result
# --------------------
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    umap_2d[:, 0],
    umap_2d[:, 1],
    c=labels,
    cmap='coolwarm',
    s=10
)
plt.colorbar(scatter, label="Class Label")
plt.title("UMAP Projection (2D)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()

print("✅ Saved: embeddings_umap_2d.csv and embeddings_umap_30d.csv")
