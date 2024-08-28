# Clustering Techniques for Animal Behavior Analysis
# Part of the "QuantifyingBehavior" course

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score

# Generate sample data
np.random.seed(42)
n_samples = 300

# Simulating animal movement data (x, y coordinates)
data = np.concatenate([
    np.random.normal(0, 1, (n_samples//3, 2)),
    np.random.normal(3, 1, (n_samples//3, 2)),
    np.random.normal(-2, 1.5, (n_samples//3, 2))
])

# Create a DataFrame
df = pd.DataFrame(data, columns=['x', 'y'])

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], alpha=0.6)
plt.title('Simulated Animal Movement Data')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.show()

# 1. K-means Clustering

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(data_scaled)

# Visualize K-means results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['x'], df['y'], c=df['kmeans_cluster'], cmap='viridis', alpha=0.6)
plt.title('K-means Clustering of Animal Movement Data')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.colorbar(scatter)
plt.show()

# Choosing the optimal number of clusters

# Function to compute inertia (within-cluster sum of squares)
def compute_inertia(k, data):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    return kmeans.inertia_

# Function to compute silhouette score
def compute_silhouette(k, data):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)
    return silhouette_score(data, labels)

# Range of k values to try
k_range = range(1, 11)

# Compute inertia and silhouette score for each k
inertias = [compute_inertia(k, data_scaled) for k in k_range]
silhouette_scores = [compute_silhouette(k, data_scaled) if k > 1 else 0 for k in k_range]

# Plot elbow curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(k_range[1:], silhouette_scores[1:], 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')

plt.tight_layout()
plt.show()

# Print the optimal k based on silhouette score
optimal_k = silhouette_scores.index(max(silhouette_scores[1:])) + 1
print(f"The optimal number of clusters based on silhouette analysis is: {optimal_k}")

# Visualize clustering with optimal k
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
df['kmeans_optimal'] = kmeans_optimal.fit_predict(data_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['x'], df['y'], c=df['kmeans_optimal'], cmap='viridis', alpha=0.6)
plt.title(f'K-means Clustering with Optimal k={optimal_k}')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.colorbar(scatter)
plt.show()

# Analysis and Interpretation of Cluster Selection
print("\nAnalysis and Interpretation of Cluster Selection:")
print("1. The Elbow Method:")
print("   - Plots the inertia (within-cluster sum of squares) against the number of clusters.")
print("   - The 'elbow' in the curve suggests the optimal number of clusters.")
print("   - After this point, adding more clusters doesn't significantly reduce inertia.")
print("\n2. Silhouette Analysis:")
print("   - Measures how similar an object is to its own cluster compared to other clusters.")
print("   - Higher silhouette scores indicate better-defined clusters.")
print("   - The peak in the silhouette score suggests the optimal number of clusters.")
print(f"\n3. In this case, the optimal number of clusters is {optimal_k}.")
print("   - This may differ from our initial assumption of 3 clusters.")
print("\n4. Implications for Animal Behavior Analysis:")
print("   - The optimal k might reveal more or fewer distinct behavioral states than expected.")
print("   - It's crucial to balance statistical optimization with biological interpretation.")
print("   - Consider domain knowledge when deciding on the final number of clusters to use.")

# 2. Hierarchical Clustering

# Perform hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
df['hierarchical_cluster'] = hierarchical.fit_predict(data_scaled)

# Visualize hierarchical clustering results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['x'], df['y'], c=df['hierarchical_cluster'], cmap='viridis', alpha=0.6)
plt.title('Hierarchical Clustering of Animal Movement Data')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.colorbar(scatter)
plt.show()

# Create dendrogram
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)

hierarchical_dendro = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
hierarchical_dendro = hierarchical_dendro.fit(data_scaled)

plt.figure(figsize=(10, 7))
plot_dendrogram(hierarchical_dendro, truncate_mode='level', p=3)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Comparison of clustering results
print(df[['kmeans_cluster', 'hierarchical_cluster', 'kmeans_optimal']].value_counts().sort_index())

# Analysis and Interpretation
print("\nAnalysis and Interpretation:")
print("1. Both K-means and Hierarchical clustering identified three main clusters in the animal movement data.")
print("2. The clusters likely represent different behavioral states or habitats.")
print("3. K-means provides centroids that could represent average positions for each behavior.")
print("4. Hierarchical clustering allows for exploration of sub-clusters within main clusters.")
print("5. The dendrogram visualizes the hierarchy of clusters, which can be useful for understanding relationships between behaviors.")
print("\nNext steps could include:")
print("- Analyzing the characteristics of each cluster (e.g., average speed, direction)")
print("- Investigating transitions between clusters over time")
print("- Correlating clusters with environmental or physiological data")