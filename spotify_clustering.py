import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/spotify_dataset.csv')
print("Data loaded successfully!")
print(df.head())

# Select important features
features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness']
df_clean = df[features].dropna()

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# Reduce dimensions using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Spotify Song Clusters")
plt.show()

# 1. Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 2. Load your Spotify dataset
df = pd.read_csv('data/spotify_dataset.csv')
print("Data loaded successfully!")

# 3. Select audio features for clustering
features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness']
df_clean = df[features].dropna()  # Remove rows with missing values

# 4. Scale the features (important for clustering!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# 5. Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# ✅ 6. Add the cluster labels to the original DataFrame
df = df.iloc[:len(clusters)]  # Ensure shape matches
df['cluster'] = clusters

# 7. Visualize with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 8. Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Spotify Song Clusters")
plt.show()
if 'track_name' in df.columns:
    for i in range(4):
        print(f"\n🎵 Cluster {i} songs:")
        print(df[df['cluster'] == i][['track_name', 'artist_name']].head())

import seaborn as sns
import matplotlib.pyplot as plt

for feature in ['energy', 'danceability', 'tempo', 'valence']:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='cluster', y=feature)
    plt.title(f"{feature.capitalize()} across Clusters")
    plt.show()

cluster_labels = { 0: "chilli" }
cluster_labels = {1: "Energetic Pop"}  
cluster_labels = {2: "Sad Ballads"}  
cluster_labels = {3: "Dance Floor Bangers"}

# Optional: Rename clusters based on feature trends (you can change names after reviewing)
cluster_labels = {
    0: "Chill Vibes",
    1: "Energetic Dance",
    2: "Sad & Slow",
    3: "Pop & Upbeat"
}

df['cluster_name'] = df['cluster'].map(cluster_labels)
print(df[['track_name', 'artist_name', 'cluster_name']].head())

