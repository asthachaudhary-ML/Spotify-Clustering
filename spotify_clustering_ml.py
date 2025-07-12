# spotify_clustering.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Step 1: Load the dataset
df = pd.read_csv('"C:\Users\mjcho\OneDrive\spotify-clustering-project\data\data\spotify_dataset.csv"')
print("✅ Dataset loaded")

# Step 2: Select features for clustering
features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness']
df_clean = df[features].dropna()

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Step 4: Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Step 5: Assign human-readable cluster names
cluster_labels = {
    0: "Chill Vibes",
    1: "Energetic Pop",
    2: "Sad Ballads",
    3: "Dance Floor Bangers"
}
df['cluster_name'] = df['cluster'].map(cluster_labels)

# Step 6: Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster_name'], palette='Set2', s=100)
plt.title("🎵 Spotify Song Clusters (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Mood Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Show top 5 songs from each cluster (if available)
if 'track_name' in df.columns and 'artist_name' in df.columns:
    for cluster_id, name in cluster_labels.items():
        print(f"\n🎶 Top songs in '{name}' cluster:")
        songs = df[df['cluster'] == cluster_id][['track_name', 'artist_name']].head(5)
        print(songs.to_string(index=False))
