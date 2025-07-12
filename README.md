
# 🎧 Spotify Music Clustering Using Machine Learning

This project applies **unsupervised machine learning (KMeans Clustering)** to a Spotify dataset to group similar songs based on their audio features. By analyzing properties like energy, danceability, and valence, the model finds patterns and forms clusters of songs with similar moods.

---

## 📌 Objective

To use **KMeans clustering** to segment Spotify songs into distinct clusters such as:
- 🎵 Chill / Acoustic Vibes
- 💃 Energetic Dance Tracks
- 😢 Sad Ballads
- 🔊 High-Energy Pop Songs

This project is ideal for learning **data preprocessing, clustering, and dimensionality reduction with PCA**.

---

## 📁 Dataset Used

The dataset is a cleaned Spotify dataset containing audio features for a large number of tracks.

### ✅ Key columns used:
| Column           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `track_name`     | Name of the song                                                            |
| `artist_name`    | Name of the artist                                                          |
| `danceability`   | How suitable a track is for dancing (0.0–1.0)                              |
| `energy`         | Intensity and activity of the song (0.0–1.0)                                |
| `valence`        | Musical positivity/happiness of the song (0.0–1.0)                          |
| `tempo`          | Speed or pace of the song (beats per minute)                               |
| `acousticness`   | Confidence measure of whether the track is acoustic                        |
| `instrumentalness`| Predicts whether a track contains no vocals                              |

📂 Place the dataset file as:
```
spotify-clustering-project/
├── data/
│   └── spotify_dataset.csv
```

---

## 🛠️ Technologies & Libraries

- Python 3
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🚀 Steps Performed

### 1. **Data Loading and Cleaning**
- Load CSV data using `pandas`
- Drop missing values for selected features

### 2. **Feature Selection**
Used only numerical features relevant for mood-based clustering:
```python
features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness']
```

### 3. **Data Scaling**
Used `StandardScaler` to normalize feature values for fair clustering.

### 4. **KMeans Clustering**
- Set `n_clusters=4` based on experimental tuning
- Fit model and predict cluster labels for each song

### 5. **PCA for Visualization**
- Reduced high-dimensional feature space into 2D
- Plotted clusters using a color-coded scatterplot

### 6. **Cluster Interpretation**
Mapped cluster numbers to meaningful mood labels:
```python
cluster_labels = {
    0: "Chill Vibes",
    1: "Energetic Pop",
    2: "Sad Ballads",
    3: "Dance Floor Bangers"
}
```

---

## 📊 Sample Output

| track_name     | artist_name     | cluster_name        |
|----------------|-----------------|---------------------|
| Let Her Go     | Passenger        | Sad Ballads         |
| Uptown Funk    | Bruno Mars       | Dance Floor Bangers |
| Shape of You   | Ed Sheeran       | Energetic Pop       |
| Someone Like You | Adele         | Sad Ballads         |

---

## 📈 Visualization Example

PCA-based 2D clustering plot shows how songs are grouped:

```
X-axis: PCA Component 1  
Y-axis: PCA Component 2  
Color: Cluster number
```

Each point is a song.

---

## ✅ Learnings

- Real-world application of **unsupervised learning**
- Mastered **feature scaling** and **dimensionality reduction**
- Developed understanding of how audio features reflect musical mood

---

## 🔮 Future Improvements

- Tune number of clusters using Elbow Method / Silhouette Score
- Add more genre or lyrical data
- Connect to Spotify API for dynamic song analysis

---

## 📂 Project Structure

```
spotify-clustering-project/
├── data/
│   └── spotify_dataset.csv
├── spotify_clustering.py       ← Main ML script
├── README.md                   ← Project documentation
```

---

## 🧑‍💻 Author

**[Your Name]**  
📫 [Your Email] · 🌐 [LinkedIn/GitHub link]

---

## 📜 License

This project is open-source under the MIT License.
