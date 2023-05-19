import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# loading CSV file into dataframe
df = pd.read_csv('result.csv', encoding='unicode_escape')
# extract text from Passage column
passages = df['Passage'].tolist()

# initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(passages)
X = tfidf_matrix.toarray()

# Extract additional columns for clustering
cols = ['Rank', 'Offset', 'length', 'tagid']
X_additional = df[cols].values

# Concatenate TF-IDF matrix with the additional columns
X_combined = np.concatenate((X, X_additional), axis=1)

# Perform KMeans clustering with n clusters
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(X)

# Add new column with cluster labels to DataFrame
df['Cluster'] = labels

# Save clustered data to new CSV file
df.to_csv('cluster_file.csv', index=False)

# Fit the PCA model to the tf_idf matrix
pca = PCA(n_components=2)
X_pca = pca.fit_transform(tfidf_matrix.toarray())

# Plot the clusters
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='rainbow')
legend = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Clusters")
ax.add_artist(legend)
plt.show()