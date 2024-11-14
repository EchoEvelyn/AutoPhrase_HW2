from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np

model = Word2Vec.load("/Users/echo/Desktop/AutoPhrase/models/DBLP/word2vec_model.model")

phrases = list(model.wv.index_to_key)
embeddings = np.array([model.wv[phrase] for phrase in phrases])

num_clusters = 6

kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans_labels = kmeans.fit_predict(embeddings)

gmm = GaussianMixture(n_components=num_clusters, random_state=0)
gmm_labels = gmm.fit_predict(embeddings)

def print_cluster_samples(phrases, labels, method_name):
    print(f"\n{method_name} Clustering Results:")
    for cluster_num in range(num_clusters):
        cluster_indices = np.where(labels == cluster_num)[0]
        cluster_phrases = [phrases[i] for i in cluster_indices[:20]]
        print(f"Cluster {cluster_num + 1}: {', '.join(cluster_phrases)}")

print_cluster_samples(phrases, kmeans_labels, "K-Means")
print_cluster_samples(phrases, gmm_labels, "GMM")
