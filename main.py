from sentence_transformers import SentenceTransformer
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from matplotlib import font_manager
from adjustText import adjust_text
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import umap.umap_ as umap
import numpy as np
import hdbscan
import pickle
import os

font_path = load_dotenv('FONT_PATH')

def load_articles(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        articles = file.readlines()
    return articles

def generate_embeddings(articles, model_name='nlpai-lab/KoE5', save_path='embeddings.pkl'):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        model = SentenceTransformer(model_name)
        prefixed_articles = ["passage: " + article.strip() for article in articles]
        embeddings = model.encode(prefixed_articles)
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings

def reduce_dimensions(embeddings, n_neighbors=15, n_components=5):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric='cosine',
        n_jobs=-1
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

def perform_clustering(embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric='euclidean'
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    return cluster_labels

def set_korean_font(font_path):
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False

def visualize_clusters(embeddings, labels):

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    cmap = ListedColormap(plt.get_cmap('tab20').colors[:n_clusters])

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap=cmap,
        s=100,
        alpha=0.7,
        edgecolors='k'
    )
    plt.title("UMAP + HDBSCAN 군집화 결과 시각화", fontsize=20)
    plt.xlabel("PCA 1", fontsize=15)
    plt.ylabel("PCA 2", fontsize=15)

    handles, _ = scatter.legend_elements(prop="colors")
    legend_labels = [f"cluster {label}" for label in unique_labels]
    plt.legend(handles, legend_labels, title="cluster", fontsize=12, title_fontsize=14)

    texts = []
    for i, label in enumerate(labels):
        texts.append(plt.text(
            reduced_embeddings[i, 0],
            reduced_embeddings[i, 1],
            str(label),
            fontsize=10,
            ha='center',
            va='center'
        ))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    plt.grid(True)
    plt.show()

articles = load_articles('example_articles.txt')
embeddings = generate_embeddings(articles)
umap_embeddings = reduce_dimensions(embeddings)
cluster_labels = perform_clustering(umap_embeddings)
set_korean_font(font_path)

visualize_clusters(umap_embeddings, cluster_labels)

print("\n[HDBSCAN 군집화 결과]")
cluster_dict = {}
unique_clusters = set(cluster_labels)
for cluster in unique_clusters:
    cluster_dict[cluster] = [articles[i] for i, label in enumerate(cluster_labels) if label == cluster]
    print(f"\n클러스터 {cluster}:")
    for article in cluster_dict[cluster]:
        print(f"- {article.strip()}")