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
import platform

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

def setup_korean_font():
    """Set up Korean font based on the operating system."""
    system = platform.system()

    fonts_to_try = []

    if system == 'Darwin':  
        fonts_to_try = [
            '/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
            '/Library/Fonts/NanumGothic.ttf'
        ]
    elif system == 'Windows':
        fonts_to_try = [
            'C:/Windows/Fonts/malgun.ttf',
            'C:/Windows/Fonts/gulim.ttf',
            'C:/Windows/Fonts/NanumGothic.ttf'
        ]
    elif system == 'Linux':
        fonts_to_try = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf'
        ]

    for font_path in fonts_to_try:
        if os.path.exists(font_path):
            font_properties = font_manager.FontProperties(fname=font_path)
            font_name = font_properties.get_name()
            plt.rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Using font: {font_name} from {font_path}")
            return font_path

    print("No Korean font found, trying fallback method...")
    try:

        plt.rc('font', family='sans-serif')
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Malgun Gothic', 'AppleGothic', 
                                           'NanumGothic', 'NanumBarunGothic', 'Noto Sans CJK KR']
        plt.rcParams['axes.unicode_minus'] = False
        return None
    except:
        print("Warning: Could not set up Korean font. Text may not display correctly.")
        return None

def visualize_clusters(embeddings, labels, articles):
    """Visualize clusters with simplified text labels."""
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 10:
        cmap = plt.cm.get_cmap('tab10', n_clusters)
    elif n_clusters <= 20:
        cmap = plt.cm.get_cmap('tab20', n_clusters)
    else:
        cmap = plt.cm.get_cmap('nipy_spectral', n_clusters)

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

    plt.title("UMAP + HDBSCAN Visualize Result", fontsize=20)
    plt.xlabel("PCA 1", fontsize=15)
    plt.ylabel("PCA 2", fontsize=15)

    handles, _ = scatter.legend_elements(prop="colors")
    legend_labels = [f"Cluster {label}" for label in unique_labels]
    plt.legend(handles, legend_labels, title="Clusters", fontsize=12, title_fontsize=14)

    plt.tight_layout()
    plt.grid(True)
    plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nReference list of articles by index:")
    for i, article in enumerate(articles):
        print(f"Article {i}: {article.strip()}")

def main():

    setup_korean_font()

    articles = load_articles('example_articles.txt')
    embeddings = generate_embeddings(articles)
    umap_embeddings = reduce_dimensions(embeddings)
    cluster_labels = perform_clustering(umap_embeddings)

    visualize_clusters(umap_embeddings, cluster_labels, articles)

    print("\n[HDBSCAN 군집화 결과]")
    cluster_dict = {}
    unique_clusters = sorted(set(cluster_labels))
    for cluster in unique_clusters:
        cluster_dict[cluster] = [articles[i] for i, label in enumerate(cluster_labels) if label == cluster]
        print(f"\n클러스터 {cluster}:")
        for article in cluster_dict[cluster]:
            print(f"- {article.strip()}")

    with open('cluster_results.txt', 'w', encoding='utf-8') as f:
        f.write("[HDBSCAN 군집화 결과]\n")
        for cluster in unique_clusters:
            f.write(f"\n클러스터 {cluster}:\n")
            for article in cluster_dict[cluster]:
                f.write(f"- {article.strip()}\n")

    print("\nClustering results saved to 'cluster_results.txt'")
    print("Visualization saved as 'cluster_visualization.png'")

if __name__ == "__main__":
    main()