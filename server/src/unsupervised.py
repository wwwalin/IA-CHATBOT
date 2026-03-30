from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from preprocessing import preprocesar_texto

BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR.parent
DATA_DIR = SERVER_DIR / "data"


def ejecutar_no_supervisado():
    ruta_productos = DATA_DIR / "amazon_1000_products.csv"
    df = pd.read_csv(ruta_productos)
    df = df.dropna(subset=["review"])
    df["review_clean"] = df["review"].astype(str).apply(preprocesar_texto)

    print("Dataset Amazon cargado:", df.shape)

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["review_clean"])

    scores = []
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append((k, score))
        print(f"K={k} -> silhouette={score:.4f}")

    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"\n✅ Mejor número de clusters para KMeans: {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["kmeans"] = kmeans.fit_predict(X)

    dbscan = DBSCAN(eps=0.9, min_samples=3)
    df["dbscan"] = dbscan.fit_predict(X.toarray())

    agglomerative = AgglomerativeClustering(n_clusters=best_k)
    df["agglomerative"] = agglomerative.fit_predict(X.toarray())

    print("\nDistribución KMeans:")
    print(df["kmeans"].value_counts())

    print("\nDistribución DBSCAN:")
    print(df["dbscan"].value_counts())

    print("\nDistribución Agglomerative:")
    print(df["agglomerative"].value_counts())

    print("\nEjemplos por cluster KMeans:")
    for cluster_id in sorted(df["kmeans"].unique()):
        print(f"\n--- Cluster {cluster_id} ---")
        muestras = df[df["kmeans"] == cluster_id][["name", "category", "review"]].head(5)
        print(muestras.to_string(index=False))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["kmeans"])
    plt.title("Clusters KMeans con PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()


if __name__ == "__main__":
    ejecutar_no_supervisado()