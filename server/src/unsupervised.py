# IMPORTACIONES

# Path para manejo seguro de rutas
from pathlib import Path

# Pandas para manipulación de datos
import pandas as pd

# Matplotlib para visualización de resultados
import matplotlib.pyplot as plt

# TF-IDF para vectorizar texto
from sklearn.feature_extraction.text import TfidfVectorizer

# Algoritmos de clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# PCA para reducción de dimensionalidad
from sklearn.decomposition import PCA

# Métrica para evaluar clustering
from sklearn.metrics import silhouette_score

# Función de preprocesamiento de texto
from preprocessing import preprocesar_texto


# CONFIGURACIÓN DE RUTAS

BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR.parent
DATA_DIR = SERVER_DIR / "data"



# FUNCIÓN PRINCIPAL


def ejecutar_no_supervisado():
    """
    Realiza análisis no supervisado sobre las reviews de productos.

    Incluye:
    - Limpieza de texto
    - Vectorización con TF-IDF
    - Evaluación de KMeans con silhouette score
    - Aplicación de KMeans, DBSCAN y Agglomerative Clustering
    - Visualización con PCA
    """

    
    # CARGA DE DATOS
  

    ruta_productos = DATA_DIR / "amazon_1000_products.csv"

    # Leemos el dataset
    df = pd.read_csv(ruta_productos)

    # Eliminamos filas sin review
    df = df.dropna(subset=["review"])

    # Creamos una columna con texto preprocesado
    df["review_clean"] = df["review"].astype(str).apply(preprocesar_texto)

    print("Dataset Amazon cargado:", df.shape)

    
    # VECTORIZACIÓN DEL TEXTO
   

    # Convertimos las reviews limpias en vectores numéricos
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["review_clean"])

    
    # BÚSQUEDA DEL MEJOR K PARA KMEANS
    

    scores = []

    # Probamos distintos números de clusters
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)

        # Entrenamos y obtenemos etiquetas
        labels = km.fit_predict(X)

        # Calculamos silhouette score
        score = silhouette_score(X, labels)

        # Guardamos resultado
        scores.append((k, score))

        print(f"K={k} -> silhouette={score:.4f}")

    # Seleccionamos el mejor k según silhouette
    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"\n✅ Mejor número de clusters para KMeans: {best_k}")

    
    # KMEANS FINAL
    

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

    # Asignamos cluster a cada review
    df["kmeans"] = kmeans.fit_predict(X)

    
    # DBSCAN
    

    # DBSCAN requiere matriz densa
    dbscan = DBSCAN(eps=0.9, min_samples=3)
    df["dbscan"] = dbscan.fit_predict(X.toarray())

    
    # AGGLOMERATIVE CLUSTERING
    

    agglomerative = AgglomerativeClustering(n_clusters=best_k)
    df["agglomerative"] = agglomerative.fit_predict(X.toarray())

    
    # DISTRIBUCIÓN DE CLUSTERS
    

    print("\nDistribución KMeans:")
    print(df["kmeans"].value_counts())

    print("\nDistribución DBSCAN:")
    print(df["dbscan"].value_counts())

    print("\nDistribución Agglomerative:")
    print(df["agglomerative"].value_counts())

    
    # EJEMPLOS POR CLUSTER

    print("\nEjemplos por cluster KMeans:")

    for cluster_id in sorted(df["kmeans"].unique()):
        print(f"\n--- Cluster {cluster_id} ---")

        # Mostramos algunos ejemplos de cada cluster
        muestras = df[df["kmeans"] == cluster_id][["name", "category", "review"]].head(5)
        print(muestras.to_string(index=False))

    
    # REDUCCIÓN DE DIMENSIONALIDAD CON PCA
    

    # Reducimos los vectores a 2 dimensiones para graficar
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    
    # VISUALIZACIÓN
    

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["kmeans"])
    plt.title("Clusters KMeans con PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()



# PUNTO DE ENTRADA

if __name__ == "__main__":
    ejecutar_no_supervisado()