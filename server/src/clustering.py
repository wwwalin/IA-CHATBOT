import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


def ejecutar_clustering():
    print("🔵 Iniciando clustering con KMeans...")

    base_dir = os.path.dirname(__file__)

    ruta_entrada = os.path.join(base_dir, "..", "data", "conversaciones_500.csv")
    ruta_salida = os.path.join(base_dir, "..", "data", "dataset_clusterizado.csv")

    print("📥 Leyendo archivo:", ruta_entrada)

    data = pd.read_csv(ruta_entrada)

    if "input" not in data.columns:
        raise ValueError("El CSV debe tener una columna 'input'")

    textos = data["input"].astype(str)

    # 🔥 TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500
    )

    X = vectorizer.fit_transform(textos)

    # 🔥 KMeans
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    data["cluster"] = kmeans.labels_

    # 🔥 GUARDAR
    data.to_csv(ruta_salida, index=False)

    print("✅ Clustering completado")
    print("💾 Archivo guardado en:", ruta_salida)

    # =========================
    # 📊 ESTADÍSTICAS
    # =========================

    print("\n📊 DISTRIBUCIÓN DE CLUSTERS:")

    conteo = data["cluster"].value_counts().sort_index()
    total = len(data)

    for cluster, cantidad in conteo.items():
        porcentaje = (cantidad / total) * 100
        print(f"Cluster {cluster}: {cantidad} elementos ({porcentaje:.2f}%)")

    # =========================
    # 🔥 PALABRAS MÁS IMPORTANTES
    # =========================

    print("\n🧠 PALABRAS CLAVE POR CLUSTER:")

    terms = vectorizer.get_feature_names_out()

    for i in range(kmeans.n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-10:][::-1]

        palabras = [terms[ind] for ind in top_indices]

        print(f"\nCluster {i}:")
        print(", ".join(palabras))

    # =========================
    # 📌 EJEMPLOS
    # =========================

    print("\n📌 EJEMPLOS POR CLUSTER:")

    for i in range(3):
        print(f"\n🔹 Cluster {i}:")
        ejemplos = data[data["cluster"] == i]["input"].head(5)

        if ejemplos.empty:
            print("   (sin ejemplos)")
        else:
            for j, texto in enumerate(ejemplos, start=1):
                print(f"   {j}. {texto}")


if __name__ == "__main__":
    ejecutar_clustering()