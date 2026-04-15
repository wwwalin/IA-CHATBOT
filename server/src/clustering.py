# Importamos librerías necesarias
import os  # Para manejar rutas de archivos
import pandas as pd  # Para trabajar con datos en formato tabla
from sklearn.feature_extraction.text import TfidfVectorizer  # Para vectorizar texto
from sklearn.cluster import KMeans  # Algoritmo de clustering
import numpy as np  # Operaciones numéricas


def ejecutar_clustering():
    # Mensaje inicial
    print("🔵 Iniciando clustering con KMeans...")

    # Obtenemos la ruta base del archivo actual
    base_dir = os.path.dirname(__file__)

    # Definimos rutas de entrada y salida
    ruta_entrada = os.path.join(base_dir, "..", "data", "conversaciones_500.csv")
    ruta_salida = os.path.join(base_dir, "..", "data", "dataset_clusterizado.csv")

    print("📥 Leyendo archivo:", ruta_entrada)

    # Cargamos el dataset
    data = pd.read_csv(ruta_entrada)

    # Validamos que exista la columna necesaria
    if "input" not in data.columns:
        raise ValueError("El CSV debe tener una columna 'input'")

    # Convertimos los textos a string
    textos = data["input"].astype(str)


    #  VECTORIZACIÓN (TF-IDF)
  
    # Convertimos texto a números para que el modelo pueda entenderlo
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigramas y bigramas
        max_features=500     # Limitamos el número de características
    )

    # Transformamos los textos en vectores numéricos
    X = vectorizer.fit_transform(textos)

 
    # CLUSTERING (KMEANS)
 
    # Creamos el modelo de clustering
    kmeans = KMeans(
        n_clusters=3,     # Número de grupos
        random_state=42,  # Reproducibilidad
        n_init=10         # Número de inicializaciones
    )

    # Entrenamos el modelo
    kmeans.fit(X)

    # Asignamos a cada texto su cluster correspondiente
    data["cluster"] = kmeans.labels_


    # 💾 GUARDAR RESULTADOS

    data.to_csv(ruta_salida, index=False)

    print("✅ Clustering completado")
    print("💾 Archivo guardado en:", ruta_salida)


    # 📊 ESTADÍSTICAS

    print("\n📊 DISTRIBUCIÓN DE CLUSTERS:")

    conteo = data["cluster"].value_counts().sort_index()
    total = len(data)

    # Mostramos cantidad y porcentaje por cluster
    for cluster, cantidad in conteo.items():
        porcentaje = (cantidad / total) * 100
        print(f"Cluster {cluster}: {cantidad} elementos ({porcentaje:.2f}%)")

    # PALABRAS CLAVE POR CLUSTER
    print("\n🧠 PALABRAS CLAVE POR CLUSTER:")

    # Obtenemos términos del vectorizador
    terms = vectorizer.get_feature_names_out()

    # Para cada cluster, mostramos las palabras más representativas
    for i in range(kmeans.n_clusters):
        center = kmeans.cluster_centers_[i]

        # Ordenamos por importancia
        top_indices = center.argsort()[-10:][::-1]

        palabras = [terms[ind] for ind in top_indices]

        print(f"\nCluster {i}:")
        print(", ".join(palabras))

    # EJEMPLOS POR CLUSTER
   
    print("\n📌 EJEMPLOS POR CLUSTER:")

    # Mostramos ejemplos reales de cada grupo
    for i in range(3):
        print(f"\n🔹 Cluster {i}:")
        ejemplos = data[data["cluster"] == i]["input"].head(5)

        if ejemplos.empty:
            print("   (sin ejemplos)")
        else:
            for j, texto in enumerate(ejemplos, start=1):
                print(f"   {j}. {texto}")


# Punto de entrada del programa
if __name__ == "__main__":
    ejecutar_clustering()