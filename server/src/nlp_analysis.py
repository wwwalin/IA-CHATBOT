
# IMPORTACIONES


# Path para manejar rutas de archivos de forma segura
from pathlib import Path

# Pandas para leer y manipular datasets
import pandas as pd

# Matplotlib para generar visualizaciones
import matplotlib.pyplot as plt

# Counter para contar frecuencias de palabras
from collections import Counter

# CountVectorizer para extraer palabras y bigramas
from sklearn.feature_extraction.text import CountVectorizer

# LDA para detección de temas en texto
from sklearn.decomposition import LatentDirichletAllocation

# WordCloud para generar nube de palabras
from wordcloud import WordCloud

# Funciones propias de preprocesamiento
from preprocessing import tokenizar, preprocesar_texto



# CONFIGURACIÓN DE RUTAS


# Ruta del archivo actual
BASE_DIR = Path(__file__).resolve().parent

# Carpeta principal del servidor
SERVER_DIR = BASE_DIR.parent

# Carpeta de datos
DATA_DIR = SERVER_DIR / "data"



# FUNCIÓN PARA OBTENER BIGRAMAS

def obtener_bigramas(textos, top_n=10):
    """
    Extrae los bigramas (pares de palabras consecutivas)
    más frecuentes del conjunto de textos.
    """

    # Vectorizador configurado para bigramas
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words="english")

    # Convierte los textos en matriz de frecuencias
    X = vectorizer.fit_transform(textos)

    # Suma las frecuencias de cada bigrama
    frecuencias = X.sum(axis=0).A1

    # Obtiene el vocabulario de bigramas
    vocabulario = vectorizer.get_feature_names_out()

    # Une vocabulario y frecuencias, luego ordena de mayor a menor
    pares = sorted(zip(vocabulario, frecuencias), key=lambda x: x[1], reverse=True)

    # Devuelve solo los top_n más frecuentes
    return pares[:top_n]



# FUNCIÓN PARA MOSTRAR TEMAS CON LDA


def mostrar_temas_lda(textos, n_topics=3, n_palabras=6):
    """
    Aplica LDA (Latent Dirichlet Allocation) para detectar
    temas principales dentro del conjunto de textos.
    """

    # Vectorizador basado en conteo de palabras
    vectorizer = CountVectorizer(stop_words="english")

    # Convierte textos a matriz documento-término
    X = vectorizer.fit_transform(textos)

    # Crea el modelo LDA con el número de temas indicado
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

    # Entrena el modelo sobre los textos
    lda.fit(X)

    # Obtiene las palabras del vocabulario
    palabras = vectorizer.get_feature_names_out()

    print("\nTemas detectados con LDA:")

    # Recorre cada tema detectado
    for i, tema in enumerate(lda.components_):
        # Selecciona las palabras más representativas del tema
        top = tema.argsort()[-n_palabras:][::-1]
        palabras_tema = [palabras[j] for j in top]

        print(f"Tema {i + 1}: {palabras_tema}")



# FUNCIÓN PRINCIPAL DE ANÁLISIS NLP

def ejecutar_nlp():
    """
    Realiza análisis de procesamiento de lenguaje natural
    sobre las reviews de productos del dataset.
    """

    # Ruta del archivo de productos
    ruta_productos = DATA_DIR / "amazon_1000_products.csv"

    # Carga el dataset
    df = pd.read_csv(ruta_productos)

    # Elimina filas sin review o rating
    df = df.dropna(subset=["review", "rating"])

    # Convierte rating a numérico
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Elimina filas con rating inválido
    df = df.dropna(subset=["rating"])

    # Crea una etiqueta simple de sentimiento basada en el rating
    df["sentiment"] = df["rating"].apply(lambda x: "positivo" if x >= 4 else "negativo")

    # Limpia y preprocesa las reviews
    df["review_clean"] = df["review"].astype(str).apply(preprocesar_texto)

    print("Dataset Amazon cargado:", df.shape)

    
    # TOKENIZACIÓN Y FRECUENCIA DE PALABRAS

    tokens = []

    # Recorremos cada review limpia y acumulamos sus tokens
    for texto in df["review_clean"]:
        tokens.extend(tokenizar(texto))

    print("\nTop 15 palabras más frecuentes:")
    print(Counter(tokens).most_common(15))

    # BIGRAMAS

    print("\nTop 10 bigramas:")
    print(obtener_bigramas(df["review_clean"], top_n=10))

    # TEMAS CON LDA
    

    mostrar_temas_lda(df["review_clean"], n_topics=3, n_palabras=6)

    # PALABRAS FRECUENTES POR SENTIMIENTO

    print("\nPalabras más frecuentes en opiniones positivas:")
    positivos = " ".join(df[df["sentiment"] == "positivo"]["review_clean"])
    print(Counter(positivos.split()).most_common(10))

    print("\nPalabras más frecuentes en opiniones negativas:")
    negativos = " ".join(df[df["sentiment"] == "negativo"]["review_clean"])
    print(Counter(negativos.split()).most_common(10))

    # NUBE DE PALABRAS

    # Creamos la nube de palabras con todas las reviews limpias
    nube = WordCloud(width=1000, height=500, background_color="white")
    nube.generate(" ".join(df["review_clean"]))

    # Mostramos la visualización
    plt.figure(figsize=(12, 6))
    plt.imshow(nube, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nube de palabras de reviews")
    plt.show()


# PUNTO DE ENTRADA

if __name__ == "__main__":
    ejecutar_nlp()