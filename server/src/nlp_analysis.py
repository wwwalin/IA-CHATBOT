from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from preprocessing import tokenizar, preprocesar_texto

BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR.parent
DATA_DIR = SERVER_DIR / "data"


def obtener_bigramas(textos, top_n=10):
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words="english")
    X = vectorizer.fit_transform(textos)
    frecuencias = X.sum(axis=0).A1
    vocabulario = vectorizer.get_feature_names_out()
    pares = sorted(zip(vocabulario, frecuencias), key=lambda x: x[1], reverse=True)
    return pares[:top_n]


def mostrar_temas_lda(textos, n_topics=3, n_palabras=6):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(textos)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    palabras = vectorizer.get_feature_names_out()

    print("\nTemas detectados con LDA:")
    for i, tema in enumerate(lda.components_):
        top = tema.argsort()[-n_palabras:][::-1]
        palabras_tema = [palabras[j] for j in top]
        print(f"Tema {i + 1}: {palabras_tema}")


def ejecutar_nlp():
    ruta_productos = DATA_DIR / "amazon_1000_products.csv"
    df = pd.read_csv(ruta_productos)
    df = df.dropna(subset=["review", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    df["sentiment"] = df["rating"].apply(lambda x: "positivo" if x >= 4 else "negativo")
    df["review_clean"] = df["review"].astype(str).apply(preprocesar_texto)

    print("Dataset Amazon cargado:", df.shape)

    tokens = []
    for texto in df["review_clean"]:
        tokens.extend(tokenizar(texto))

    print("\nTop 15 palabras más frecuentes:")
    print(Counter(tokens).most_common(15))

    print("\nTop 10 bigramas:")
    print(obtener_bigramas(df["review_clean"], top_n=10))

    mostrar_temas_lda(df["review_clean"], n_topics=3, n_palabras=6)

    print("\nPalabras más frecuentes en opiniones positivas:")
    positivos = " ".join(df[df["sentiment"] == "positivo"]["review_clean"])
    print(Counter(positivos.split()).most_common(10))

    print("\nPalabras más frecuentes en opiniones negativas:")
    negativos = " ".join(df[df["sentiment"] == "negativo"]["review_clean"])
    print(Counter(negativos.split()).most_common(10))

    nube = WordCloud(width=1000, height=500, background_color="white")
    nube.generate(" ".join(df["review_clean"]))

    plt.figure(figsize=(12, 6))
    plt.imshow(nube, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nube de palabras de reviews")
    plt.show()


if __name__ == "__main__":
    ejecutar_nlp()