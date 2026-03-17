import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

# descargar recursos (solo la primera vez)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def limpiar_texto(texto):
    """
    Limpia una reseña aplicando:
    - minúsculas
    - eliminación de signos
    - tokenización
    - eliminación de stopwords
    - lematización
    """

    if pd.isna(texto):
        return ""

    # minúsculas
    texto = texto.lower()

    # eliminar signos y números
    texto = re.sub(r'[^a-z\s]', '', texto)

    # tokenizar
    palabras = word_tokenize(texto)

    # eliminar stopwords
    palabras = [p for p in palabras if p not in stop_words]

    # lematizar
    palabras = [lemmatizer.lemmatize(p) for p in palabras]

    return " ".join(palabras)


def aplicar_limpieza(df, columna_texto):
    """
    Aplica la limpieza a todo el dataset
    """
    df["clean_review"] = df[columna_texto].apply(limpiar_texto)
    return df


def vectorizar_tfidf(df, columna_texto="clean_review", max_features=3000):
    """
    Convierte el texto limpio en números usando TF-IDF
    """

    vectorizer = TfidfVectorizer(max_features=max_features)

    X = vectorizer.fit_transform(df[columna_texto])

    return X, vectorizer