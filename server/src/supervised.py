# IMPORTACIONES

# Path para manejo de rutas
from pathlib import Path

# pickle para guardar el modelo entrenado
import pickle

# pandas para manejar datos
import pandas as pd

# TF-IDF para convertir texto en vectores numéricos
from sklearn.feature_extraction.text import TfidfVectorizer

# Modelo de clasificación (Machine Learning supervisado)
from sklearn.linear_model import LogisticRegression

# Función de limpieza de texto
from preprocessing import limpiar_texto



# CONFIGURACIÓN DE RUTAS


BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR.parent
DATA_DIR = SERVER_DIR / "data"
MODELS_DIR = SERVER_DIR / "models"


# DETECCIÓN DE INTENCIÓN (REGLAS)

def detectar_intencion(texto):
    """
    Clasifica el texto en una intención básica usando reglas.

    Esto se usa para generar etiquetas automáticamente
    (ya que el dataset no viene etiquetado).
    """

    texto = limpiar_texto(texto)

    # Saludos
    if any(p in texto for p in ["hola", "hello", "hi", "buenas"]):
        return "saludo"

    # Categorías de productos
    if "laptop" in texto or "notebook" in texto:
        return "laptop"

    if "tablet" in texto:
        return "tablet"

    if any(p in texto for p in ["celular", "telefono", "teléfono", "phone", "movil", "móvil"]):
        return "celular"

    # Consultas de precio
    if any(p in texto for p in ["precio", "cuanto cuesta", "cuánto cuesta", "costo", "vale"]):
        return "precio"

    # Solicitudes de recomendación
    if any(p in texto for p in ["recomiendame", "recomiéndame", "recomendar"]):
        return "recomendacion"

    # Caso general
    return "general"


# ENTRENAMIENTO DEL MODELO

def entrenar_modelo_intencion():
    """
    Entrena un modelo de Machine Learning para clasificar
    la intención del usuario a partir de texto.
    """

    print(" Iniciando entrenamiento...")

    # CARGA DE DATOS

    ruta_conversaciones = DATA_DIR / "conversaciones_500.csv"

    df = pd.read_csv(ruta_conversaciones)

    # Aseguramos nombres de columnas correctos
    df.columns = ["input", "output"]

    print(" Dataset conversaciones cargado:", df.shape)
    print(df.head(5).to_string(index=False))

    # PREPROCESAMIENTO

    # Limpieza del texto
    df["input"] = df["input"].astype(str).apply(limpiar_texto)

    # Generamos etiquetas automáticamente
    df["intent"] = df["input"].apply(detectar_intencion)

    print("\n Distribución de intenciones:")
    print(df["intent"].value_counts())

    # VARIABLES PARA EL MODELO

    X = df["input"]   # Texto
    y = df["intent"]  # Etiqueta (intención)

    # VECTORIZACIÓN (TF-IDF)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigramas y bigramas
        max_features=3000    # Número máximo de características
    )

    # Transformamos texto en vectores
    X_vec = vectorizer.fit_transform(X)

    # ENTRENAMIENTO DEL MODELO

    # Usamos Regresión Logística
    model = LogisticRegression(max_iter=1000)

    # Entrenamos el modelo
    model.fit(X_vec, y)

    # GUARDADO DEL MODELO

    # Creamos carpeta si no existe
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Guardamos el modelo entrenado
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Guardamos el vectorizador
    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n Modelo de intención entrenado correctamente")
    print("Archivos guardados en:", MODELS_DIR)

    # Mostramos las clases aprendidas
    print("Clases aprendidas:", list(model.classes_))


# PUNTO DE ENTRADA

if __name__ == "__main__":
    entrenar_modelo_intencion()