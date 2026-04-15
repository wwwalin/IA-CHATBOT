# IMPORTACIONES

# Manejo de rutas
from pathlib import Path

# Para guardar modelos
import pickle

# Manejo de datos
import pandas as pd

# Vectorización de texto
from sklearn.feature_extraction.text import TfidfVectorizer

# Modelo supervisado
from sklearn.linear_model import LogisticRegression

# Función de limpieza de texto
from preprocessing import limpiar_texto


# CONFIGURACIÓN DE RUTAS

BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR.parent
DATA_DIR = SERVER_DIR / "data"
MODELS_DIR = SERVER_DIR / "models"



# FUNCIÓN DE ETIQUETADO


def detectar_intencion(texto):
    """
    Clasifica el texto en una intención usando reglas simples.

    Sirve para generar etiquetas automáticamente
    para el entrenamiento del modelo.
    """

    texto = limpiar_texto(texto)

    # Saludos
    if any(p in texto for p in ["hola", "hello", "hi", "buenas"]):
        return "saludo"

    # Categorías
    if "laptop" in texto or "notebook" in texto:
        return "laptop"

    if "tablet" in texto:
        return "tablet"

    if any(p in texto for p in ["celular", "telefono", "teléfono", "phone", "movil", "móvil"]):
        return "celular"

    # Precio
    if any(p in texto for p in ["precio", "cuanto cuesta", "cuánto cuesta", "costo", "vale"]):
        return "precio"

    # Recomendaciones
    if any(p in texto for p in ["recomiendame", "recomiéndame", "recomendar"]):
        return "recomendacion"

    # Caso general
    return "general"



# ENTRENAMIENTO DEL MODELO


def entrenar_modelo_intencion():
    """
    Entrena un modelo de clasificación de intención
    utilizando TF-IDF + Regresión Logística.
    """

    print("Iniciando entrenamiento...")

    
    # CARGA DEL DATASET
    

    ruta_conversaciones = DATA_DIR / "conversaciones_500.csv"

    df = pd.read_csv(ruta_conversaciones)

    # Aseguramos nombres correctos
    df.columns = ["input", "output"]

    print(" Dataset conversaciones cargado:", df.shape)
    print(df.head(5).to_string(index=False))

    
    # PREPROCESAMIENTO
    

    # Limpieza del texto
    df["input"] = df["input"].astype(str).apply(limpiar_texto)

    # Generación automática de etiquetas
    df["intent"] = df["input"].apply(detectar_intencion)

    print("\n Distribución de intenciones:")
    print(df["intent"].value_counts())

    
    # VARIABLES DEL MODELO
   

    X = df["input"]   # Entrada (texto)
    y = df["intent"]  # Salida (intención)

    
    # VECTORIZACIÓN
    

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # unigramas y bigramas
        max_features=3000    # límite de características
    )

    # Convertimos texto a vectores
    X_vec = vectorizer.fit_transform(X)

    
    # ENTRENAMIENTO
    

    model = LogisticRegression(max_iter=1000)

    model.fit(X_vec, y)

    
    # GUARDADO
    

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n Modelo de intención entrenado correctamente")
    print(" Archivos guardados en:", MODELS_DIR)

    # Clases aprendidas
    print(" Clases aprendidas:", list(model.classes_))



# EJECUCIÓN


if __name__ == "__main__":
    entrenar_modelo_intencion()