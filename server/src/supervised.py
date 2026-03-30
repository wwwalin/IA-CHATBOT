from pathlib import Path
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocessing import limpiar_texto

BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR.parent
DATA_DIR = SERVER_DIR / "data"
MODELS_DIR = SERVER_DIR / "models"


def detectar_intencion(texto):
    texto = limpiar_texto(texto)

    if any(p in texto for p in ["hola", "hello", "hi", "buenas"]):
        return "saludo"
    if "laptop" in texto or "notebook" in texto:
        return "laptop"
    if "tablet" in texto:
        return "tablet"
    if any(p in texto for p in ["celular", "telefono", "teléfono", "phone", "movil", "móvil"]):
        return "celular"
    if any(p in texto for p in ["precio", "cuanto cuesta", "cuánto cuesta", "costo", "vale"]):
        return "precio"
    if any(p in texto for p in ["recomiendame", "recomiéndame", "recomendar"]):
        return "recomendacion"
    return "general"


def entrenar_modelo_intencion():
    print(" Iniciando entrenamiento...")

    ruta_conversaciones = DATA_DIR / "conversaciones_500.csv"
    df = pd.read_csv(ruta_conversaciones)
    df.columns = ["input", "output"]

    print(" Dataset conversaciones cargado:", df.shape)
    print(df.head(5).to_string(index=False))

    df["input"] = df["input"].astype(str).apply(limpiar_texto)
    df["intent"] = df["input"].apply(detectar_intencion)

    print("\n Distribución de intenciones:")
    print(df["intent"].value_counts())

    X = df["input"]
    y = df["intent"]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000
    )

    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n Modelo de intención entrenado correctamente")
    print("Archivos guardados en:", MODELS_DIR)
    print("Clases aprendidas:", list(model.classes_))


if __name__ == "__main__":
    entrenar_modelo_intencion()