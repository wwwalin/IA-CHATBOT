# Path para manejar rutas de archivos de forma segura
from pathlib import Path

# pickle para cargar modelos previamente entrenados
import pickle


# CONFIGURACIÓN DE RUTAS


# Ruta del archivo actual
BASE_DIR = Path(__file__).resolve().parent

# Carpeta principal del servidor
SERVER_DIR = BASE_DIR.parent

# Carpeta donde se almacenan los modelos entrenados
MODELS_DIR = SERVER_DIR / "models"



# CLASE MODELO DE INTENCIÓN


class ModeloIntencion:
    def __init__(self):
        """
        Constructor de la clase.
        Carga el modelo de clasificación y el vectorizador previamente entrenados.
        """

        # Cargamos el modelo entrenado (ej: Logistic Regression, Random Forest, etc.)
        with open(MODELS_DIR / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Cargamos el vectorizador TF-IDF utilizado durante el entrenamiento
        with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predecir(self, texto):
        """
        Predice la intención del usuario a partir de un texto.

        Pasos:
        1. Convierte el texto en un vector numérico usando el vectorizador.
        2. El modelo clasifica ese vector.
        3. Devuelve la intención predicha.
        """

        # Transformamos el texto en vector (igual que en entrenamiento)
        X = self.vectorizer.transform([texto])

        # Realizamos la predicción y devolvemos el resultado
        return self.model.predict(X)[0]