# Importamos Path para trabajar con rutas de archivos
# de forma segura y compatible entre sistemas operativos.
from pathlib import Path

# Importamos pandas para leer y manipular archivos CSV.
import pandas as pd

# Ruta del directorio donde se encuentra este archivo.
BASE_DIR = Path(__file__).resolve().parent

# Directorio raíz del backend o carpeta superior.
SERVER_DIR = BASE_DIR.parent

# Carpeta donde se almacenan los datasets del proyecto.
DATA_DIR = SERVER_DIR / "data"


def cargar_conversaciones():
    # Construye la ruta completa del archivo de conversaciones.
    ruta = DATA_DIR / "conversaciones_500.csv"

    # Lee el archivo CSV en un DataFrame.
    df = pd.read_csv(ruta)

    # Renombra las columnas para asegurar una estructura uniforme:
    # input = mensaje del usuario
    # output = respuesta asociada
    df.columns = ["input", "output"]

    # Convierte ambas columnas a texto para evitar problemas de tipo.
    df["input"] = df["input"].astype(str)
    df["output"] = df["output"].astype(str)

    # Devuelve el DataFrame ya preparado.
    return df


def cargar_productos():
    # Construye la ruta completa del archivo de productos.
    ruta = DATA_DIR / "amazon_1000_products.csv"

    # Lee el archivo CSV en un DataFrame.
    df = pd.read_csv(ruta)

    # Normaliza los nombres de columnas:
    # - convierte a minúsculas
    # - elimina espacios innecesarios
    df.columns = [c.lower().strip() for c in df.columns]

    # Limpia la columna de categoría para trabajar con valores consistentes.
    df["category"] = df["category"].astype(str).str.lower().str.strip()

    # Convierte el precio a valor numérico.
    # Si hay errores de conversión, se reemplazan por NaN.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Convierte el rating a valor numérico.
    # Si hay errores de conversión, se reemplazan por NaN.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Elimina filas que no tengan información esencial para el sistema.
    # Esto garantiza que el chatbot solo trabaje con productos válidos.
    df = df.dropna(subset=["name", "brand", "category", "price", "rating", "review"])

    # Devuelve el DataFrame limpio y listo para usar.
    return df