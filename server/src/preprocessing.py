
# IMPORTACIONES


# re para trabajar con expresiones regulares
import re

# string para manejar signos de puntuación
import string



# LIMPIEZA BÁSICA DE TEXTO

def limpiar_texto(texto):
    """
    Limpia un texto eliminando ruido innecesario.

    Pasos:
    1. Convierte a minúsculas
    2. Elimina números
    3. Elimina signos de puntuación
    4. Elimina espacios innecesarios
    """

    # Convertimos el texto a string y a minúsculas
    texto = str(texto).lower()

    # Eliminamos números (ej: "iphone 13" → "iphone ")
    texto = re.sub(r"\d+", "", texto)

    # Eliminamos signos de puntuación (.,!? etc.)
    texto = texto.translate(str.maketrans("", "", string.punctuation))

    # Eliminamos espacios al inicio y final
    texto = texto.strip()

    return texto



# PREPROCESAMIENTO COMPLETO


def preprocesar_texto(texto):
    """
    Aplica limpieza y normalización del texto.

    - Limpia el texto usando limpiar_texto()
    - Divide en palabras
    - Vuelve a unir para tener texto limpio uniforme
    """

    # Limpieza básica
    texto = limpiar_texto(texto)

    # Separa el texto en palabras
    palabras = texto.split()

    # Une nuevamente en una cadena limpia
    return " ".join(palabras)



# TOKENIZACIÓN


def tokenizar(texto):
    """
    Convierte el texto en una lista de palabras (tokens).

    Ejemplo:
    "hola mundo" → ["hola", "mundo"]
    """

    # Primero se limpia y normaliza el texto
    texto = preprocesar_texto(texto)

    # Se divide en palabras individuales
    return texto.split()