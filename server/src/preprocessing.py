import re
import string


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"\d+", "", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = texto.strip()
    return texto


def preprocesar_texto(texto):
    texto = limpiar_texto(texto)
    palabras = texto.split()
    return " ".join(palabras)


def tokenizar(texto):
    texto = preprocesar_texto(texto)
    return texto.split()