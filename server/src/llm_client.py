
# IMPORTACIONES


import os  # Para manejar variables de entorno
import requests  # Para hacer solicitudes HTTP a la API
from dotenv import load_dotenv  # Para cargar variables desde archivo .env

# Carga las variables de entorno desde el archivo .env
load_dotenv()


# CONFIGURACIÓN DE API


# Token de autenticación de Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")

# Modelo a utilizar (por defecto Llama 3)
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

# Endpoint de la API de Hugging Face
API_URL = "https://router.huggingface.co/v1/chat/completions"

# Cabeceras para autenticación y tipo de contenido
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Creamos una sesión HTTP reutilizable (mejora rendimiento)
SESSION = requests.Session()
SESSION.headers.update(HEADERS)



# VALIDACIÓN PARA USAR LLM


def _debe_omitir_llm(respuesta_base):
    """
    Determina si NO se debe usar el modelo LLM.
    Se evita en respuestas cortas o estructuradas
    para no dañar formato ni gastar recursos innecesarios.
    """

    if not respuesta_base:
        return True

    texto = str(respuesta_base).strip()

    # Si la respuesta es muy corta, no vale la pena mejorarla
    if len(texto) < 40:
        return True

    # Detecta si es una lista numerada
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    numeradas = sum(
        1 for l in lineas
        if l[:2].isdigit() or (len(l) > 1 and l[0].isdigit() and l[1] == ".")
    )

    if numeradas >= 2:
        return True

    # Detecta patrones típicos de respuestas estructuradas
    pistas = [
        "1.", "2.", "3.",
        "rating:", "review:", "precio:",
        "si quieres, puedo darte más detalles",
        "estos son los productos",
        "te recomiendo estos productos"
    ]

    texto_lower = texto.lower()

    # Si contiene estos patrones, se omite el LLM
    return any(p in texto_lower for p in pistas)


# MEJORADOR DE RESPUESTAS

def mejorar_respuesta(mensaje_usuario, respuesta_base):
    """
    Mejora la redacción de la respuesta utilizando un modelo LLM.

    - Solo se aplica cuando vale la pena.
    - Si falla la API, devuelve la respuesta original.
    """

    # Si no hay token, no se usa el LLM
    if not HF_TOKEN:
        return respuesta_base

    # Si la respuesta no necesita mejora, se devuelve igual
    if _debe_omitir_llm(respuesta_base):
        return respuesta_base

    try:

        # CREACIÓN DEL PROMPT

        payload = {
            "model": HF_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Reescribe el texto en español de forma natural, clara y amable. "
                        "No cambies nombres de productos, marcas, precios, ratings ni categorías. "
                        "NO elimines información importante. "
                        "No inventes datos."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Usuario: {mensaje_usuario}\n\n"
                        f"Texto base:\n{respuesta_base}\n\n"
                        "Mejora solo la redacción."
                    )
                }
            ],
            # Control de creatividad del modelo
            "temperature": 0.4,

            # Límite de tokens de respuesta
            "max_tokens": 180
        }


        # LLAMADA A LA API


        response = SESSION.post(API_URL, json=payload, timeout=20)

        # Si la API falla, devolvemos la respuesta original
        if response.status_code != 200:
            return respuesta_base

        # Convertimos la respuesta a JSON
        data = response.json()

        # Extraemos el contenido generado
        contenido = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Validamos que haya contenido útil
        if not contenido or not contenido.strip():
            return respuesta_base

        # Retornamos la respuesta mejorada
        return contenido.strip()

    except Exception:
        # Si ocurre cualquier error (timeout, conexión, etc.)
        # devolvemos la respuesta original
        return respuesta_base