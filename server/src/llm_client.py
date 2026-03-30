import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
API_URL = "https://router.huggingface.co/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def _debe_omitir_llm(respuesta_base):
    if not respuesta_base:
        return True

    texto = str(respuesta_base).strip()
    if len(texto) < 40:
        return True

    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    numeradas = sum(1 for l in lineas if l[:2].isdigit() or (len(l) > 1 and l[0].isdigit() and l[1] == "."))

    if numeradas >= 2:
        return True

    pistas = [
        "1.", "2.", "3.",
        "rating:", "review:", "precio:",
        "si quieres, puedo darte más detalles",
        "estos son los productos",
        "te recomiendo estos productos"
    ]

    texto_lower = texto.lower()
    return any(p in texto_lower for p in pistas)


def mejorar_respuesta(mensaje_usuario, respuesta_base):
    """
    Mejora respuestas solo cuando vale la pena.
    Si HF falla, devuelve la respuesta original.
    """
    if not HF_TOKEN:
        return respuesta_base

    if _debe_omitir_llm(respuesta_base):
        return respuesta_base

    try:
        payload = {
            "model": HF_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Reescribe el texto en español de forma natural, clara y amable. "
                        "No cambies nombres de productos, marcas, precios, ratings ni categorías. "
                        "No resumas listas ni elimines información importante. "
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
            "temperature": 0.4,
            "max_tokens": 180
        }

        response = SESSION.post(API_URL, json=payload, timeout=20)

        if response.status_code != 200:
            return respuesta_base

        data = response.json()
        contenido = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not contenido or not contenido.strip():
            return respuesta_base

        return contenido.strip()

    except Exception:
        return respuesta_base