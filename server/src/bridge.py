# Importamos el módulo sys para acceder a argumentos de consola
# y a la configuración de entrada/salida estándar.
import sys

# Si la salida estándar permite reconfiguración,
# se establece la codificación UTF-8 para evitar errores con acentos y caracteres especiales.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Lo mismo para la salida de errores estándar.
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Importamos la clase principal del chatbot.
from bot import ChatBot

# Creamos una única instancia global del bot.
# Esto permite reutilizar el chatbot cada vez que se ejecute una consulta.
bot = ChatBot()


def main():
    # Verificamos que el usuario haya enviado al menos un argumento
    # después del nombre del archivo.
    if len(sys.argv) < 2:
        print("Mensaje vacio")
        return

    # Unimos todos los argumentos recibidos en un solo texto.
    # Ejemplo:
    # python bridge.py quiero una laptop barata
    # se convierte en: "quiero una laptop barata"
    texto = " ".join(sys.argv[1:])

    # Enviamos el texto al chatbot para obtener una respuesta.
    respuesta = bot.responder(texto)

    # Mostramos la respuesta generada por el bot.
    print(respuesta)


# Punto de entrada del programa.
# Solo ejecuta main() si este archivo es corrido directamente.
if __name__ == "__main__":
    main()