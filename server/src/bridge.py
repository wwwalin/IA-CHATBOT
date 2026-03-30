import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from bot import ChatBot

bot = ChatBot()

def main():
    if len(sys.argv) < 2:
        print("Mensaje vacio")
        return

    texto = " ".join(sys.argv[1:])
    respuesta = bot.responder(texto)
    print(respuesta)

if __name__ == "__main__":
    main()