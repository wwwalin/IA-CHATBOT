# Esta clase contiene toda la lógica del sistema conversacional
from bot import ChatBot

# Mensaje inicial para indicar que el programa está iniciando
print("Iniciando programa...")


def main():
    # Se crea una instancia del chatbot
    # Esto inicializa modelos, datos y configuraciones necesarias
    bot = ChatBot()

    # Mensaje indicando que el bot está listo para interactuar
    print("BOT IA listo (exit para salir)")

    # Bucle infinito para mantener la conversación activa
    while True:
        try:
            # Se solicita al usuario que ingrese un mensaje
            user = input("Tu: ")
        except KeyboardInterrupt:
            # Manejo de interrupción con Ctrl + C
            # Permite salir del programa de forma controlada
            print("\nBot: Hasta luego")
            break

        # Si el usuario escribe "exit", se finaliza el programa
        if user.lower() == "exit":
            print("Bot: Hasta luego")
            break

        # Se envía el mensaje al chatbot y se obtiene la respuesta
        # Luego se imprime en consola
        print("Bot:", bot.responder(user))


# Punto de entrada del programa
if __name__ == "__main__":
    main()