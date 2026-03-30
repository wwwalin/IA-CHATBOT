from bot import ChatBot

print("Iniciando programa...")

def main():
    bot = ChatBot()
    print("BOT IA listo (exit para salir)")

    while True:
        try:
            user = input("Tu: ")
        except KeyboardInterrupt:
            print("\nBot: Hasta luego")
            break

        if user.lower() == "exit":
            print("Bot: Hasta luego")
            break

        print("Bot:", bot.responder(user))

if __name__ == "__main__":
    main()