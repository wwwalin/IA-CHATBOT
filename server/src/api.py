from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

bot = None
bot_error = None

try:
    print("Importando ChatBot...")
    from bot import ChatBot

    print("Creando bot...")
    bot = ChatBot()

    print("Bot cargado correctamente")
except Exception as e:
    bot_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    print(bot_error)

@app.get("/")
def root():
    if bot_error:
        return {"ok": False, "error": bot_error}
    return {"ok": True, "message": "API del bot funcionando"}

@app.post("/chat")
def chat(req: ChatRequest):
    if bot_error:
        return {"reply": "El bot no pudo cargarse", "error": bot_error}

    texto = req.message.strip()

    if not texto:
        return {"reply": "Escribe un mensaje válido."}

    respuesta = bot.responder(texto)
    return {"reply": respuesta}