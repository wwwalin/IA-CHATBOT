// 
// IMPORTACIONES DE REACT
// 

import { useEffect, useRef, useState } from "react";


// 
// COMPONENTE PRINCIPAL
// 

export default function App() {

  // 
  // ESTADOS DEL COMPONENTE
  // 

  // Lista de mensajes del chat (historial)
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hola, ¿Qué producto estás buscando?" }
  ]);

  // Texto actual del input
  const [text, setText] = useState("");

  // Estado de carga (cuando el bot está respondiendo)
  const [loading, setLoading] = useState(false);

  // Referencia para hacer scroll automático al último mensaje
  const bottomRef = useRef(null);


  // 
  // EFECTO: AUTO-SCROLL
  // 

  useEffect(() => {
    // Cada vez que llegan mensajes nuevos, baja automáticamente
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);


  // 
  // FUNCIÓN PARA ENVIAR MENSAJES
  // 

  const sendMessage = async (e) => {
    e.preventDefault(); // Evita recargar la página

    const clean = text.trim();

    // Validación: evitar mensajes vacíos o múltiples envíos
    if (!clean || loading) return;

    // Mensaje del usuario
    const userMsg = {
      sender: "user",
      text: clean
    };

    // Se agrega el mensaje del usuario al chat
    setMessages((prev) => [...prev, userMsg]);

    // Limpia el input
    setText("");

    // Activa estado de carga
    setLoading(true);

    try {
      // 
      // PETICIÓN A LA API (BACKEND)
      // 

      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: clean })
      });

      // Manejo de error si falla la conexión
      if (!response.ok) {
        throw new Error("No se pudo conectar con la API");
      }

      // Convierte la respuesta a JSON
      const data = await response.json();

      // 
      // RESPUESTA DEL BOT
      // 

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: data.reply || "No hubo respuesta del servidor."
        }
      ]);

    } catch (error) {
      // 
      // MANEJO DE ERRORES
      // 

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `Error: ${error.message}`
        }
      ]);

    } finally {
      // Siempre desactiva el loading
      setLoading(false);
    }
  };


  // 
  // RENDER DEL COMPONENTE
  // 

  return (
    <div style={styles.page}>
      <div style={styles.card}>

        {/* TÍTULO */}
        <h1 style={styles.title}>Chat IA</h1>

        {/* 
            CAJA DE MENSAJES
         */}
        <div style={styles.chatBox}>

          {/* Renderiza todos los mensajes */}
          {messages.map((msg, i) => (
            <div
              key={i}
              style={{
                ...styles.message,

                // Alineación según quien envía
                alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",

                // Color diferente para usuario y bot
                background: msg.sender === "user" ? "#dbeafe" : "#f3f4f6"
              }}
            >
              <strong>{msg.sender === "user" ? "Tú" : "Bot"}:</strong>

              {/* Permite saltos de línea */}
              <div style={{ whiteSpace: "pre-wrap" }}>{msg.text}</div>
            </div>
          ))}

          {/* Mensaje mientras el bot escribe */}
          {loading && (
            <div
              style={{
                ...styles.message,
                alignSelf: "flex-start",
                background: "#f3f4f6"
              }}
            >
              <strong>Bot:</strong>
              <div>Escribiendo...</div>
            </div>
          )}

          {/* Referencia para scroll automático */}
          <div ref={bottomRef} />
        </div>


        {/* 
            FORMULARIO DE ENVÍO
         */}
        <form onSubmit={sendMessage} style={styles.form}>

          {/* Input del usuario */}
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Escribe un mensaje..."
            style={styles.input}
          />

          {/* Botón de envío */}
          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? "Enviando..." : "Enviar"}
          </button>
        </form>

      </div>
    </div>
  );
}


// 
// ESTILOS (INLINE CSS)
// 

const styles = {

  // Página completa
  page: {
    minHeight: "100vh",
    display: "grid",
    placeItems: "center",
    background: "#f1f5f9",
    padding: 20
  },

  // Tarjeta principal
  card: {
    width: "100%",
    maxWidth: 700,
    background: "white",
    borderRadius: 16,
    boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
    padding: 20
  },

  title: {
    marginBottom: 15
  },

  // Contenedor del chat
  chatBox: {
    height: 450,
    overflowY: "auto",
    border: "1px solid #e5e7eb",
    borderRadius: 12,
    padding: 12,
    display: "flex",
    flexDirection: "column",
    gap: 10,
    marginBottom: 16
  },

  // Mensajes individuales
  message: {
    maxWidth: "75%",
    padding: "10px 12px",
    borderRadius: 12,
    lineHeight: 1.4
  },

  // Formulario
  form: {
    display: "flex",
    gap: 8
  },

  input: {
    flex: 1,
    padding: 12,
    borderRadius: 10,
    border: "1px solid #d1d5db"
  },

  button: {
    padding: "12px 16px",
    border: "none",
    borderRadius: 10,
    background: "#2563eb",
    color: "white",
    cursor: "pointer",
    opacity: 1
  }
};