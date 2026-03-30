import { useEffect, useRef, useState } from "react";

export default function App() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hola, ¿Qué producto estás buscando?" }
  ]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async (e) => {
    e.preventDefault();

    const clean = text.trim();
    if (!clean || loading) return;

    const userMsg = {
      sender: "user",
      text: clean
    };

    setMessages((prev) => [...prev, userMsg]);
    setText("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: clean })
      });

      if (!response.ok) {
        throw new Error("No se pudo conectar con la API");
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: data.reply || "No hubo respuesta del servidor."
        }
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `Error: ${error.message}`
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h1 style={styles.title}>Chat IA</h1>

        <div style={styles.chatBox}>
          {messages.map((msg, i) => (
            <div
              key={i}
              style={{
                ...styles.message,
                alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
                background: msg.sender === "user" ? "#dbeafe" : "#f3f4f6"
              }}
            >
              <strong>{msg.sender === "user" ? "Tú" : "Bot"}:</strong>
              <div style={{ whiteSpace: "pre-wrap" }}>{msg.text}</div>
            </div>
          ))}

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

          <div ref={bottomRef} />
        </div>

        <form onSubmit={sendMessage} style={styles.form}>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Escribe un mensaje..."
            style={styles.input}
          />
          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? "Enviando..." : "Enviar"}
          </button>
        </form>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    display: "grid",
    placeItems: "center",
    background: "#f1f5f9",
    padding: 20
  },
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
  message: {
    maxWidth: "75%",
    padding: "10px 12px",
    borderRadius: 12,
    lineHeight: 1.4
  },
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