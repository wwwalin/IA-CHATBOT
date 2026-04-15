
// IMPORTACIONES PRINCIPALES


// Importa React (necesario para usar JSX)
import React from "react";

// Importa ReactDOM para renderizar la app en el navegador
import ReactDOM from "react-dom/client";

// Importa el componente principal de la aplicación
import App from "./App.jsx";


// =========================
// RENDERIZADO DE LA APP
// =========================

// Se selecciona el elemento HTML con id="root"
// Este elemento está en el archivo index.html
const rootElement = document.getElementById("root");

// Se crea la raíz de React (modo moderno - React 18+)
const root = ReactDOM.createRoot(rootElement);

// Se renderiza la aplicación dentro del DOM
root.render(

  // StrictMode es una herramienta de React para detectar errores
  // Solo funciona en desarrollo (no afecta producción)
  <React.StrictMode>

    {/* Componente principal de la app */}
    <App />

  </React.StrictMode>
);