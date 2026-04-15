from difflib import get_close_matches
import re
import unicodedata

# Carga las conversaciones simples del dataset
from chat_dataset import cargar_conversaciones, cargar_productos

# Limpia el texto base del usuario
from preprocessing import limpiar_texto

# Modelo supervisado para detectar intención
from model import ModeloIntencion

# Función opcional para mejorar redacción con LLM
from llm_client import mejorar_respuesta


class ChatBot:
    def __init__(self):
        # Dataset de preguntas y respuestas generales
        self.conversaciones = cargar_conversaciones()

        # Dataset de productos
        self.productos = cargar_productos()

        # Modelo de intención entrenado
        self.modelo = ModeloIntencion()

        # Últimos productos recomendados
        self.ultima_recomendacion = []

        # Últimos productos no recomendados
        self.ultima_no_recomendacion = []

        # Producto actual del que se está hablando
        self.producto_actual = None

        # Última categoría consultada
        self.ultima_categoria = None

        # Resultados completos de la categoría actual
        self.resultados_categoria = []

        # Índice para paginar resultados recomendados
        self.indice_resultados = 0

        # Resultados completos de productos no recomendados
        self.resultados_no_recomendados = []

        # Índice para paginar productos no recomendados
        self.indice_no_recomendados = 0

        # Tamaño de página al mostrar resultados
        self.tamano_pagina = 3

        # Indica si el último flujo fue de recomendación o no recomendación
        self.ultimo_modo = None  # recomendado | no_recomendado

        # Variables extra de contexto conversacional
        self.ultimo_producto_mencionado = None
        self.ultima_lista_mostrada = []
        self.ultimo_tema = None

    
    # UTILIDADES GENERALES
    
    def normalizar(self, texto):
        """
        Normaliza el texto del usuario o del sistema para facilitar búsquedas:
        - pasa a minúsculas
        - quita acentos
        - separa letras y números
        - elimina símbolos raros
        - reduce espacios dobles
        """
        texto = str(texto).lower().strip()
        texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
        texto = texto.replace("dle", "del")
        texto = texto.replace("qeu", "que")

        # Separa letras y números para reconocer mejor nombres tipo "Book8"
        texto = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", texto)
        texto = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", texto)

        # Elimina caracteres especiales
        texto = re.sub(r"[^\w\s]", " ", texto)

        # Elimina espacios repetidos
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    def _es_respuesta_estructurada(self, respuesta):
        """
        Detecta si una respuesta tiene estructura fija:
        listas numeradas, precios, ratings, etc.
        Esto sirve para no mandar esas respuestas al LLM.
        """
        if not respuesta:
            return False

        lineas = [l.strip() for l in respuesta.splitlines() if l.strip()]

        # Si hay al menos 2 líneas numeradas, ya se considera estructurada
        if len(lineas) >= 3:
            numeradas = sum(1 for l in lineas if re.match(r"^\d+\.", l))
            if numeradas >= 2:
                return True

        patrones = [
            "1.", "2.", "3.",
            "rating:", "review:", "precio:",
            "aqui tienes otras opciones",
            "te recomiendo estos productos",
            "estos son los productos"
        ]

        texto = self.normalizar(respuesta)
        return any(p in texto for p in patrones)

    def es_respuesta_con_datos_sensibles(self, respuesta):
        """
        Detecta respuestas con datos concretos de productos.
        Estas respuestas NO deberían ser reescritas por el LLM
        para evitar que cambie nombres, precios o ratings.
        """
        if not respuesta:
            return False

        texto = self.normalizar(respuesta)
        patrones = [
            "rating", "cuesta $", "precio de", "el precio de",
            "la marca de", "la review", "review dice",
            "es un producto de la marca", "tiene rating"
        ]
        return any(p in texto for p in patrones)

    def responder_con_estilo(self, mensaje_usuario, respuesta_base, usar_llm=True):
        """
        Mejora la redacción de la respuesta usando LLM solo si conviene.
        Se evita cuando:
        - la respuesta es muy corta
        - no hay respuesta
        - la respuesta tiene estructura fija
        - el llamador decide no usar LLM
        """
        if not respuesta_base:
            return respuesta_base

        respuesta_base = str(respuesta_base).strip()

        if len(respuesta_base) < 40:
            return respuesta_base

        if not usar_llm:
            return respuesta_base

        if self._es_respuesta_estructurada(respuesta_base):
            return respuesta_base

        return mejorar_respuesta(mensaje_usuario, respuesta_base)

    def buscar_respuesta_dataset(self, texto_limpio):
        """
        Busca una coincidencia exacta en el dataset de conversaciones.
        """
        match = self.conversaciones[
            self.conversaciones["input"].astype(str).str.lower().str.strip() == texto_limpio
        ]

        if not match.empty:
            return match.iloc[0]["output"]

        return None

    def detectar_categoria_en_texto(self, texto_limpio):
        """
        Detecta si el usuario está hablando de laptop, tablet o celular,
        incluyendo errores comunes de escritura.
        """
        texto = self.normalizar(texto_limpio)

        if any(x in texto for x in [
            "laptop", "laptops", "notebook", "notebooks",
            "latop", "laptos", "portatil", "portatiles",
            "computadora portatil", "computadoras portatiles",
            "lap", "pc portatil"
        ]):
            return "laptop"

        if any(x in texto for x in [
            "tablet", "tablets", "tableta", "tabletas",
            "table", "tabllet", "tab", "ipad"
        ]):
            return "tablet"

        if any(x in texto for x in [
            "celular", "celulares", "telefono", "telefonos",
            "phone", "phones", "movil", "moviles",
            "cel", "smartphone"
        ]):
            return "celular"

        return None

    def es_agradecimiento(self, texto_norm):
        """
        Detecta mensajes de agradecimiento.
        """
        patrones = [
            "gracias", "muchas gracias", "ok gracias",
            "gracias por la recomendacion", "te lo agradezco",
            "thanks", "thank you"
        ]
        return any(p in texto_norm for p in patrones)

    def es_confirmacion_simple(self, texto_norm):
        """
        Detecta respuestas cortas de confirmación.
        """
        patrones = [
            "ok", "okay", "vale", "bien", "perfecto",
            "esta bien", "dale", "entiendo"
        ]
        return texto_norm in patrones

    def es_reaccion_positiva(self, texto_norm):
        """
        Detecta reacciones positivas del usuario.
        """
        patrones = [
            "excelente", "genial", "muy bien", "perfecto",
            "buenisimo", "bueno", "me gusta", "esta genial",
            "suena bien", "cool", "digo que es genial", "oooh perfecto"
        ]
        return texto_norm in patrones or any(p in texto_norm for p in patrones)

    def responder_reaccion_positiva(self):
        """
        Respuesta amigable cuando el usuario expresa conformidad o entusiasmo.
        """
        if self.producto_actual:
            return (
                f"Me alegra que te guste. Si quieres, puedo decirte más sobre "
                f"{self.producto_actual['name']}, compararlo con otro o explicarte "
                f"si conviene para trabajar, estudiar o uso diario."
            )
        return (
            "Me alegra que te parezca bien. Si quieres, puedo recomendarte más opciones "
            "o compararlas."
        )

    def es_pregunta_identidad(self, texto_norm):
        """
        Detecta preguntas sobre qué es o qué hace el bot.
        """
        patrones = [
            "que eres", "quien eres", "que haces",
            "para que sirves", "eres un bot",
            "eres una ia", "eres una inteligencia artificial"
        ]
        return any(p in texto_norm for p in patrones)

    def es_pregunta_por_que_no(self, texto_norm):
        """
        Detecta preguntas sobre por qué un producto no se recomienda.
        """
        variantes = [
            "por que no", "porque no", "por que",
            "por que no el segundo", "por que no el primero",
            "por que otra cosa no la recomiendas"
        ]
        return any(v in texto_norm for v in variantes)

    def es_pregunta_por_que_si(self, texto_norm):
        """
        Detecta preguntas sobre por qué sí se recomienda un producto.
        """
        variantes = [
            "por que me lo recomiendas",
            "por que lo recomiendas",
            "por que es bueno",
            "que tan bueno es",
            "que hace que sea bueno",
            "por que si",
            "es buena para juegos",
            "es bueno para juegos"
        ]
        return any(v in texto_norm for v in variantes)

    def quiere_mas_resultados(self, texto_norm):
        """
        Detecta si el usuario quiere ver más resultados o más información.
        """
        patrones = [
            "dame otros", "dame otras", "dame otro", "dame otra",
            "dame otros productos", "dame otras opciones",
            "dame mas", "muestrame mas", "quiero ver mas",
            "otros productos", "otras opciones", "otras laptops",
            "otros celulares", "otras tablets", "dame mas informacion"
        ]
        return any(p in texto_norm for p in patrones)

    def quiere_mas_no_recomendados(self, texto_norm):
        """
        Detecta si el usuario quiere ver más productos no recomendados.
        """
        patrones = [
            "otras no me recomiendas",
            "cuales otras no me recomiendas",
            "cual otras no me recomiendas",
            "dame otras no recomendadas",
            "dame otros no recomendados",
            "otros no recomendados",
            "mas no recomendados",
            "mas productos no recomendados"
        ]
        return any(p in texto_norm for p in patrones)

    def pregunta_mejor_producto(self, texto_norm):
        """
        Detecta si el usuario pregunta por el mejor producto.
        """
        patrones = [
            "cual es la mejor", "cual es el mejor",
            "mejor laptop", "mejor celular", "mejor tablet",
            "cual me recomiendas mas", "la mejor opcion",
            "mejor producto", "cual para ti es la mejor",
            "cual para ti es el mejor",
            "cual es la mejor laptop para jugar"
        ]
        return any(p in texto_norm for p in patrones)

    def pregunta_peor_producto(self, texto_norm):
        """
        Detecta si el usuario pregunta por el peor producto.
        """
        patrones = [
            "cual es la peor", "cual es el peor",
            "peor laptop", "peor celular", "peor tablet",
            "dame una mala", "dame uno malo",
            "menos recomendable",
            "cual es la menos recomendable",
            "cual es el menos recomendable",
            "peor producto"
        ]
        return any(p in texto_norm for p in patrones)

    def pregunta_catalogo(self, texto_norm):
        """
        Detecta preguntas generales sobre productos disponibles.
        """
        patrones = [
            "cuales tienes", "que tienes",
            "que productos tienes",
            "que me puedes recomendar",
            "recomiendame productos"
        ]
        return any(p in texto_norm for p in patrones)

    def quiere_ver_categoria(self, texto_norm):
        """
        Detecta intenciones de ver productos de una categoría.
        """
        patrones = [
            "hablame de", "quiero ver", "muestrame",
            "dame", "ensename", "recomiendame",
            "busco", "necesito", "que tienes",
            "cuales tienes", "que productos tienes",
            "recomiendame productos"
        ]
        return any(p in texto_norm for p in patrones)

    def quiere_comparacion_para_trabajo(self, texto_norm):
        """
        Detecta si el usuario busca recomendaciones orientadas a trabajo.
        """
        patrones = [
            "para trabajar", "para trabajo", "para oficina",
            "para productividad", "para estudiar y trabajar",
            "para comparar", "algo interesante para comparar"
        ]
        return any(p in texto_norm for p in patrones)

    def quiere_recomendacion_para_jugar(self, texto_norm):
        """
        Detecta si el usuario busca algo para juegos.
        """
        patrones = [
            "para jugar", "para juegos", "gaming",
            "para game", "para correr juegos"
        ]
        return any(p in texto_norm for p in patrones)

    def _categoria_dataset(self, categoria):
        """
        Ajusta el nombre de categoría al formato usado en el dataset.
        """
        return "phone" if categoria == "celular" else categoria

    def _guardar_contexto_recomendacion(self, categoria, productos):
        """
        Guarda el contexto cuando el flujo es de recomendación.
        """
        self.ultima_categoria = categoria
        self.resultados_categoria = productos
        self.indice_resultados = 0
        self.ultima_recomendacion = []
        self.producto_actual = None
        self.ultimo_modo = None

    def _guardar_contexto_no_recomendacion(self, categoria, productos):
        """
        Guarda el contexto cuando el flujo es de no recomendación.
        """
        self.ultima_categoria = categoria
        self.resultados_no_recomendados = productos
        self.indice_no_recomendados = 0
        self.ultima_no_recomendacion = []
        self.producto_actual = None
        self.ultimo_modo = None

    def _formatear_lista_productos(self, productos, categoria, titulo=None):
        """
        Formatea una lista de productos recomendados.
        """
        if not productos:
            return f"No encontré productos para {categoria}."

        if titulo is None:
            titulo = f"Te recomiendo estos productos de tipo {categoria}:"

        respuesta = titulo + "\n"
        for i, row in enumerate(productos, start=1):
            respuesta += (
                f"{i}. {row['name']} ({row['brand']}) - "
                f"${row['price']} - Rating: {row['rating']}\n"
            )

        respuesta += (
            "\nSi quieres, puedo darte más detalles. "
            "Por ejemplo: 'háblame del primero', 'precio del segundo', "
            "'qué tan bueno es', 'por qué me lo recomiendas', "
            "'cuál no recomiendas' o 'dame otros'."
        )
        return respuesta.strip()

    def _formatear_lista_no_recomendados(self, productos, categoria, titulo=None):
        """
        Formatea una lista de productos no recomendados.
        """
        if not productos:
            return f"No encontré productos poco recomendables para {categoria}."

        if titulo is None:
            titulo = f"Estos son los productos que menos te recomendaría en {categoria}:"

        respuesta = titulo + "\n"
        for i, row in enumerate(productos, start=1):
            respuesta += (
                f"{i}. {row['name']} ({row['brand']}) - "
                f"${row['price']} - Rating: {row['rating']} - Review: {row['review']}\n"
            )

        respuesta += (
            "\nSi quieres, te explico el motivo. "
            "Puedes decirme: 'por qué no', 'por qué no el segundo', "
            "'háblame del primero' o 'dame otros no recomendados'."
        )
        return respuesta.strip()

    def _puntaje_producto(self, producto):
        """
        Calcula un puntaje simple para balancear rating y precio.
        Mayor rating y menor precio = mejor puntaje.
        """
        rating = float(producto["rating"])
        price = float(producto["price"])
        return (rating * 100) - (price * 0.05)

    
    # RECOMENDACIONES
 
    def recomendar(self, categoria, top_n=3):
        """
        Recomienda productos de una categoría.
        """
        categoria = categoria.lower()
        categoria_dataset = self._categoria_dataset(categoria)

        df = self.productos.copy()
        filtrados = df[df["category"] == categoria_dataset]

        if filtrados.empty:
            return f"No encontré productos para {categoria}."

        # Ordena por mejor rating y luego menor precio
        filtrados = filtrados.sort_values(by=["rating", "price"], ascending=[False, True])
        resultados = filtrados.to_dict(orient="records")

        self._guardar_contexto_recomendacion(categoria, resultados)

        primeros = self.resultados_categoria[:top_n]
        self.indice_resultados = len(primeros)
        self.ultima_recomendacion = primeros
        self.producto_actual = primeros[0] if primeros else None
        self.ultimo_modo = "recomendado"

        self.ultima_lista_mostrada = primeros
        self.ultimo_producto_mencionado = primeros[0] if primeros else None
        self.ultimo_tema = "recomendacion"

        return self._formatear_lista_productos(primeros, categoria)

    def mejor_producto(self, categoria):
        """
        Devuelve el mejor producto de una categoría usando el puntaje calculado.
        """
        categoria_dataset = self._categoria_dataset(categoria)
        df = self.productos.copy()
        filtrados = df[df["category"] == categoria_dataset]

        if filtrados.empty:
            return f"No encontré productos para {categoria}."

        productos = filtrados.to_dict(orient="records")
        productos = sorted(productos, key=lambda x: self._puntaje_producto(x), reverse=True)
        mejor = productos[0]

        self.ultima_categoria = categoria
        self.producto_actual = mejor
        self.ultima_recomendacion = [mejor]
        self.ultimo_modo = "recomendado"
        self.ultimo_producto_mencionado = mejor

        return (
            f"La mejor opción en {categoria} es {mejor['name']} de {mejor['brand']}. "
            f"La escogería porque combina muy bien su rating de {mejor['rating']} "
            f"con un precio de ${mejor['price']}. "
            f"Además, su review dice: '{mejor['review']}'."
        )

    def dame_otros(self):
        """
        Muestra otra página de resultados recomendados.
        """
        if not self.resultados_categoria or not self.ultima_categoria:
            return "Primero dime si buscas una laptop, una tablet o un celular."

        inicio = self.indice_resultados
        fin = inicio + self.tamano_pagina
        siguientes = self.resultados_categoria[inicio:fin]

        if not siguientes:
            return f"Ya te mostré todas las opciones que tenía para {self.ultima_categoria}."

        self.indice_resultados = fin
        self.ultima_recomendacion = siguientes
        self.producto_actual = siguientes[0]
        self.ultimo_modo = "recomendado"
        self.ultima_lista_mostrada = siguientes
        self.ultimo_producto_mencionado = siguientes[0]
        self.ultimo_tema = "recomendacion"

        return self._formatear_lista_productos(
            siguientes,
            self.ultima_categoria,
            titulo=f"Aquí tienes otras opciones de tipo {self.ultima_categoria}:"
        )

    def no_recomendar(self, categoria=None, top_n=3):
        """
        Devuelve los productos menos recomendables de una categoría.
        """
        if not categoria:
            categoria = self.ultima_categoria

        if not categoria:
            return "Primero dime si buscas laptop, tablet o celular."

        categoria_dataset = self._categoria_dataset(categoria)
        df = self.productos.copy()
        filtrados = df[df["category"] == categoria_dataset]

        if filtrados.empty:
            return f"No encontré productos para {categoria}."

        # Ordena de peor a mejor rating
        filtrados = filtrados.sort_values(by=["rating", "price"], ascending=[True, True])
        resultados = filtrados.to_dict(orient="records")

        self._guardar_contexto_no_recomendacion(categoria, resultados)

        peores = self.resultados_no_recomendados[:top_n]
        self.indice_no_recomendados = len(peores)
        self.ultima_no_recomendacion = peores
        self.producto_actual = peores[0] if peores else None
        self.ultimo_modo = "no_recomendado"

        self.ultima_lista_mostrada = peores
        self.ultimo_producto_mencionado = peores[0] if peores else None
        self.ultimo_tema = "no_recomendacion"

        return self._formatear_lista_no_recomendados(peores, categoria)

    def dame_otros_no_recomendados(self):
        """
        Muestra más productos no recomendados.
        """
        if not self.resultados_no_recomendados or not self.ultima_categoria:
            return "Primero pídeme productos poco recomendables de una categoría."

        inicio = self.indice_no_recomendados
        fin = inicio + self.tamano_pagina
        siguientes = self.resultados_no_recomendados[inicio:fin]

        if not siguientes:
            return f"Ya te mostré todas las opciones menos recomendables que tenía para {self.ultima_categoria}."

        self.indice_no_recomendados = fin
        self.ultima_no_recomendacion = siguientes
        self.producto_actual = siguientes[0]
        self.ultimo_modo = "no_recomendado"
        self.ultima_lista_mostrada = siguientes
        self.ultimo_producto_mencionado = siguientes[0]
        self.ultimo_tema = "no_recomendacion"

        return self._formatear_lista_no_recomendados(
            siguientes,
            self.ultima_categoria,
            titulo=f"Aquí tienes otros productos poco recomendables en {self.ultima_categoria}:"
        )

    
    # BÚSQUEDA DE PRODUCTOS
    
    def _producto_exacto_global(self, texto_limpio):
        """
        Busca coincidencias exactas de producto en todo el dataset.
        Prioriza coincidencias completas del nombre.
        """
        texto = self.normalizar(texto_limpio)

        coincidencias_exactas = []
        for _, row in self.productos.iterrows():
            nombre = self.normalizar(row["name"])
            if nombre == texto or nombre in texto:
                coincidencias_exactas.append(row.to_dict())

        if coincidencias_exactas:
            coincidencias_exactas = sorted(
                coincidencias_exactas,
                key=lambda x: len(self.normalizar(x["name"])),
                reverse=True
            )
            return coincidencias_exactas[0]

        # También intenta coincidencia por marca + tokens completos del nombre
        for _, row in self.productos.iterrows():
            nombre = self.normalizar(row["name"])
            marca = self.normalizar(row["brand"])
            tokens_nombre = nombre.split()

            if marca in texto and all(tok in texto for tok in tokens_nombre):
                return row.to_dict()

        return None

    def _buscar_en_lista(self, texto_limpio, lista_productos):
        """
        Busca un producto dentro de una lista concreta.
        Soporta referencias como:
        - primero
        - segundo
        - tercero
        - otro / otra
        - coincidencia exacta
        - coincidencia aproximada
        """
        if not lista_productos:
            return None

        texto = self.normalizar(texto_limpio)

        if "primero" in texto or "primera" in texto:
            return lista_productos[0]
        if ("segundo" in texto or "segunda" in texto) and len(lista_productos) >= 2:
            return lista_productos[1]
        if ("tercero" in texto or "tercera" in texto) and len(lista_productos) >= 3:
            return lista_productos[2]

        if "otro" in texto or "otra" in texto:
            if self.ultimo_modo == "no_recomendado" and len(self.ultima_no_recomendacion) >= 2:
                return self.ultima_no_recomendacion[1]
            if self.ultimo_modo == "recomendado" and len(self.ultima_recomendacion) >= 2:
                return self.ultima_recomendacion[1]

        # Coincidencia exacta por nombre
        exactos = []
        for producto in lista_productos:
            nombre = self.normalizar(producto["name"])
            if nombre == texto or nombre in texto:
                exactos.append(producto)

        if exactos:
            exactos = sorted(exactos, key=lambda x: len(self.normalizar(x["name"])), reverse=True)
            return exactos[0]

        # Coincidencia por tokens completos
        for producto in lista_productos:
            nombre = self.normalizar(producto["name"])
            tokens_nombre = nombre.split()
            if all(tok in texto for tok in tokens_nombre):
                return producto

        # Coincidencia por marca + nombre
        for producto in lista_productos:
            nombre = self.normalizar(producto["name"])
            marca = self.normalizar(producto["brand"])
            tokens_nombre = nombre.split()

            if marca in texto and all(tok in texto for tok in tokens_nombre):
                return producto

        # Búsqueda aproximada como último recurso
        nombres = [self.normalizar(p["name"]) for p in lista_productos]
        texto_simple = re.sub(
            r"\b(review|precio|marca|rating|hablame|del|de|la|el|cual|es|su|sobre)\b",
            " ",
            texto
        )
        texto_simple = re.sub(r"\s+", " ", texto_simple).strip()

        posibles = [texto]
        if texto_simple:
            posibles.append(texto_simple)

        for candidato in posibles:
            match = get_close_matches(candidato, nombres, n=1, cutoff=0.80)
            if match:
                nombre_match = match[0]
                for producto in lista_productos:
                    if self.normalizar(producto["name"]) == nombre_match:
                        return producto

        return None

    def obtener_producto_por_referencia(self, texto_limpio):
        """
        Intenta encontrar a qué producto se refiere el usuario,
        usando varias fuentes de contexto.
        """
        producto_global = self._producto_exacto_global(texto_limpio)
        if producto_global:
            return producto_global

        producto = self._buscar_en_lista(texto_limpio, self.ultima_recomendacion)
        if producto:
            return producto

        producto = self._buscar_en_lista(texto_limpio, self.ultima_no_recomendacion)
        if producto:
            return producto

        producto = self._buscar_en_lista(texto_limpio, self.ultima_lista_mostrada)
        if producto:
            return producto

        texto = self.normalizar(texto_limpio)
        menciona_nombre = any(ch.isdigit() for ch in texto) or any(
            palabra in texto for palabra in [
                "dell", "asus", "lenovo", "samsung", "google",
                "realme", "huawei", "hp", "apple", "xiaomi", "acer"
            ]
        )

        if not menciona_nombre and self.ultimo_producto_mencionado:
            return self.ultimo_producto_mencionado

        if not menciona_nombre and self.producto_actual:
            return self.producto_actual

        return None

    def corregir_referencia_producto(self, texto_norm):
        """
        Corrige el producto de referencia cuando el usuario aclara
        de cuál producto está hablando.
        """
        texto = self.normalizar(texto_norm)

        for lista in [self.ultima_recomendacion, self.ultima_no_recomendacion, self.ultima_lista_mostrada]:
            for producto in lista:
                nombre = self.normalizar(producto["name"])
                if nombre in texto:
                    self.producto_actual = producto
                    self.ultimo_producto_mencionado = producto
                    return (
                        f"Entendido, entonces hablas de {producto['name']}. "
                        f"Puedes preguntarme su precio, marca, rating o si te la recomiendo."
                    )
        return None

    
    # EXPLICACIONES Y VALORACIONES
    
    def hablar_producto(self, producto):
        """
        Da una descripción general del producto seleccionado.
        """
        if not producto:
            return "Todavía no tengo un producto seleccionado. Pídeme una laptop, tablet o celular."

        self.producto_actual = producto
        self.ultimo_producto_mencionado = producto
        categoria_mostrar = "celular" if producto["category"] == "phone" else producto["category"]

        return (
            f"{producto['name']} es un producto de la marca {producto['brand']}. "
            f"Pertenece a la categoría {categoria_mostrar}, cuesta ${producto['price']}, "
            f"tiene rating {producto['rating']} y su review dice: '{producto['review']}'."
        )

    def explicar_por_que_si(self, texto_limpio):
        """
        Explica por qué un producto sí es recomendable.
        """
        producto = self.obtener_producto_por_referencia(texto_limpio)
        if not producto:
            return None

        self.producto_actual = producto
        self.ultimo_producto_mencionado = producto
        motivos = []

        if float(producto["rating"]) >= 4.8:
            motivos.append(f"tiene un rating excelente de {producto['rating']}")
        elif float(producto["rating"]) >= 4.5:
            motivos.append(f"tiene un rating muy bueno de {producto['rating']}")
        else:
            motivos.append(f"tiene una valoración aceptable de {producto['rating']}")

        review = self.normalizar(producto["review"])

        if "highly recommended" in review:
            motivos.append("la review lo describe como altamente recomendado")
        elif "excellent" in review:
            motivos.append("la review resalta su calidad")
        elif "battery" in review:
            motivos.append("la review destaca la batería y la fluidez")
        elif "great performance" in review:
            motivos.append("la review destaca su buen rendimiento")
        elif "very good for the price" in review:
            motivos.append("la review indica buena relación calidad-precio")
        else:
            motivos.append("está entre las mejores opciones del dataset")

        motivos.append(f"su precio de ${producto['price']} es competitivo frente a productos similares")

        if "juego" in self.normalizar(texto_limpio) or "gaming" in self.normalizar(texto_limpio):
            motivos.append("dentro del dataset aparece como una de las laptops más convenientes para ese uso")

        return f"Yo diría que {producto['name']} es una buena opción porque {', '.join(motivos)}."

    def explicar_por_que_no(self, texto_limpio):
        """
        Explica por qué un producto no es recomendable.
        """
        producto = self.obtener_producto_por_referencia(texto_limpio)

        if not producto and self.ultima_no_recomendacion:
            producto = self.ultima_no_recomendacion[0]

        if not producto:
            return None

        self.producto_actual = producto
        self.ultimo_producto_mencionado = producto
        motivos = []

        if float(producto["rating"]) <= 3.7:
            motivos.append(f"tiene una valoración baja de {producto['rating']}")
        else:
            motivos.append(
                f"queda por debajo de otras alternativas de su categoría con rating {producto['rating']}"
            )

        review = self.normalizar(producto["review"])

        if "not bad" in review:
            motivos.append("la opinión dice que no está mal, pero presenta algunos problemas")
        elif "could be better" in review:
            motivos.append("la review sugiere que podría ser mejor")
        elif "issue" in review or "issues" in review:
            motivos.append("la review menciona posibles problemas")
        elif "decent" in review:
            motivos.append("la review lo deja como una opción apenas aceptable")
        else:
            motivos.append("hay opciones mejor valoradas dentro del dataset")

        return (
            f"No te lo recomendaría mucho porque {', '.join(motivos)}. "
            f"En esa misma categoría encontré alternativas con mejor equilibrio entre rating y precio."
        )

    def explicar_por_que_son_las_peores(self):
        """
        Explica en plural por qué varias opciones son poco recomendables.
        """
        if not self.ultima_no_recomendacion:
            return None

        razones = []
        for producto in self.ultima_no_recomendacion[:3]:
            razones.append(
                f"{producto['name']} tiene rating {producto['rating']} y una review que no destaca demasiado"
            )

        return (
            "Las considero de las menos recomendables porque, comparadas con otras opciones de la misma categoría, "
            + "; ".join(razones)
            + ". En general, hay alternativas con mejor equilibrio entre calidad, precio y valoración."
        )

    def valorar_producto_actual(self):
        """
        Da una valoración resumida del producto actual.
        """
        producto = self.producto_actual
        if not producto:
            return None

        rating = float(producto["rating"])
        review = self.normalizar(producto["review"])

        comentario = []

        if rating >= 4.8:
            comentario.append("es una opción excelente")
        elif rating >= 4.5:
            comentario.append("es una opción bastante sólida")
        elif rating >= 4.0:
            comentario.append("es una opción aceptable")
        else:
            comentario.append("no destaca demasiado frente a otras alternativas")

        if "battery" in review:
            comentario.append("su review resalta batería y fluidez")
        elif "great performance" in review:
            comentario.append("su punto fuerte parece ser el rendimiento")
        elif "excellent" in review:
            comentario.append("la opinión destaca calidad de construcción")
        elif "very good for the price" in review:
            comentario.append("ofrece buena relación calidad-precio")
        elif "could be better" in review:
            comentario.append("aunque la review sugiere que podría mejorar")

        comentario.append(f"además cuesta ${producto['price']}")

        return f"{producto['name']} me parece que {', '.join(comentario)}."

    def responder_comparacion_para_trabajo(self):
        """
        Responde con criterios útiles de comparación para trabajo.
        """
        if self.ultima_recomendacion:
            nombres = ", ".join([p["name"] for p in self.ultima_recomendacion[:3]])
            return (
                f"Si piensas usar una tablet para trabajar, lo más importante es comparar "
                f"rendimiento, tamaño de pantalla, batería, comodidad y precio. "
                f"De las opciones recientes ({nombres}), puedo compararte cuál conviene más "
                f"para productividad, videollamadas, documentos o multitarea."
            )

        return (
            "Si buscas un producto para trabajar, conviene comparar rendimiento, batería, pantalla, "
            "comodidad de uso y precio. Dime si quieres laptop, tablet o celular y te digo cuál conviene más."
        )

    def comparar_con_otro_para_juegos(self):
        """
        Compara el producto actual con otra laptop pensada para juegos.
        """
        if not self.producto_actual:
            return "Primero selecciona una laptop y luego puedo compararla con otra opción para juegos."

        categoria = "laptop"
        categoria_dataset = self._categoria_dataset(categoria)
        df = self.productos.copy()
        filtrados = df[df["category"] == categoria_dataset]

        if filtrados.empty:
            return "No encontré laptops para comparar."

        productos = filtrados.sort_values(by=["rating", "price"], ascending=[False, True]).to_dict(orient="records")

        actual = self.producto_actual
        alternativa = None

        for p in productos:
            if self.normalizar(p["name"]) != self.normalizar(actual["name"]):
                alternativa = p
                break

        if not alternativa:
            return f"No encontré otra laptop para comparar con {actual['name']}."

        return (
            f"Si comparamos {actual['name']} con {alternativa['name']}, "
            f"ambas son opciones interesantes. "
            f"{actual['name']} tiene rating {actual['rating']} y precio de ${actual['price']}, "
            f"mientras que {alternativa['name']} tiene rating {alternativa['rating']} "
            f"y cuesta ${alternativa['price']}. "
            f"Si quieres, puedo decirte cuál elegiría específicamente para juegos."
        )

    
    # DETALLE DE PRODUCTO
    
    def responder_detalle_producto(self, texto_limpio):
        """
        Responde detalles específicos de un producto:
        - descripción general
        - precio
        - marca
        - rating
        - review
        """
        producto = self.obtener_producto_por_referencia(texto_limpio)

        if not producto:
            return None

        self.producto_actual = producto
        self.ultimo_producto_mencionado = producto

        if any(x in texto_limpio for x in [
            "hablame", "dime del", "informacion",
            "detalle", "cuentame", "sobre"
        ]):
            return self.hablar_producto(producto)

        if "precio" in texto_limpio:
            return f"El precio de {producto['name']} es ${producto['price']}."

        if "marca" in texto_limpio:
            return f"La marca de {producto['name']} es {producto['brand']}."

        if any(x in texto_limpio for x in ["rating", "calificacion", "puntuacion"]):
            return f"El rating de {producto['name']} es {producto['rating']}."

        if any(x in texto_limpio for x in ["review", "opinion", "comentario"]):
            return f"La review asociada a {producto['name']} dice: '{producto['review']}'."

        if any(x in texto_limpio for x in ["me gusta", "quiero ese", "quiero este", "lo quiero"]):
            if float(producto["rating"]) >= 4.5:
                return (
                    f"Buena elección. {producto['name']} está muy bien valorado, "
                    f"con rating {producto['rating']} y precio de ${producto['price']}."
                )
            else:
                return (
                    f"Puedes elegir {producto['name']}, pero su rating es {producto['rating']} "
                    f"y hay alternativas mejor valoradas en el dataset."
                )

        return None

    def explicar_sistema(self):
        """
        Explica de forma breve qué hace el bot.
        """
        return (
            "Soy un bot de recomendaciones de productos. "
            "Puedo sugerirte laptops, tablets y celulares, mostrar opciones buenas o menos recomendables, "
            "y darte detalles de cada producto."
        )

    
    # RESPUESTA PRINCIPAL
   
    def responder(self, texto):
        """
        Método principal del bot.
        Aquí se decide cómo responder según:
        - saludos
        - agradecimientos
        - comparaciones
        - detalles de producto
        - reglas por categoría
        - dataset
        - modelo supervisado
        """
        texto_limpio = limpiar_texto(texto)
        texto_norm = self.normalizar(texto_limpio)

        if not texto_norm:
            return "Escribe una consulta, por ejemplo: laptop, tablet o celular."

        # Saludos
        if texto_norm in ["hola", "buenas", "hello", "hi", "klk"]:
            return self.responder_con_estilo(
                texto,
                "Hola. Puedo ayudarte a encontrar laptops, tablets y celulares. ¿Qué estás buscando?"
            )

        # Preguntas de cortesía
        if texto_norm in ["como estas", "como te va", "como andas"]:
            return self.responder_con_estilo(
                texto,
                "Estoy bien. ¿Buscas una laptop, una tablet o un celular?"
            )

        # Agradecimientos
        if self.es_agradecimiento(texto_norm):
            return self.responder_con_estilo(
                texto,
                "De nada. Si quieres, puedo mostrarte más opciones o comparar productos."
            )

        # Reacciones positivas
        if self.es_reaccion_positiva(texto_norm):
            return self.responder_con_estilo(texto, self.responder_reaccion_positiva())

        # Confirmaciones simples
        if self.es_confirmacion_simple(texto_norm):
            return self.responder_con_estilo(
                texto,
                "Perfecto. Puedes pedirme laptops, tablets o celulares."
            )

        # Preguntas sobre el sistema
        if self.es_pregunta_identidad(texto_norm):
            return self.responder_con_estilo(texto, self.explicar_sistema())

        if self.pregunta_catalogo(texto_norm):
            return self.responder_con_estilo(
                texto,
                "Puedo recomendarte laptops, tablets y celulares. Dime cuál te interesa y te muestro opciones."
            )

        if any(x in texto_norm for x in ["como funciona", "explica el sistema", "explica resultados"]):
            return self.responder_con_estilo(texto, self.explicar_sistema())

        # Comparación para juegos
        if "compar" in texto_norm and self.quiere_recomendacion_para_jugar(texto_norm):
            respuesta = self.comparar_con_otro_para_juegos()
            return self.responder_con_estilo(texto, respuesta, usar_llm=False)

        # Comparación para trabajo
        if self.quiere_comparacion_para_trabajo(texto_norm):
            return self.responder_con_estilo(texto, self.responder_comparacion_para_trabajo())

        # Corrección de referencia
        if any(x in texto_norm for x in ["no hablo de", "hablo de", "me refiero a"]):
            correccion = self.corregir_referencia_producto(texto_norm)
            if correccion:
                return self.responder_con_estilo(texto, correccion)

        # Detalle de producto
        detalle = self.responder_detalle_producto(texto_norm)
        if detalle:
            usar_llm = not self.es_respuesta_con_datos_sensibles(detalle)
            return self.responder_con_estilo(texto, detalle, usar_llm=usar_llm)

        # Detecta categoría
        categoria_texto = self.detectar_categoria_en_texto(texto_norm)

        # Más no recomendados
        if self.quiere_mas_no_recomendados(texto_norm):
            return self.responder_con_estilo(texto, self.dame_otros_no_recomendados())

        # Explicación en plural de peores
        if any(x in texto_norm for x in [
            "por que son las peores", "por que son las peor", "por que esas son las peores",
            "por que son malas", "por que esas no"
        ]):
            explicacion = self.explicar_por_que_son_las_peores()
            if explicacion:
                return self.responder_con_estilo(texto, explicacion, usar_llm=False)

        # Mejor producto
        if self.pregunta_mejor_producto(texto_norm):
            categoria = categoria_texto or self.ultima_categoria or ("laptop" if self.quiere_recomendacion_para_jugar(texto_norm) else None)
            if categoria:
                respuesta = self.mejor_producto(categoria)
                return self.responder_con_estilo(texto, respuesta, usar_llm=False)
            return self.responder_con_estilo(
                texto,
                "Puedo decirte cuál es el mejor producto dentro de una categoría. Por ejemplo: mejor laptop, mejor tablet o mejor celular."
            )

        # Peor producto
        if self.pregunta_peor_producto(texto_norm):
            categoria = categoria_texto or self.ultima_categoria
            if categoria:
                return self.responder_con_estilo(texto, self.no_recomendar(categoria), usar_llm=False)
            return self.responder_con_estilo(
                texto,
                "Puedo decirte cuál es el menos recomendable dentro de una categoría. Por ejemplo: peor laptop, peor tablet o peor celular."
            )

        # Valoración del producto actual
        if any(x in texto_norm for x in [
            "que tan bueno es", "que tan buena es",
            "es bueno", "es buena"
        ]):
            valoracion = self.valorar_producto_actual()
            if valoracion:
                return self.responder_con_estilo(texto, valoracion, usar_llm=False)

        # Explicación de por qué no
        if self.es_pregunta_por_que_no(texto_norm):
            explicacion = self.explicar_por_que_no(texto_norm)
            if explicacion:
                return self.responder_con_estilo(texto, explicacion, usar_llm=False)

        # Explicación de por qué sí
        if self.es_pregunta_por_que_si(texto_norm):
            explicacion = self.explicar_por_que_si(texto_norm)
            if explicacion:
                return self.responder_con_estilo(texto, explicacion, usar_llm=False)

        # Productos no recomendables
        if ("no" in texto_norm and "recom" in texto_norm) or any(x in texto_norm for x in [
            "peor producto", "cual no me recomiendas",
            "cual no recomiendas", "que no me recomiendas",
            "menos recomendable"
        ]):
            categoria = categoria_texto or self.ultima_categoria
            return self.responder_con_estilo(texto, self.no_recomendar(categoria), usar_llm=False)

        # Recomendación para jugar
        if self.quiere_recomendacion_para_jugar(texto_norm):
            return self.responder_con_estilo(texto, self.recomendar("laptop"), usar_llm=False)

        # Ver categoría con intención explícita
        if categoria_texto and self.quiere_ver_categoria(texto_norm):
            return self.responder_con_estilo(texto, self.recomendar(categoria_texto), usar_llm=False)

        # Si solo mencionó la categoría
        if categoria_texto and texto_norm in [
            "laptop", "laptops", "tablet", "tablets", "tableta", "tabletas",
            "celular", "celulares", "phone", "phones", "latop", "laptos",
            "portatil", "portatiles", "table"
        ]:
            return self.responder_con_estilo(texto, self.recomendar(categoria_texto), usar_llm=False)

        # Más resultados o más información
        if self.quiere_mas_resultados(texto_norm):
            if "informacion" in texto_norm and self.producto_actual:
                detalle = self.hablar_producto(self.producto_actual)
                return self.responder_con_estilo(texto, detalle, usar_llm=False)

            if categoria_texto and self.ultima_categoria != categoria_texto:
                return self.responder_con_estilo(texto, self.recomendar(categoria_texto), usar_llm=False)
            return self.responder_con_estilo(texto, self.dame_otros(), usar_llm=False)

        # Solicitudes generales
        if any(x in texto_norm for x in [
            "necesito un producto", "recomiendame un producto",
            "recomiendame algo", "quiero un producto"
        ]):
            return self.responder_con_estilo(
                texto,
                "Claro. ¿Buscas una laptop, una tablet o un celular?"
            )

        # Si detectó categoría directamente
        if categoria_texto:
            return self.responder_con_estilo(texto, self.recomendar(categoria_texto), usar_llm=False)

        # Busca respuesta prefijada en dataset
        respuesta_dataset = self.buscar_respuesta_dataset(texto_norm)
        if respuesta_dataset:
            usar_llm = not self.es_respuesta_con_datos_sensibles(respuesta_dataset)
            return self.responder_con_estilo(texto, respuesta_dataset, usar_llm=usar_llm)

        # Usa el modelo supervisado como último recurso
        intencion = self.modelo.predecir(texto_norm)

        if intencion == "saludo":
            return self.responder_con_estilo(
                texto,
                "Hola. ¿Buscas una laptop, una tablet o un celular?"
            )
        if intencion == "laptop":
            return self.responder_con_estilo(texto, self.recomendar("laptop"), usar_llm=False)
        if intencion == "tablet":
            return self.responder_con_estilo(texto, self.recomendar("tablet"), usar_llm=False)
        if intencion == "celular":
            return self.responder_con_estilo(texto, self.recomendar("celular"), usar_llm=False)

        # Respuesta por defecto
        return self.responder_con_estilo(
            texto,
            "No entendí bien tu consulta. Puedo ayudarte con laptops, tablets o celulares."
        )