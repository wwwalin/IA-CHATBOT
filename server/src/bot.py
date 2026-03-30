from difflib import get_close_matches
import re
import unicodedata

from chat_dataset import cargar_conversaciones, cargar_productos
from preprocessing import limpiar_texto
from model import ModeloIntencion
from llm_client import mejorar_respuesta


class ChatBot:
    def __init__(self):
        self.conversaciones = cargar_conversaciones()
        self.productos = cargar_productos()
        self.modelo = ModeloIntencion()

        self.ultima_recomendacion = []
        self.ultima_no_recomendacion = []
        self.producto_actual = None
        self.ultima_categoria = None

        self.resultados_categoria = []
        self.indice_resultados = 0

        self.resultados_no_recomendados = []
        self.indice_no_recomendados = 0

        self.tamano_pagina = 3
        self.ultimo_modo = None  # recomendado | no_recomendado

    # =========================
    # UTILIDADES
    # =========================
    def normalizar(self, texto):
        texto = str(texto).lower().strip()
        texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")

        # corrijo algunos typos frecuentes
        texto = texto.replace("dle", "del")
        texto = texto.replace("qeu", "que")

        # separo letras y números: note13 -> note 13
        texto = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", texto)
        texto = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", texto)

        texto = re.sub(r"[^\w\s]", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    def _es_respuesta_estructurada(self, respuesta):
        if not respuesta:
            return False

        lineas = [l.strip() for l in respuesta.splitlines() if l.strip()]
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

    def responder_con_estilo(self, mensaje_usuario, respuesta_base):
        if not respuesta_base:
            return respuesta_base

        respuesta_base = str(respuesta_base).strip()

        if len(respuesta_base) < 40:
            return respuesta_base

        if self._es_respuesta_estructurada(respuesta_base):
            return respuesta_base

        return mejorar_respuesta(mensaje_usuario, respuesta_base)

    def buscar_respuesta_dataset(self, texto_limpio):
        match = self.conversaciones[
            self.conversaciones["input"].astype(str).str.lower().str.strip() == texto_limpio
        ]
        if not match.empty:
            return match.iloc[0]["output"]
        return None

    def detectar_categoria_en_texto(self, texto_limpio):
        texto = self.normalizar(texto_limpio)

        if any(x in texto for x in [
            "laptop", "laptops", "notebook", "notebooks",
            "latop", "laptos", "portatil", "portatiles",
            "computadora portatil", "computadoras portatiles"
        ]):
            return "laptop"

        if any(x in texto for x in ["tablet", "tablets", "tableta", "tabletas"]):
            return "tablet"

        if any(x in texto for x in [
            "celular", "celulares", "telefono", "telefonos",
            "phone", "phones", "movil", "moviles"
        ]):
            return "celular"

        return None

    def es_agradecimiento(self, texto_norm):
        patrones = [
            "gracias", "muchas gracias", "ok gracias",
            "gracias por la recomendacion", "te lo agradezco",
            "thanks", "thank you"
        ]
        return any(p in texto_norm for p in patrones)

    def es_confirmacion_simple(self, texto_norm):
        patrones = [
            "ok", "okay", "vale", "bien", "perfecto",
            "esta bien", "dale", "entiendo"
        ]
        return texto_norm in patrones

    def es_pregunta_identidad(self, texto_norm):
        patrones = [
            "que eres", "quien eres", "que haces",
            "para que sirves", "eres un bot",
            "eres una ia", "eres una inteligencia artificial"
        ]
        return any(p in texto_norm for p in patrones)

    def es_pregunta_por_que_no(self, texto_norm):
        variantes = [
            "por que no", "porque no", "por que",
            "por que no el segundo", "por que no el primero"
        ]
        return any(v in texto_norm for v in variantes)

    def es_pregunta_por_que_si(self, texto_norm):
        variantes = [
            "por que me lo recomiendas",
            "por que lo recomiendas",
            "por que es bueno",
            "que tan bueno es",
            "que hace que sea bueno",
            "por que si"
        ]
        return any(v in texto_norm for v in variantes)

    def quiere_mas_resultados(self, texto_norm):
        patrones = [
            "dame otros", "dame otras", "dame otro", "dame otra",
            "dame otros productos", "dame otras opciones",
            "dame mas", "muestrame mas", "quiero ver mas",
            "otros productos", "otras opciones", "otras laptops",
            "otros celulares", "otras tablets"
        ]
        return any(p in texto_norm for p in patrones)

    def quiere_mas_no_recomendados(self, texto_norm):
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
        patrones = [
            "cual es la mejor", "cual es el mejor",
            "mejor laptop", "mejor celular", "mejor tablet",
            "cual me recomiendas mas", "la mejor opcion",
            "mejor producto"
        ]
        return any(p in texto_norm for p in patrones)

    def pregunta_peor_producto(self, texto_norm):
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
        patrones = [
            "cuales tienes", "que tienes",
            "que productos tienes",
            "que me puedes recomendar",
            "recomiendame productos"
        ]
        return any(p in texto_norm for p in patrones)

    def quiere_ver_categoria(self, texto_norm):
        patrones = [
            "hablame de", "quiero ver", "muestrame",
            "dame", "ensename", "recomiendame",
            "busco", "necesito", "que tienes",
            "cuales tienes", "que productos tienes",
            "recomiendame productos"
        ]
        return any(p in texto_norm for p in patrones)

    def _categoria_dataset(self, categoria):
        return "phone" if categoria == "celular" else categoria

    def _guardar_contexto_recomendacion(self, categoria, productos):
        self.ultima_categoria = categoria
        self.resultados_categoria = productos
        self.indice_resultados = 0
        self.ultima_recomendacion = []
        self.producto_actual = None
        self.ultimo_modo = None

    def _guardar_contexto_no_recomendacion(self, categoria, productos):
        self.ultima_categoria = categoria
        self.resultados_no_recomendados = productos
        self.indice_no_recomendados = 0
        self.ultima_no_recomendacion = []
        self.producto_actual = None
        self.ultimo_modo = None

    def _formatear_lista_productos(self, productos, categoria, titulo=None):
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

    # =========================
    # RECOMENDACIONES
    # =========================
    def recomendar(self, categoria, top_n=3):
        categoria = categoria.lower()
        categoria_dataset = self._categoria_dataset(categoria)

        df = self.productos.copy()
        filtrados = df[df["category"] == categoria_dataset]

        if filtrados.empty:
            return f"No encontré productos para {categoria}."

        filtrados = filtrados.sort_values(by=["rating", "price"], ascending=[False, True])
        resultados = filtrados.to_dict(orient="records")

        self._guardar_contexto_recomendacion(categoria, resultados)

        primeros = self.resultados_categoria[:top_n]
        self.indice_resultados = len(primeros)
        self.ultima_recomendacion = primeros
        self.producto_actual = primeros[0] if primeros else None
        self.ultimo_modo = "recomendado"

        return self._formatear_lista_productos(primeros, categoria)

    def mejor_producto(self, categoria):
        categoria_dataset = self._categoria_dataset(categoria)
        df = self.productos.copy()
        filtrados = df[df["category"] == categoria_dataset]

        if filtrados.empty:
            return f"No encontré productos para {categoria}."

        filtrados = filtrados.sort_values(by=["rating", "price"], ascending=[False, True])
        mejor = filtrados.iloc[0].to_dict()

        self.ultima_categoria = categoria
        self.producto_actual = mejor
        self.ultima_recomendacion = [mejor]
        self.ultimo_modo = "recomendado"

        return (
            f"La mejor opción en {categoria} es {mejor['name']} de {mejor['brand']}. "
            f"Tiene rating {mejor['rating']}, cuesta ${mejor['price']} y la review dice: "
            f"'{mejor['review']}'."
        )

    def dame_otros(self):
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

        return self._formatear_lista_productos(
            siguientes,
            self.ultima_categoria,
            titulo=f"Aquí tienes otras opciones de tipo {self.ultima_categoria}:"
        )

    def no_recomendar(self, categoria=None, top_n=3):
        if not categoria:
            categoria = self.ultima_categoria

        if not categoria:
            return "Primero dime si buscas laptop, tablet o celular."

        categoria_dataset = self._categoria_dataset(categoria)

        df = self.productos.copy()
        filtrados = df[df["category"] == categoria_dataset]

        if filtrados.empty:
            return f"No encontré productos para {categoria}."

        filtrados = filtrados.sort_values(by=["rating", "price"], ascending=[True, True])
        resultados = filtrados.to_dict(orient="records")

        self._guardar_contexto_no_recomendacion(categoria, resultados)

        peores = self.resultados_no_recomendados[:top_n]
        self.indice_no_recomendados = len(peores)
        self.ultima_no_recomendacion = peores
        self.producto_actual = peores[0] if peores else None
        self.ultimo_modo = "no_recomendado"

        return self._formatear_lista_no_recomendados(peores, categoria)

    def dame_otros_no_recomendados(self):
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

        return self._formatear_lista_no_recomendados(
            siguientes,
            self.ultima_categoria,
            titulo=f"Aquí tienes otros productos poco recomendables en {self.ultima_categoria}:"
        )

    # =========================
    # BÚSQUEDA DE PRODUCTO
    # =========================
    def _producto_exacto_global(self, texto_limpio):
        texto = self.normalizar(texto_limpio)

        for _, row in self.productos.iterrows():
            nombre = self.normalizar(row["name"])
            marca = self.normalizar(row["brand"])

            if nombre in texto:
                return row.to_dict()

            nombre_tokens = nombre.split()
            if marca in texto and all(tok in texto for tok in nombre_tokens[1:]):
                return row.to_dict()

        return None

    def _buscar_en_lista(self, texto_limpio, lista_productos):
        if not lista_productos:
            return None

        texto = self.normalizar(texto_limpio)

        if "primero" in texto or "primera" in texto:
            return lista_productos[0]
        if ("segundo" in texto or "segunda" in texto) and len(lista_productos) >= 2:
            return lista_productos[1]
        if ("tercero" in texto or "tercera" in texto) and len(lista_productos) >= 3:
            return lista_productos[2]

        for producto in lista_productos:
            nombre = self.normalizar(producto["name"])
            if nombre in texto:
                return producto

        for producto in lista_productos:
            nombre = self.normalizar(producto["name"])
            marca = self.normalizar(producto["brand"])
            tokens_nombre = nombre.split()

            if marca in texto and all(tok in texto for tok in tokens_nombre[1:]):
                return producto

        nombres = [self.normalizar(p["name"]) for p in lista_productos]
        posibles = [texto]

        # intento extraer solo parte útil
        texto_simple = re.sub(r"\b(review|precio|marca|rating|hablame|del|de|la|el|cual|es)\b", " ", texto)
        texto_simple = re.sub(r"\s+", " ", texto_simple).strip()
        if texto_simple:
            posibles.append(texto_simple)

        for candidato in posibles:
            match = get_close_matches(candidato, nombres, n=1, cutoff=0.55)
            if match:
                nombre_match = match[0]
                for producto in lista_productos:
                    if self.normalizar(producto["name"]) == nombre_match:
                        return producto

        return None

    def obtener_producto_por_referencia(self, texto_limpio):
        producto_global = self._producto_exacto_global(texto_limpio)
        if producto_global:
            return producto_global

        producto = self._buscar_en_lista(texto_limpio, self.ultima_recomendacion)
        if producto:
            return producto

        producto = self._buscar_en_lista(texto_limpio, self.ultima_no_recomendacion)
        if producto:
            return producto

        texto = self.normalizar(texto_limpio)
        menciona_nombre = any(ch.isdigit() for ch in texto) or any(
            palabra in texto for palabra in [
                "dell", "asus", "lenovo", "samsung", "google",
                "realme", "huawei", "hp", "apple", "xiaomi", "acer"
            ]
        )

        if not menciona_nombre and self.producto_actual:
            return self.producto_actual

        return None

    # =========================
    # EXPLICACIONES
    # =========================
    def hablar_producto(self, producto):
        if not producto:
            return "Todavía no tengo un producto seleccionado. Pídeme una laptop, tablet o celular."

        self.producto_actual = producto
        categoria_mostrar = "celular" if producto["category"] == "phone" else producto["category"]

        return (
            f"{producto['name']} es un producto de la marca {producto['brand']}. "
            f"Pertenece a la categoría {categoria_mostrar}, cuesta ${producto['price']}, "
            f"tiene rating {producto['rating']} y su review dice: '{producto['review']}'."
        )

    def explicar_por_que_si(self, texto_limpio):
        producto = self.obtener_producto_por_referencia(texto_limpio)
        if not producto:
            return None

        self.producto_actual = producto
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

        return f"Yo diría que {producto['name']} es una buena opción porque {', '.join(motivos)}."

    def explicar_por_que_no(self, texto_limpio):
        producto = self.obtener_producto_por_referencia(texto_limpio)

        if not producto and self.ultima_no_recomendacion:
            producto = self.ultima_no_recomendacion[0]

        if not producto:
            return None

        self.producto_actual = producto
        motivos = []

        if float(producto["rating"]) <= 3.7:
            motivos.append(f"tiene una valoración baja de {producto['rating']}")
        else:
            motivos.append(f"queda por debajo de otras alternativas de su categoría con rating {producto['rating']}")

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

    def valorar_producto_actual(self):
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

    # =========================
    # DETALLE DE PRODUCTO
    # =========================
    def responder_detalle_producto(self, texto_limpio):
        producto = self.obtener_producto_por_referencia(texto_limpio)

        if not producto:
            return None

        self.producto_actual = producto

        if any(x in texto_limpio for x in [
            "hablame", "dime del", "informacion",
            "detalle", "cuentame"
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
        return (
            "Soy un bot de recomendaciones de productos. "
            "Puedo sugerirte laptops, tablets y celulares, mostrar opciones buenas o menos recomendables, "
            "y darte detalles de cada producto."
        )

    # =========================
    # RESPUESTA PRINCIPAL
    # =========================
    def responder(self, texto):
        texto_limpio = limpiar_texto(texto)
        texto_norm = self.normalizar(texto_limpio)

        if not texto_norm:
            return "Escribe una consulta, por ejemplo: laptop, tablet o celular."

        if texto_norm in ["hola", "buenas", "hello", "hi"]:
            return self.responder_con_estilo(
                texto,
                "Hola. Puedo ayudarte a encontrar laptops, tablets y celulares. ¿Qué estás buscando?"
            )

        if texto_norm in ["como estas", "como te va", "como andas"]:
            return self.responder_con_estilo(
                texto,
                "Estoy bien. ¿Buscas una laptop, una tablet o un celular?"
            )

        if self.es_agradecimiento(texto_norm):
            return self.responder_con_estilo(
                texto,
                "De nada. Si quieres, puedo mostrarte más opciones o comparar productos."
            )

        if self.es_confirmacion_simple(texto_norm):
            return self.responder_con_estilo(
                texto,
                "Perfecto. Puedes pedirme laptops, tablets o celulares."
            )

        if self.es_pregunta_identidad(texto_norm):
            return self.responder_con_estilo(texto, self.explicar_sistema())

        if self.pregunta_catalogo(texto_norm):
            return self.responder_con_estilo(
                texto,
                "Puedo recomendarte laptops, tablets y celulares. Dime cuál te interesa y te muestro opciones."
            )

        if any(x in texto_norm for x in ["como funciona", "explica el sistema", "explica resultados"]):
            return self.responder_con_estilo(texto, self.explicar_sistema())

        # PRIMERO: si menciona un producto o pide detalle, lo resuelvo antes que categoría/intención
        detalle = self.responder_detalle_producto(texto_norm)
        if detalle:
            return self.responder_con_estilo(texto, detalle)

        categoria_texto = self.detectar_categoria_en_texto(texto_norm)

        if self.quiere_mas_no_recomendados(texto_norm):
            return self.responder_con_estilo(texto, self.dame_otros_no_recomendados())

        if self.pregunta_mejor_producto(texto_norm):
            categoria = categoria_texto or self.ultima_categoria
            if categoria:
                return self.responder_con_estilo(texto, self.mejor_producto(categoria))
            return self.responder_con_estilo(
                texto,
                "Puedo decirte cuál es el mejor producto dentro de una categoría. Por ejemplo: mejor laptop, mejor tablet o mejor celular."
            )

        if self.pregunta_peor_producto(texto_norm):
            categoria = categoria_texto or self.ultima_categoria
            if categoria:
                return self.responder_con_estilo(texto, self.no_recomendar(categoria))
            return self.responder_con_estilo(
                texto,
                "Puedo decirte cuál es el menos recomendable dentro de una categoría. Por ejemplo: peor laptop, peor tablet o peor celular."
            )

        if any(x in texto_norm for x in [
            "que tan bueno es", "que tan buena es",
            "es bueno", "es buena"
        ]):
            valoracion = self.valorar_producto_actual()
            if valoracion:
                return self.responder_con_estilo(texto, valoracion)

        if self.es_pregunta_por_que_no(texto_norm):
            explicacion = self.explicar_por_que_no(texto_norm)
            if explicacion:
                return self.responder_con_estilo(texto, explicacion)

        if self.es_pregunta_por_que_si(texto_norm):
            explicacion = self.explicar_por_que_si(texto_norm)
            if explicacion:
                return self.responder_con_estilo(texto, explicacion)

        if ("no" in texto_norm and "recom" in texto_norm) or any(x in texto_norm for x in [
            "peor producto", "cual no me recomiendas",
            "cual no recomiendas", "que no me recomiendas",
            "menos recomendable"
        ]):
            categoria = categoria_texto or self.ultima_categoria
            return self.responder_con_estilo(texto, self.no_recomendar(categoria))

        if categoria_texto and self.quiere_ver_categoria(texto_norm):
            return self.responder_con_estilo(texto, self.recomendar(categoria_texto))

        if categoria_texto and texto_norm in [
            "laptop", "laptops", "tablet", "tablets", "tableta", "tabletas",
            "celular", "celulares", "phone", "phones", "latop", "laptos",
            "portatil", "portatiles"
        ]:
            return self.responder_con_estilo(texto, self.recomendar(categoria_texto))

        if self.quiere_mas_resultados(texto_norm):
            if categoria_texto and self.ultima_categoria != categoria_texto:
                return self.responder_con_estilo(texto, self.recomendar(categoria_texto))
            return self.responder_con_estilo(texto, self.dame_otros())

        if any(x in texto_norm for x in [
            "necesito un producto", "recomiendame un producto",
            "recomiendame algo", "quiero un producto"
        ]):
            return self.responder_con_estilo(
                texto,
                "Claro. ¿Buscas una laptop, una tablet o un celular?"
            )

        if categoria_texto:
            return self.responder_con_estilo(texto, self.recomendar(categoria_texto))

        respuesta_dataset = self.buscar_respuesta_dataset(texto_norm)
        if respuesta_dataset:
            return self.responder_con_estilo(texto, respuesta_dataset)

        intencion = self.modelo.predecir(texto_norm)

        if intencion == "saludo":
            return self.responder_con_estilo(
                texto,
                "Hola. ¿Buscas una laptop, una tablet o un celular?"
            )
        if intencion == "laptop":
            return self.responder_con_estilo(texto, self.recomendar("laptop"))
        if intencion == "tablet":
            return self.responder_con_estilo(texto, self.recomendar("tablet"))
        if intencion == "celular":
            return self.responder_con_estilo(texto, self.recomendar("celular"))

        return self.responder_con_estilo(
            texto,
            "No entendí bien tu consulta. Puedo ayudarte con laptops, tablets o celulares."
        )