
# IMPORTACIÓN DE DATOS


# Importamos la función que carga el dataset de productos
from chat_dataset import cargar_productos


# FUNCIÓN DE ESTADÍSTICAS


def mostrar_estadisticas():
    """
    Muestra estadísticas generales del dataset de productos.

    Incluye:
    - Total de productos
    - Precio promedio
    - Rating promedio
    - Distribución por categoría
    - Marcas más frecuentes
    """

    # Cargamos el dataset de productos
    df = cargar_productos()

   
    # MÉTRICAS GENERALES
    

    # Total de productos
    print("Total productos:", len(df))

    # Precio promedio (redondeado a 2 decimales)
    print("Precio promedio:", round(df["price"].mean(), 2))

    # Rating promedio (redondeado a 2 decimales)
    print("Rating promedio:", round(df["rating"].mean(), 2))

    
    # DISTRIBUCIÓN POR CATEGORÍA
  

    print("\nProductos por categoría:")
    print(df["category"].value_counts())

    
    # TOP MARCAS
  

    print("\nProductos por marca:")
    
    # Mostramos las 10 marcas más frecuentes
    print(df["brand"].value_counts().head(10))



# PUNTO DE ENTRADA


# Ejecuta la función solo si este archivo se corre directamente
if __name__ == "__main__":
    mostrar_estadisticas()