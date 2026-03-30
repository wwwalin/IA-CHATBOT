from chat_dataset import cargar_productos


def mostrar_estadisticas():
    df = cargar_productos()

    print("Total productos:", len(df))
    print("Precio promedio:", round(df["price"].mean(), 2))
    print("Rating promedio:", round(df["rating"].mean(), 2))

    print("\nProductos por categoría:")
    print(df["category"].value_counts())

    print("\nProductos por marca:")
    print(df["brand"].value_counts().head(10))


if __name__ == "__main__":
    mostrar_estadisticas()