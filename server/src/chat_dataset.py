from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR.parent
DATA_DIR = SERVER_DIR / "data"


def cargar_conversaciones():
    ruta = DATA_DIR / "conversaciones_500.csv"
    df = pd.read_csv(ruta)
    df.columns = ["input", "output"]
    df["input"] = df["input"].astype(str)
    df["output"] = df["output"].astype(str)
    return df


def cargar_productos():
    ruta = DATA_DIR / "amazon_1000_products.csv"
    df = pd.read_csv(ruta)
    df.columns = [c.lower().strip() for c in df.columns]

    df["category"] = df["category"].astype(str).str.lower().str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df.dropna(subset=["name", "brand", "category", "price", "rating", "review"])
    return df