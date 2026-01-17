from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

def load_orders(filename="Sample_Superstore.csv"):
    try:
        return pd.read_csv(RAW_DIR / filename)
    except FileNotFoundError:
        raise FileNotFoundError("CSV file not found in data/raw/")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
