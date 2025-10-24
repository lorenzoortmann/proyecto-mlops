# src/data_prep.py
import pandas as pd
from pathlib import Path
import os
import sys

# Define las rutas
RAW_DATA_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/clean_telco.csv")

def main():
    print("---Etapa 2: data_prep.py ---")

    # Cargar datos
    if not RAW_DATA_PATH.exists():
        print(f"Error: No se encontró el archivo de datos en {RAW_DATA_PATH}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Cargando datos crudos de: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    #  Manejo de Duplicados ---
    full_duplicates = df.duplicated().sum()
    print(f"Filas completamente duplicadas: {full_duplicates}")
    
    if full_duplicates > 0:
        df = df.drop_duplicates(keep='first')
        print(f"Forma después de eliminar filas duplicadas: {df.shape}")

    # Manejo de Datos Faltantes ---
    nan_count = df.isnull().sum().sum()
    print(f"Total de valores NaN encontrados: {nan_count}")
    

    #  Columnas Irrelevantes ---
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])

    #  Guardar Datos Procesados ---
    print(f"Guardando datos limpios en: {PROCESSED_DATA_PATH}")
    # Asegurarse de que el directorio 'data/processed' exista
    os.makedirs(PROCESSED_DATA_PATH.parent, exist_ok=True)
    
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print("--- Etapa 2: data_prep.py completada ---")


if __name__ == "__main__":
    main()