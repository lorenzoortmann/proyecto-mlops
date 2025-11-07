import pandas as pd
import yaml
import os

# Cargar parámetros
with open("params/params.yaml") as f:
    params = yaml.safe_load(f)

def load_data(file_path):
    """Carga datos desde un archivo CSV y devuelve un DataFrame de pandas."""
    try:
        data = pd.read_csv(file_path)
        print(f"Datos cargados exitosamente desde {file_path}")
        return data
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None
    
def preprocess_data(data):
    """Realiza la limpieza básica de los datos."""
    if data is not None:
        # Eliminar ids y aplicar get dummies para variables categóricas
        data.drop(columns=['customer_id'], inplace=True, errors='ignore')
        data = pd.get_dummies(data, drop_first=True)
        
        print("Datos preprocesados (limpieza y dummies) exitosamente.")
    return data


def save_data(data, file_path):
    """Guarda el dataframe procesado en un archivo CSV."""
    if data is not None:
        try:
            # Asegurarse de que el directorio de salida exista
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data.to_csv(file_path, index=False)
            print(f"Datos preprocesados guardados en {file_path}")
        except Exception as e:
            print(f"Error al guardar los datos procesados: {e}")

def main():
    data = load_data(params['paths']['raw_data'])
    data = preprocess_data(data) 
    # Ahora sí guardamos el resultado
    output_file = params['paths']['processed_data']
    save_data(data, output_file)


if __name__ == "__main__":
    main()