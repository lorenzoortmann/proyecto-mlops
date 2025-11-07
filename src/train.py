import pandas as pd
import yaml
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # === Cargar parámetros ===
    with open("params/params.yaml") as f:
        params = yaml.safe_load(f)

    data_path = params["paths"]["processed_data"]
    model_output_path = params["paths"]["model_path"]
    metrics_output_path = params["paths"]["metrics"]
    scaler_path = params["paths"]["scaler"]

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    # === Cargar los datos ===
    data = pd.read_csv(data_path)
    X = data.drop(columns=["churn"])
    y = data["churn"]

    # === Split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # === Escalamiento ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Entrenamiento del modelo ===
    if params["model"]["type"] == "LogisticRegression":
        model = LogisticRegression(max_iter=params["model"]["max_iter"])
    else:
        raise ValueError("model.type no soportado")
    model.fit(X_train_scaled, y_train)

    # === Predicciones ===
    y_pred = model.predict(X_test_scaled)

    # === Cálculo de métricas ===
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print(f"\nMétricas del modelo:\n{json.dumps(metrics, indent=4)}")

    # === Guardar modelo, scaler y métricas ===
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_path)


    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nModelo guardado en: {model_output_path}")
    print(f"Métricas guardadas en: {metrics_output_path}")


if __name__ == "__main__":
    main()
