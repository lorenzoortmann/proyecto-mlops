import pandas as pd
import yaml
import joblib
import json
import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===  Cargar variables del archivo .env ===
load_dotenv()

# ===  Configurar MLflow con las variables ===
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

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

    # === Configurar MLflow ===
    mlflow.set_tracking_uri("https://dagshub.com/lorenzoortmann/proyecto-mlops.mlflow")
    # === Nombramos el experimento ===
    mlflow.set_experiment("Logistic_Regression")

    with mlflow.start_run(run_name="train_logreg"):
    # Log de parámetros "globales"
        mlflow.log_params({
            "model.type": params["model"]["type"],
            "model.max_iter": params["model"]["max_iter"],
            "split.test_size": test_size,
            "split.random_state": random_state,
            "data.processed_path": data_path
        })

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
        # Log en MLflow
        mlflow.log_metrics(metrics)

        # === Guardar modelo, scaler y métricas ===
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

        joblib.dump(model, model_output_path)
        joblib.dump(scaler, scaler_path)

        # Modelo en formato MLflow
        model_info=mlflow.sklearn.log_model(model, "modelo_regresion_logistica")
        model_uri=model_info.model_uri
        model_name="Logistic_Regression_Model"
        mlflow.register_model(model_uri, model_name)
        
        # Guardar métricas en archivo
        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Loguear artefactos
        mlflow.log_artifact(scaler_path, artifact_path="artifacts")
        mlflow.log_artifact("params/params.yaml", artifact_path="artifacts")
        mlflow.log_artifact(metrics_output_path, artifact_path="artifacts")

        print(f"\nModelo guardado en: {model_output_path}")
        print(f"Métricas guardadas en: {metrics_output_path}")


if __name__ == "__main__":
    main()
