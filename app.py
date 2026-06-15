import os
import pickle
import joblib
import pandas as pd

from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de modelos
PIPELINE_PATH = os.path.join(BASE_DIR, "pipeline.pkl")
MODEL_RF_PATH = os.path.join(BASE_DIR, "model_rf.joblib")
MODEL_LR_PATH = os.path.join(BASE_DIR, "model_lr.joblib")

# Variables globales para modelos
pipeline_model = None
models = {}


def cargar_pipeline():
    """
    Carga el pipeline principal guardado como pipeline.pkl.
    Este pipeline ya incluye preprocesamiento, escalamiento y modelo.
    """
    global pipeline_model

    if os.path.exists(PIPELINE_PATH):
        try:
            with open(PIPELINE_PATH, "rb") as archivo_modelo:
                pipeline_model = pickle.load(archivo_modelo)
            print("Pipeline cargado correctamente desde pipeline.pkl")
        except Exception as error:
            print(f"Error al cargar pipeline.pkl: {error}")
    else:
        print("No se encontró pipeline.pkl. Ejecuta primero pipeline_train.py")


def cargar_modelos_joblib():
    """
    Carga modelos adicionales del proyecto, si existen.
    Estos modelos sirven para el endpoint avanzado /predict.
    """
    if os.path.exists(MODEL_RF_PATH):
        try:
            models["rf"] = joblib.load(MODEL_RF_PATH)
            print("Modelo Random Forest cargado correctamente")
        except Exception as error:
            print(f"Error al cargar model_rf.joblib: {error}")

    if os.path.exists(MODEL_LR_PATH):
        try:
            models["lr"] = joblib.load(MODEL_LR_PATH)
            print("Modelo Regresión Logística cargado correctamente")
        except Exception as error:
            print(f"Error al cargar model_lr.joblib: {error}")


cargar_pipeline()
cargar_modelos_joblib()


@app.route("/")
def index():
    """
    Página principal de la aplicación.
    Si existe templates/index.html, lo muestra.
    """
    try:
        return render_template("index.html")
    except Exception:
        return jsonify({
            "mensaje": "API de predicción del Titanic funcionando correctamente",
            "endpoints": {
                "predecir": "/predecir",
                "predict": "/predict"
            }
        })


@app.route("/predecir", methods=["POST"])
def predecir():
    """
    Endpoint principal de la clase 2.6 Pipelines.

    Recibe datos sin transformar, por ejemplo:
    {
        "Pclass": 2,
        "Sex": "male",
        "Age": 46,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "C"
    }

    El pipeline se encarga de transformar los datos y hacer la predicción.
    """
    try:
        if pipeline_model is None:
            return jsonify({
                "error": "El pipeline no está disponible. Ejecuta primero pipeline_train.py para generar pipeline.pkl."
            }), 500

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No se recibieron datos en formato JSON."
            }), 400

        columnas_requeridas = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]

        columnas_faltantes = [
            columna for columna in columnas_requeridas
            if columna not in data
        ]

        if columnas_faltantes:
            return jsonify({
                "error": "Faltan columnas en el JSON.",
                "columnas_faltantes": columnas_faltantes
            }), 400

        input_data = pd.DataFrame([{
            "Pclass": int(data["Pclass"]),
            "Sex": str(data["Sex"]).lower(),
            "Age": float(data["Age"]),
            "SibSp": int(data["SibSp"]),
            "Parch": int(data["Parch"]),
            "Fare": float(data["Fare"]),
            "Embarked": str(data["Embarked"]).upper()
        }])

        prediccion = pipeline_model.predict(input_data)

        return jsonify({
            "Survived": int(prediccion[0])
        })

    except Exception as error:
        return jsonify({
            "error": f"Error al realizar la predicción: {str(error)}"
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint avanzado del proyecto.
    Permite elegir entre Random Forest y Regresión Logística si existen los modelos joblib.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No se recibieron datos en formato JSON."
            }), 400

        model_type = data.get("model", "rf")
        modelo = models.get(model_type)

        if modelo is None:
            return jsonify({
                "error": f"El modelo '{model_type}' no está disponible. Verifica que exista model_rf.joblib o model_lr.joblib."
            }), 500

        input_data = pd.DataFrame([{
            "Pclass": int(data.get("Pclass", 3)),
            "Sex": str(data.get("Sex", "male")).lower(),
            "Age": float(data.get("Age", 28)),
            "SibSp": int(data.get("SibSp", 0)),
            "Parch": int(data.get("Parch", 0)),
            "Fare": float(data.get("Fare", 7.25)),
            "Embarked": str(data.get("Embarked", "S")).upper()
        }])

        prediccion = int(modelo.predict(input_data)[0])

        probabilidad = None

        try:
            probabilidades = modelo.predict_proba(input_data)[0]
            probabilidad = float(probabilidades[prediccion])
        except Exception:
            pass

        return jsonify({
            "prediction": prediccion,
            "probability": probabilidad,
            "model_used": "Random Forest" if model_type == "rf" else "Regresión Logística"
        })

    except Exception as error:
        return jsonify({
            "error": f"Error en el procesamiento: {str(error)}"
        }), 500


if __name__ == "__main__":
    print("API de Flask ejecutándose en http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)