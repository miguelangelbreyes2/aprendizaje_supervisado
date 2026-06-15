import os
import pickle
import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modelos del proyecto avanzado
MODEL_RF_PATH = os.path.join(BASE_DIR, "model_rf.joblib")
MODEL_LR_PATH = os.path.join(BASE_DIR, "model_lr.joblib")

# Modelo simple solicitado por la clase
MODELO_SIMPLE_PATH = os.path.join(BASE_DIR, "modelo.pkl")

models = {}
modelo_simple = None


def load_models():
    """Carga los modelos avanzados guardados con joblib."""
    if os.path.exists(MODEL_RF_PATH):
        try:
            models["rf"] = joblib.load(MODEL_RF_PATH)
            print(f"Modelo Random Forest cargado desde {MODEL_RF_PATH}")
        except Exception as e:
            print(f"Error al cargar Random Forest: {e}")

    if os.path.exists(MODEL_LR_PATH):
        try:
            models["lr"] = joblib.load(MODEL_LR_PATH)
            print(f"Modelo Regresión Logística cargado desde {MODEL_LR_PATH}")
        except Exception as e:
            print(f"Error al cargar Regresión Logística: {e}")


def load_simple_model():
    """Carga el modelo simple solicitado en la lección usando pickle."""
    global modelo_simple

    if os.path.exists(MODELO_SIMPLE_PATH):
        try:
            with open(MODELO_SIMPLE_PATH, "rb") as file:
                modelo_simple = pickle.load(file)
            print(f"Modelo simple cargado desde {MODELO_SIMPLE_PATH}")
        except Exception as e:
            print(f"Error al cargar modelo.pkl: {e}")
    else:
        print("No se encontró modelo.pkl")


load_models()
load_simple_model()


@app.route("/")
def index():
    """Renderiza la página principal."""
    return render_template("index.html")


@app.route("/predecir", methods=["POST"])
def predecir():
    """
    Endpoint solicitado en la lección 2.5.

    Recibe un JSON con este formato:
    {
        "input": [0, 0, 0, 0, 0, 0, 0]
    }

    Regresa:
    {
        "prediccion": 1
    }
    """
    try:
        if modelo_simple is None:
            return jsonify({
                "error": "El modelo simple no está disponible. Verifica que exista modelo.pkl."
            }), 500

        data = request.get_json(force=True)

        if "input" not in data:
            return jsonify({
                "error": "Falta la clave 'input' en el JSON."
            }), 400

        input_data = np.array(data["input"]).reshape(1, -1)

        prediccion = modelo_simple.predict(input_data)

        return jsonify({
            "prediccion": int(prediccion[0])
        })

    except Exception as e:
        return jsonify({
            "error": f"Error al realizar la predicción: {str(e)}"
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint avanzado del proyecto.

    Este endpoint usa pipelines de Scikit-Learn y permite enviar datos más entendibles,
    como Sex='male' o Embarked='S'.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No se proporcionaron datos de entrada"}), 400

        model_type = data.get("model", "rf")
        pipeline = models.get(model_type)

        if not pipeline:
            load_models()
            pipeline = models.get(model_type)

            if not pipeline:
                return jsonify({
                    "error": f'El modelo "{model_type}" no está disponible. Ejecuta train.py primero.'
                }), 500

        features = {
            "Pclass": int(data.get("Pclass", 3)),
            "Sex": str(data.get("Sex", "male")).lower(),
            "Age": float(data.get("Age", 28.0)),
            "SibSp": int(data.get("SibSp", 0)),
            "Parch": int(data.get("Parch", 0)),
            "Fare": float(data.get("Fare", 20.0)),
            "Embarked": str(data.get("Embarked", "S")).upper()
        }

        input_df = pd.DataFrame([features])

        prediction = int(pipeline.predict(input_df)[0])

        probability = None

        try:
            prob_array = pipeline.predict_proba(input_df)[0]
            probability = float(prob_array[prediction])
        except Exception:
            pass

        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "model_used": "Random Forest" if model_type == "rf" else "Regresión Logística"
        })

    except Exception as e:
        return jsonify({
            "error": f"Error en el procesamiento: {str(e)}"
        }), 500


if __name__ == "__main__":
    print("API de Flask ejecutándose en http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)