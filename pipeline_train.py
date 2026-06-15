import pickle
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.metrics import accuracy_score


# 1. Cargar datos limpios
df = pd.read_csv("./data/titanic_clean.csv")

# 2. Separar variables predictoras y variable objetivo
X = df.drop(["Survived"], axis=1)
y = df["Survived"]

# 3. Crear preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["Sex", "Embarked"]),
        ("age", QuantileTransformer(output_distribution="normal", n_quantiles=500), ["Age"]),
        ("fare", QuantileTransformer(output_distribution="normal", n_quantiles=500), ["Fare"])
    ],
    remainder="passthrough"
)

# 4. Definir modelos e hiperparámetros
modelos = {
    "Regresión Logística": {
        "modelo": LogisticRegression(),
        "parametros": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"],
            "model__max_iter": [500, 1000]
        }
    },
    "Árbol de Decisión": {
        "modelo": DecisionTreeClassifier(),
        "parametros": {
            "model__splitter": ["best", "random"],
            "model__max_depth": [None, 1, 2, 3, 4]
        }
    },
    "Bosques Aleatorios": {
        "modelo": RandomForestClassifier(),
        "parametros": {
            "model__n_estimators": [10, 100],
            "model__max_depth": [None, 1, 2, 3, 4],
            "model__max_features": ["sqrt", "log2", None]
        }
    },
    "Gradient Boosting": {
        "modelo": GradientBoostingClassifier(),
        "parametros": {
            "model__n_estimators": [10, 100],
            "model__max_depth": [1, 2, 3, 4]
        }
    },
    "AdaBoost": {
        "modelo": AdaBoostClassifier(),
        "parametros": {
            "model__n_estimators": [10, 100]
        }
    },
    "KNN": {
        "modelo": KNeighborsClassifier(),
        "parametros": {
            "model__n_neighbors": [3, 5, 7]
        }
    },
    "GaussianNB": {
        "modelo": GaussianNB(),
        "parametros": {}
    },
    "BernoulliNB": {
        "modelo": BernoulliNB(),
        "parametros": {
            "model__alpha": [0.1, 1.0, 10.0]
        }
    }
}

# 5. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=100
)

# 6. Variables auxiliares
puntajes_modelos = []
mejor_precision = 0
mejor_estimador = None
mejor_modelo = None
estimadores = {}

# 7. Entrenar modelos con GridSearchCV y Pipeline
for nombre, info_modelo in modelos.items():
    print(f"Entrenando modelo: {nombre}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("scaler", MinMaxScaler()),
        ("model", info_modelo["modelo"])
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=info_modelo["parametros"],
        cv=5,
        scoring="accuracy",
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)

    precision = accuracy_score(y_test, y_pred)

    puntajes_modelos.append({
        "Modelo": nombre,
        "Precisión": precision
    })

    estimadores[nombre] = grid_search.best_estimator_

    if precision > mejor_precision:
        mejor_modelo = nombre
        mejor_precision = precision
        mejor_estimador = grid_search.best_estimator_

# 8. Mostrar resultados
metricas = pd.DataFrame(puntajes_modelos).sort_values("Precisión", ascending=False)

print("\nRendimiento de los modelos de clasificación")
print(metricas.round(2))

print("---------------------------------------------------")
print("MEJOR MODELO DE CLASIFICACIÓN")
print(f"Modelo: {mejor_modelo}")
print(f"Precisión: {mejor_precision:.2f}")

# 9. Guardar el mejor pipeline
with open("pipeline.pkl", "wb") as archivo_estimador:
    pickle.dump(mejor_estimador, archivo_estimador)

print("---------------------------------------------------")
print("Pipeline guardado correctamente como pipeline.pkl")