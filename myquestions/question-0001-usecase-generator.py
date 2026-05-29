import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split


def generar_caso_de_uso_detectar_sesgo_creditos(
    n=1000,
    seed=42
):

    # =========================================================
    # Semilla para reproducibilidad
    # =========================================================
    np.random.seed(seed)

    # =========================================================
    # Generación de datos sintéticos
    # =========================================================
    edad = np.random.randint(
        18,
        65,
        n
    )

    ingresos = np.random.normal(
        3000,
        1000,
        n
    )

    puntaje_credito = np.random.normal(
        600,
        100,
        n
    )

    genero = np.random.randint(
        0,
        2,
        n
    )  # 0 mujer, 1 hombre

    # =========================================================
    # Regla de aprobación
    # =========================================================
    prob = (
        (puntaje_credito / 850) +
        (ingresos / 10000)
    )

    aprobado = (
        prob > 0.9
    ).astype(int)

    # =========================================================
    # DataFrame original
    # =========================================================
    df = pd.DataFrame({
        "edad": edad,
        "ingresos": ingresos,
        "puntaje_credito": puntaje_credito,
        "genero": genero,
        "aprobado": aprobado
    })

    # =========================================================
    # Input serializable
    # =========================================================
    input_dict = {
        'df': df.to_dict(
            orient='records'
        )
    }

    # =========================================================
    # Features y target
    # =========================================================
    features = [
        'edad',
        'ingresos',
        'puntaje_credito',
        'genero'
    ]

    X = df[features].values

    y = df['aprobado'].values

    # =========================================================
    # División train/test
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # =========================================================
    # Modelo
    # =========================================================
    modelo = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(
        X_train,
        y_train
    )

    # =========================================================
    # Predicciones
    # =========================================================
    preds = modelo.predict(X_test)

    # =========================================================
    # DataFrame resultados
    # =========================================================
    X_test_df = pd.DataFrame(
        X_test,
        columns=features
    )

    resultados = pd.DataFrame({
        'genero': X_test_df['genero'].astype(int),
        'aprobado_real': y_test,
        'prediccion_aprobado': preds
    })

    # =========================================================
    # Output esperado
    # =========================================================
    output_esperado = {}

    generos = sorted(
        resultados['genero'].unique()
    )

    for g in generos:

        subset = resultados[
            resultados['genero'] == g
        ]

        y_real = subset[
            'aprobado_real'
        ]

        y_pred = subset[
            'prediccion_aprobado'
        ]

        output_esperado[
            f'genero_{g}'
        ] = {

            'n_solicitantes':
                int(len(subset)),

            'tasa_aprobacion_real':
                float(y_real.mean()),

            'tasa_aprobacion_predicha':
                float(y_pred.mean()),

            'accuracy':
                float(
                    accuracy_score(
                        y_real,
                        y_pred
                    )
                ),

            'precision':
                float(
                    precision_score(
                        y_real,
                        y_pred,
                        zero_division=0
                    )
                ),

            'recall':
                float(
                    recall_score(
                        y_real,
                        y_pred,
                        zero_division=0
                    )
                ),

            'f1_score':
                float(
                    f1_score(
                        y_real,
                        y_pred,
                        zero_division=0
                    )
                )
        }

    # =========================================================
    # Comparación entre géneros
    # =========================================================
    if len(generos) == 2:

        g0_real = output_esperado[
            f'genero_{generos[0]}'
        ][
            'tasa_aprobacion_real'
        ]

        g1_real = output_esperado[
            f'genero_{generos[1]}'
        ][
            'tasa_aprobacion_real'
        ]

        diff_real = abs(
            g0_real - g1_real
        )

        g0_pred = output_esperado[
            f'genero_{generos[0]}'
        ][
            'tasa_aprobacion_predicha'
        ]

        g1_pred = output_esperado[
            f'genero_{generos[1]}'
        ][
            'tasa_aprobacion_predicha'
        ]

        diff_pred = abs(
            g0_pred - g1_pred
        )

        output_esperado[
            'diferencia_tasa_aprobacion_real'
        ] = float(diff_real)

        output_esperado[
            'diferencia_tasa_aprobacion_predicha'
        ] = float(diff_pred)

        output_esperado[
            'sesgo_detectado_real'
        ] = bool(
            diff_real > 0.05
        )

        output_esperado[
            'sesgo_detectado_predicho'
        ] = bool(
            diff_pred > 0.05
        )

    return (
        input_dict,
        output_esperado
    )
