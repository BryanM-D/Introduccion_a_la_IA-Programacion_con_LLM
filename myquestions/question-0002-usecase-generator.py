import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def generar_caso_de_uso_estudiantes(n=800, seed=42):

    # =========================================================
    # Generación de datos sintéticos
    # =========================================================
    np.random.seed(seed)

    promedio = np.random.normal(
        3.0,
        0.5,
        n
    )

    inasistencias = np.random.randint(
        0,
        30,
        n
    )

    nivel_socioeconomico = np.random.randint(
        1,
        4,
        n
    )

    horas_estudio = np.random.normal(
        5,
        2,
        n
    )

    # =========================================================
    # Regla simple de deserción
    # =========================================================
    deserta = (
        (promedio < 2.8) |
        (inasistencias > 15) |
        (horas_estudio < 3)
    ).astype(int)

    # =========================================================
    # DataFrame original
    # =========================================================
    df = pd.DataFrame({
        "promedio": promedio,
        "inasistencias": inasistencias,
        "nivel_socioeconomico": nivel_socioeconomico,
        "horas_estudio": horas_estudio,
        "deserta": deserta
    })

    # =========================================================
    # Primer output: diccionario serializable
    # =========================================================
    input_dict = {
        'df': df.copy().to_dict(orient='records')
    }

    # =========================================================
    # Features y target
    # =========================================================
    features = [
        'promedio',
        'inasistencias',
        'nivel_socioeconomico',
        'horas_estudio'
    ]

    X = df[features].values

    y = df['deserta'].values

    # =========================================================
    # Análisis de balance
    # =========================================================
    n_no_desertan = int((y == 0).sum())

    n_desertan = int((y == 1).sum())

    ratio = (
        round(n_desertan / n_no_desertan, 4)
        if n_no_desertan > 0 else float('inf')
    )

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

    modelo.fit(X_train, y_train)

    # =========================================================
    # Predicciones
    # =========================================================
    preds = modelo.predict(X_test)

    # =========================================================
    # Métricas
    # =========================================================
    cm = confusion_matrix(
        y_test,
        preds
    )

    f1 = round(
        f1_score(
            y_test,
            preds,
            zero_division=0
        ),
        4
    )

    f1_w = round(
        f1_score(
            y_test,
            preds,
            average='weighted',
            zero_division=0
        ),
        4
    )

    # =========================================================
    # Segundo output
    # =========================================================
    output = {

        'modelo': str(modelo),

        'confusion_matrix': cm.tolist(),

        'f1': f1,

        'f1_weighted': f1_w,

        'balance': {

            'no_desertan': n_no_desertan,

            'desertan': n_desertan,

            'ratio_desertan/no_desertan': ratio,

            'desbalanceado': ratio < 0.2
        },

        'predicciones': preds.tolist()
    }

    return input_dict, output
