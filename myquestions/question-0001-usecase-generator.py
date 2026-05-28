import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def generar_caso_de_uso_creditos(n=1000, seed=42):

    # =========================================================
    # Generación de datos sintéticos
    # =========================================================
    np.random.seed(seed)

    edad = np.random.randint(18, 65, n)

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
    # Regla simple de aprobación
    # =========================================================
    prob = (
        (puntaje_credito / 850) +
        (ingresos / 10000)
    )

    aprobado = (prob > 0.9).astype(int)

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
    # Primer output: diccionario serializable
    # =========================================================
    input_dict = {
        'df': df.copy().to_dict(orient='records')
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
    # Análisis de balance
    # =========================================================
    n_no_aprobados = int((y == 0).sum())

    n_aprobados = int((y == 1).sum())

    ratio = (
        round(n_aprobados / n_no_aprobados, 4)
        if n_no_aprobados > 0 else float('inf')
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

            'no_aprobados': n_no_aprobados,

            'aprobados': n_aprobados,

            'ratio_aprobados/no_aprobados': ratio,

            'desbalanceado': ratio < 0.2
        },

        'predicciones': preds.tolist()
    }

    return input_dict, output
