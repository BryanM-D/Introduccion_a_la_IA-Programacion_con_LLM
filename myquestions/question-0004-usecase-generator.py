import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def generar_caso_de_uso_edificios(n=700, seed=42):

    # =========================================================
    # Generación de datos sintéticos
    # =========================================================
    np.random.seed(seed)

    consumo = np.random.normal(
        300,
        50,
        n
    )

    area = np.random.normal(
        120,
        30,
        n
    )

    anio = np.random.randint(
        1980,
        2023,
        n
    )

    aislamiento = np.random.randint(
        0,
        2,
        n
    )

    # =========================================================
    # Regla simple de clasificación energética
    # =========================================================
    categoria = np.where(
        consumo < 250,
        "A",
        np.where(
            consumo < 320,
            "B",
            "C"
        )
    )

    # =========================================================
    # DataFrame original
    # =========================================================
    df = pd.DataFrame({
        "consumo_kwh": consumo,
        "area_m2": area,
        "anio_construccion": anio,
        "aislamiento": aislamiento,
        "categoria": categoria
    })

    # =========================================================
    # Primer output: diccionario serializable
    # =========================================================
    input_dict = {
        'df': df.copy().to_dict(orient='records')
    }

    # =========================================================
    # Preprocesamiento
    # =========================================================
    le = LabelEncoder()

    df['categoria'] = le.fit_transform(
        df['categoria']
    )

    # =========================================================
    # Features y target
    # =========================================================
    features = [
        'consumo_kwh',
        'area_m2',
        'anio_construccion',
        'aislamiento'
    ]

    X = df[features].values

    y = df['categoria'].values

    # =========================================================
    # Distribución de clases
    # =========================================================
    clase_A = int((y == 0).sum())

    clase_B = int((y == 1).sum())

    clase_C = int((y == 2).sum())

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
            average='macro',
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

        'f1_macro': f1,

        'f1_weighted': f1_w,

        'distribucion_clases': {

            'A': clase_A,

            'B': clase_B,

            'C': clase_C
        },

        'predicciones': preds.tolist()
    }

    return input_dict, output
