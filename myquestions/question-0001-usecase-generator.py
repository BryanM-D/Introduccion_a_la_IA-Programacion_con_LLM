import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def generar_caso_de_uso_detectar_sesgo_creditos(n=1000):

    # =========================================================
    # Generación de datos sintéticos
    # =========================================================
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
    # Primer output: diccionario serializable (input para la función del usuario)
    # =========================================================
    input_dict = {
        'df': df.copy().to_dict(orient='records')
    }

    # =========================================================
    # Features y target para el entrenamiento del modelo del generador
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
    # Análisis de balance (solo para información interna del generador, no en el output esperado)
    # =========================================================
    n_no_aprobados = int((y == 0).sum())

    n_aprobados = int((y == 1).sum())

    ratio = (
        round(n_aprobados / n_no_aprobados, 4)
        if n_no_aprobados > 0 else float('inf')
    )

    # =========================================================
    # División train/test (igual que en la función del usuario)
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # =========================================================
    # Modelo (igual que en la función del usuario)
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
    # Calcular el output esperado (reporte)
    # =========================================================
    # Re-crear X_test como un DataFrame para facilitar el acceso a 'genero'
    X_test_df = pd.DataFrame(X_test, columns=features)

    test_results_df = pd.DataFrame({
        'genero': X_test_df['genero'].astype(int).values,
        'aprobado_real': y_test,
        'prediccion_aprobado': preds
    })

    output_esperado = {}
    generos = test_results_df['genero'].unique()
    generos.sort()

    for g in generos:
        subset = test_results_df[test_results_df['genero'] == g]
        actual_aprobado = subset['aprobado_real']
        prediccion_aprobado = subset['prediccion_aprobado']

        n_solicitantes = len(subset)
        tasa_aprobacion_real = actual_aprobado.mean()
        tasa_aprobacion_predicha = prediccion_aprobado.mean()
        accuracy = accuracy_score(actual_aprobado, prediccion_aprobado)
        precision = precision_score(actual_aprobado, prediccion_aprobado, zero_division=0)
        recall = recall_score(actual_aprobado, prediccion_aprobado, zero_division=0)
        f1 = f1_score(actual_aprobado, prediccion_aprobado, zero_division=0)

        output_esperado[f'genero_{g}'] = {
            'n_solicitantes': n_solicitantes,
            'tasa_aprobacion_real': tasa_aprobacion_real,
            'tasa_aprobacion_predicha': tasa_aprobacion_predicha,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    if len(generos) == 2:
        g0_actual_rate = output_esperado[f'genero_{generos[0]}']['tasa_aprobacion_real']
        g1_actual_rate = output_esperado[f'genero_{generos[1]}']['tasa_aprobacion_real']
        diff_actual = abs(g0_actual_rate - g1_actual_rate)

        g0_pred_rate = output_esperado[f'genero_{generos[0]}']['tasa_aprobacion_predicha']
        g1_pred_rate = output_esperado[f'genero_{generos[1]}']['tasa_aprobacion_predicha']
        diff_pred = abs(g0_pred_rate - g1_pred_rate)

        output_esperado['diferencia_tasa_aprobacion_real'] = diff_actual
        output_esperado['diferencia_tasa_aprobacion_predicha'] = diff_pred
        # Usando un umbral de 0.05 como en la descripción del problema
        output_esperado['sesgo_detectado_real'] = diff_actual > 0.05
        output_esperado['sesgo_detectado_predicho'] = diff_pred > 0.05

    return input_dict, output_esperado
