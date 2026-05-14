

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


# ======================================================
# FUNCIÓN SOLUCIÓN
# ======================================================
def verificar_drift_distribucion(
    df_entrenamiento: pd.DataFrame,
    df_actual: pd.DataFrame,
    alpha: float
) -> list:

    # 1. Obtener columnas comunes
    cols_comunes = [
        col for col in df_entrenamiento.columns
        if col in df_actual.columns
    ]

    # 2. Lista de columnas con drift
    columnas_con_drift = []

    # 3. Aplicar KS Test por columna
    for col in cols_comunes:

        # Eliminar NaN
        datos_train = df_entrenamiento[col].dropna()
        datos_actual = df_actual[col].dropna()

        # Test KS
        _, p_value = ks_2samp(
            datos_train,
            datos_actual
        )

        # Detectar drift
        if p_value < alpha:
            columnas_con_drift.append(col)

    return columnas_con_drift


# ======================================================
# GENERADOR DE CASOS DE USO
# ======================================================
def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso aleatorio.
    El input contiene DataFrames reales.
    """

    # ----------------------------------------
    # Configuración aleatoria
    # ----------------------------------------
    n_filas = np.random.randint(5, 10)

    cols = [
        'ventas',
        'clientes',
        'temperatura'
    ]

    alpha = 0.05

    # ----------------------------------------
    # 1. Datos de entrenamiento
    # ----------------------------------------
    df_entrenamiento = pd.DataFrame({
        col: np.random.uniform(
            10,
            100,
            n_filas
        ).round(2)
        for col in cols
    })

    # ----------------------------------------
    # 2. Datos actuales
    # ----------------------------------------
    df_actual = df_entrenamiento.copy()

    cols_con_drift = []

    for col in cols:

        # Decisión aleatoria
        if np.random.choice([True, False]):

            # Cambio drástico
            df_actual[col] = (
                df_actual[col] + 500
            ).round(2)

            cols_con_drift.append(col)

        else:

            # Cambio leve
            df_actual[col] = (
                df_actual[col]
                + np.random.normal(0, 1, n_filas)
            ).round(2)

    # ----------------------------------------
    # 3. Output esperado
    # ----------------------------------------
    output_real = []

    for col in cols:

        _, p_value = ks_2samp(
            df_entrenamiento[col],
            df_actual[col]
        )

        if p_value < alpha:
            output_real.append(col)

    # ----------------------------------------
    # INPUT / OUTPUT
    # ----------------------------------------
    input_data = {
        "df_entrenamiento": df_entrenamiento,
        "df_actual": df_actual,
        "alpha": alpha
    }

    output_data = output_real

    return input_data, output_data


# ======================================================
# TEST LOCAL
# ======================================================
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("PROBLEMA — DETECCIÓN DE DRIFT DE DATOS")

    # Generar caso de uso
    entrada, salida_esperada = (
        generar_caso_de_uso_preparar_datos()
    )

    # Mostrar INPUT
    print("\n--- INPUT: DATAFRAME ENTRENAMIENTO ---")
    print(entrada["df_entrenamiento"])

    print("\n--- INPUT: DATAFRAME ACTUAL ---")
    print(entrada["df_actual"])

    print(f"\nAlpha: {entrada['alpha']}")

    # Ejecutar solución
    resultado = verificar_drift_distribucion(
        entrada["df_entrenamiento"],
        entrada["df_actual"],
        entrada["alpha"]
    )

    # Mostrar OUTPUT
    print("\n--- OUTPUT ESPERADO ---")
    print(salida_esperada)

    print("\n--- OUTPUT OBTENIDO ---")
    print(resultado)

    # Validación
    print("\n¿Coinciden?")
    print(resultado == salida_espe
