import pandas as pd
import numpy as np
import random

from sklearn.ensemble import RandomForestRegressor

# ======================================================
# FUNCIÓN SOLUCIÓN
# ======================================================
def crear_features_temporales(
    df: pd.DataFrame,
    target_col: str
):

    # 1. Crear copia
    df = df.copy()

    # 2. Convertir fecha a datetime
    df["fecha"] = pd.to_datetime(
        df["fecha"]
    )

    # 3. Crear variables temporales
    df["dia_semana"] = (
        df["fecha"].dt.dayofweek
    )

    df["mes"] = (
        df["fecha"].dt.month
    )

    df["fin_semana"] = (
        df["dia_semana"] >= 5
    ).astype(int)

    # 4. Construir X e y
    X = df[
        [
            "dia_semana",
            "mes",
            "fin_semana"
        ]
    ]

    y = df[target_col]

    # 5. Entrenar modelo
    model = RandomForestRegressor(
        random_state=42
    )

    model.fit(X, y)

    # 6. Retornar importancia
    return model.feature_importances_


# ======================================================
# GENERADOR DE CASOS DE USO
# ======================================================
def generar_caso_de_uso_crear_features_temporales():

    # ----------------------------------------
    # Número aleatorio de filas
    # ----------------------------------------
    n = random.randint(20, 50)

    # ----------------------------------------
    # Crear fechas
    # ----------------------------------------
    fechas = pd.date_range(
        start="2020-01-01",
        periods=n,
        freq="D"
    )

    # ----------------------------------------
    # Crear DataFrame
    # ----------------------------------------
    df = pd.DataFrame({
        "fecha": fechas,
        "valor": np.random.randn(n)
    })

    # ----------------------------------------
    # INPUT DATA
    # ----------------------------------------
    input_data = {
        "df": df.copy(),
        "target_col": "valor"
    }

    # ----------------------------------------
    # FEATURE ENGINEERING
    # ----------------------------------------
    df["fecha"] = pd.to_datetime(
        df["fecha"]
    )

    df["dia_semana"] = (
        df["fecha"].dt.dayofweek
    )

    df["mes"] = (
        df["fecha"].dt.month
    )

    df["fin_semana"] = (
        df["dia_semana"] >= 5
    ).astype(int)

    # ----------------------------------------
    # X e y
    # ----------------------------------------
    X = df[
        [
            "dia_semana",
            "mes",
            "fin_semana"
        ]
    ]

    y = df["valor"]

    # ----------------------------------------
    # Modelo
    # ----------------------------------------
    model = RandomForestRegressor(
        random_state=42
    )

    model.fit(X, y)

    # ----------------------------------------
    # OUTPUT ESPERADO
    # ----------------------------------------
    output_data = model.feature_importances_

    return input_data, output_data


# ======================================================
# TEST LOCAL
# ======================================================
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("PROBLEMA — FEATURES TEMPORALES")

    # Generar caso
    entrada, salida_esperada = (
        generar_caso_de_uso_crear_features_temporales()
    )

    # Mostrar INPUT
    print("\n--- INPUT DATAFRAME ---")
    print(entrada["df"].head())

    print(
        f"\nTarget column: "
        f"{entrada['target_col']}"
    )

    # Ejecutar solución
    resultado = crear_features_temporales(
        entrada["df"],
        entrada["target_col"]
    )

    # Mostrar OUTPUT
    print("\n--- IMPORTANCIA DE VARIABLES ---")
    print(resultado)

    # Validación
    print("\n¿Coincide?")
    print(
        np.allclose(
            resultado,
            salida_esperada
        )
    )
