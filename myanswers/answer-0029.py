import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ======================================================
# FUNCIÓN SOLUCIÓN
# ======================================================
def preparar_sensores(
    df: pd.DataFrame,
    target_col: str
):

    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # 2. Calcular percentiles
    lower = X.quantile(0.05)
    upper = X.quantile(0.95)

    # 3. Limitar valores con clip
    X_clipped = X.clip(
        lower=lower,
        upper=upper,
        axis=1
    )

    # 4. Escalar datos
    scaler = StandardScaler()

    X_procesado = scaler.fit_transform(
        X_clipped
    )

    # 5. Retornar tupla
    return X_procesado, y


# ======================================================
# GENERADOR DE CASOS DE USO
# ======================================================
def generar_caso_de_uso_preparar_sensores():

    # ----------------------------------------
    # Configuración aleatoria
    # ----------------------------------------
    n_rows = np.random.randint(15, 30)
    n_feats = np.random.randint(2, 4)

    # ----------------------------------------
    # Crear datos de sensores
    # ----------------------------------------
    data = (
        np.random.randn(
            n_rows,
            n_feats
        ) * 10
    )

    df = pd.DataFrame(
        data,
        columns=[f"s{i}" for i in range(n_feats)]
    )

    # ----------------------------------------
    # Variable objetivo
    # ----------------------------------------
    target_col = "target"

    df[target_col] = np.random.randint(
        0,
        2,
        n_rows
    )

    # ----------------------------------------
    # INPUT DATA
    # ----------------------------------------
    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    # ----------------------------------------
    # OUTPUT ESPERADO
    # ----------------------------------------
    X = df.drop(columns=[target_col])

    y = df[target_col].values

    # Percentiles
    lower = X.quantile(0.05)
    upper = X.quantile(0.95)

    # Clip
    X_clipped = X.clip(
        lower=lower,
        upper=upper,
        axis=1
    )

    # Escalado
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_clipped
    )

    output_data = (X_scaled, y)

    return input_data, output_data


# ======================================================
# TEST LOCAL
# ======================================================
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("PROBLEMA — PREPARAR DATOS DE SENSORES")

    # Generar caso
    entrada, salida_esperada = (
        generar_caso_de_uso_preparar_sensores()
    )

    # Mostrar INPUT
    print("\n--- INPUT DATAFRAME ---")
    print(entrada["df"].head())

    print(f"\nTarget column: {entrada['target_col']}")

    # Ejecutar solución
    resultado = preparar_sensores(
        entrada["df"],
        entrada["target_col"]
    )

    # Mostrar OUTPUT
    print("\n--- SHAPE X_PROCESADO ---")
    print(resultado[0].shape)

    print("\n--- PRIMEROS VALORES y ---")
    print(resultado[1][:10])

    # Validación
    print("\n¿Coincide X?")
    print(
        np.allclose(
            resultado[0],
            salida_esperada[0]
        )
    )

    print("\n¿Coincide y?")
    print(
        np.array_equal(
            resultado[1],
            salida_esperada[1]
        )
    )
