import random
import numpy as np
import pandas as pd


# --------------------------------------------------
# FUNCIÓN SOLUCIÓN
# --------------------------------------------------
def evaluar_lotes(df: pd.DataFrame, id_col: str) -> pd.DataFrame:

    # 1. Seleccionar columnas de tests
    test_cols = [col for col in df.columns if col != id_col]

    # 2. Mediana global por test
    medianas = df[test_cols].median()

    # 3. Umbral global
    umbral = medianas.mean()

    # 4. Promedio por lote
    promedios = df[test_cols].mean(axis=1)

    # 5. Evaluar aprobación
    resultado = pd.DataFrame({
        id_col: df[id_col],
        "promedio_lote": promedios.round(4),
        "aprueba": promedios > umbral
    }).reset_index(drop=True)

    return resultado


# --------------------------------------------------
# GENERADOR DE CASOS DE USO
# --------------------------------------------------
def generar_caso_de_uso_evaluar_lotes():
    """
    Genera un caso de prueba aleatorio (input / output esperado)
    para la función evaluar_lotes.
    """

    n_lotes = random.randint(8, 20)
    n_tests = random.randint(3, 7)

    id_col = "lote_id"

    # Nombres de tests
    test_cols = [f"test_{chr(65 + i)}" for i in range(n_tests)]

    # Datos aleatorios
    data = np.random.uniform(50, 150, size=(n_lotes, n_tests)).astype(float)

    # Introducir NaN (~15%)
    mask = np.random.choice(
        [True, False],
        size=data.shape,
        p=[0.15, 0.85]
    )

    data[mask] = np.nan

    # DataFrame de entrada
    df = pd.DataFrame(data, columns=test_cols)

    df.insert(
        0,
        id_col,
        [f"LOT-{1000 + i}" for i in range(n_lotes)]
    )

    # OUTPUT ESPERADO
    output_df = evaluar_lotes(df, id_col)

    # INPUT / OUTPUT
    input_data = {
        "df": df.copy(),
        "id_col": id_col
    }

    output_data = output_df

    return input_data, output_data


# --------------------------------------------------
# TEST LOCAL
# --------------------------------------------------
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("PREGUNTA 2 — evaluar_lotes (pandas)")

    entrada, salida = generar_caso_de_uso_evaluar_lotes()

    print(
        f"INPUT — id_col='{entrada['id_col']}', "
        f"DataFrame shape: {entrada['df'].shape}"
    )

    print("\nPrimeras filas INPUT:")
    print(entrada["df"].head())

    print("\nOUTPUT esperado:")
    print(salida.head())

    # Validación
    resultado = evaluar_lotes(
        entrada["df"],
        entrada["id_col"]
    )

    print("\n¿Coincide con la salida esperada?")
    print(resultado.equals(salida))
