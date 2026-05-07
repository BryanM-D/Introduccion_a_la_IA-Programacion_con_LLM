import numpy as np
import pandas as pd

def generar_caso_de_uso_transacciones(n=1500, seed=42):

    np.random.seed(seed)

    # -----------------------------
    # INPUT DATA
    # -----------------------------
    monto = np.random.exponential(200, n)
    hora = np.random.randint(0, 24, n)
    pais = np.random.randint(0, 5, n)
    frecuencia_transacciones = np.random.poisson(3, n)

    input_data = pd.DataFrame({
        "monto": monto,
        "hora": hora,
        "pais": pais,
        "frecuencia_transacciones": frecuencia_transacciones
    })

    # -----------------------------
    # OUTPUT DATA
    # -----------------------------
    output_data = ((monto > 500) & (hora < 5)).astype(int)

    output_data = pd.DataFrame({
        "fraude": output_data
    })

    # -----------------------------
    # RETORNO
    # -----------------------------
    return input_data, output_data
