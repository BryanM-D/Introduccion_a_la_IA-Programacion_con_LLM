import numpy as np
import pandas as pd

def generar_caso_de_uso_transacciones(n=1500, seed=42):
    
    np.random.seed(seed)
    
    # -----------------------------
    # 1. Variables simuladas
    # -----------------------------
    monto = np.random.exponential(200, n)
    hora = np.random.randint(0, 24, n)
    pais = np.random.randint(0, 5, n)
    frecuencia_transacciones = np.random.poisson(3, n)
    
    # -----------------------------
    # 2. Variable objetivo
    # -----------------------------
    fraude = ((monto > 500) & (hora < 5)).astype(int)
    
    # -----------------------------
    # 3. FEATURES (DEBE SER DICCIONARIO)
    # -----------------------------
    X = {
        "monto": monto,
        "hora": hora,
        "pais": pais,
        "frecuencia_transacciones": frecuencia_transacciones
    }
    
    # -----------------------------
    # 4. TARGET
    # -----------------------------
    y = fraude
    
    # -----------------------------
    # 5. RETORNO CORRECTO
    # -----------------------------
    return X, y
