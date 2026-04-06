import numpy as np
import pandas as pd

def generar_caso_de_uso_creditos(n=1000, seed=42):
    np.random.seed(seed)
    
    # -----------------------------
    # 1. Variables simuladas
    # -----------------------------
    edad = np.random.randint(18, 65, n)
    ingresos = np.random.normal(3000, 1000, n)
    puntaje_credito = np.random.normal(600, 100, n)
    genero = np.random.randint(0, 2, n)  # 0 mujer, 1 hombre
    
    # -----------------------------
    # 2. Regla de aprobación
    # -----------------------------
    prob = (puntaje_credito / 850) + (ingresos / 10000)
    aprobado = (prob > 0.9).astype(int)
    
    # -----------------------------
    # 3. FEATURES (DEBE SER DICCIONARIO)
    # -----------------------------
    X = {
        "edad": edad,
        "ingresos": ingresos,
        "puntaje_credito": puntaje_credito,
        "genero": genero
    }
    
    # -----------------------------
    # 4. TARGET
    # -----------------------------
    y = aprobado
    
    # -----------------------------
    # 5. RETORNO CORRECTO
    # -----------------------------
    return X, y
