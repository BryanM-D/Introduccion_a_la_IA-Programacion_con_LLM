import numpy as np
import pandas as pd

def generar_caso_de_uso_creditos(n=1000, seed=42):
    np.random.seed(seed)
    
    edad = np.random.randint(18, 65, n)
    ingresos = np.random.normal(3000, 1000, n)
    puntaje = np.random.normal(600, 100, n)
    genero = np.random.randint(0, 2, n)
    
    prob = (puntaje/850) + (ingresos/10000)
    aprobado = (prob > 0.9).astype(int)
    
    df = pd.DataFrame({
        "edad": edad,
        "ingresos": ingresos,
        "puntaje_credito": puntaje,
        "genero": genero,
        "aprobado": aprobado
    })
    
    return df
