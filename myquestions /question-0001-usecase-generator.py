import numpy as np
import pandas as pd

def generar_datos_sensores(n=500, seed=42):
    np.random.seed(seed)
    
    temperatura = np.random.normal(75, 10, n)
    vibracion = np.random.normal(5, 2, n)
    presion = np.random.normal(200, 20, n)
    ruido = np.random.normal(80, 5, n)
    horas_operacion = np.random.randint(0, 10000, n)
    
    # Generamos probabilidad de falla basada en condiciones extremas
    prob_falla = (
        (temperatura > 90).astype(int) +
        (vibracion > 8).astype(int) +
        (presion > 240).astype(int)
    )
    
    estado_falla = (prob_falla > 1).astype(int)
    
    df = pd.DataFrame({
        "temperatura": temperatura,
        "vibracion": vibracion,
        "presion": presion,
        "ruido": ruido,
        "horas_operacion": horas_operacion,
        "estado_falla": estado_falla
    })
    
    return df
