def generar_caso_de_uso_transacciones(n=1500, seed=42):
    import numpy as np
    import pandas as pd
    
    np.random.seed(seed)
    
    monto = np.random.exponential(200, n)
    hora = np.random.randint(0, 24, n)
    pais = np.random.randint(0, 5, n)
    frecuencia = np.random.poisson(3, n)
    
    fraude = ((monto > 500) & (hora < 5)).astype(int)
    
    df = pd.DataFrame({
        "monto": monto,
        "hora": hora,
        "pais": pais,
        "frecuencia_transacciones": frecuencia
    })
    
    y = fraude
    
    return df, y
