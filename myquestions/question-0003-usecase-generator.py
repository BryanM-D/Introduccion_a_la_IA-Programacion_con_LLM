def generar_caso_de_uso_transacciones(n=1500, seed=42):
    np.random.seed(seed)
    
    monto = np.random.exponential(200, n)
    hora = np.random.randint(0, 24, n)
    pais = np.random.randint(0, 5, n)
    frecuencia = np.random.poisson(3, n)
    
    riesgo = (monto > 500).astype(int) + (hora < 5).astype(int)
    fraude = (riesgo > 1).astype(int)
    
    df = pd.DataFrame({
        "monto": monto,
        "hora": hora,
        "pais": pais,
        "frecuencia_transacciones": frecuencia,
        "fraude": fraude
    })
    
    X = df.drop(columns=["fraude"])
    y = df["fraude"]
    
    return X, y
