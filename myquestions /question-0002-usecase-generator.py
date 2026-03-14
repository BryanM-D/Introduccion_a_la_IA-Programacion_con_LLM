def generar_estudiantes(n=800, seed=42):
    np.random.seed(seed)
    
    promedio = np.random.uniform(2.0, 5.0, n)
    inasistencias = np.random.randint(0, 30, n)
    nivel = np.random.randint(1, 6, n)
    horas_estudio = np.random.normal(10, 5, n)
    
    riesgo = (promedio < 3).astype(int) + (inasistencias > 15).astype(int)
    deserta = (riesgo > 1).astype(int)
    
    df = pd.DataFrame({
        "promedio": promedio,
        "inasistencias": inasistencias,
        "nivel_socioeconomico": nivel,
        "horas_estudio": horas_estudio,
        "deserta": deserta
    })
    
    return df
