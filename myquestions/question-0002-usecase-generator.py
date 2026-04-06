def generar_caso_de_uso_estudiantes(n=800, seed=42):
    import numpy as np
    
    np.random.seed(seed)
    
    promedio = np.random.normal(3.0, 0.5, n)
    inasistencias = np.random.randint(0, 30, n)
    nivel_socioeconomico = np.random.randint(1, 4, n)
    horas_estudio = np.random.normal(5, 2, n)
    
    deserta = (
        (promedio < 2.8) | 
        (inasistencias > 15) |
        (horas_estudio < 3)
    ).astype(int)
    
    df = {
        "promedio": promedio,
        "inasistencias": inasistencias,
        "nivel_socioeconomico": nivel_socioeconomico,
        "horas_estudio": horas_estudio
    }
    
    return df, deserta
