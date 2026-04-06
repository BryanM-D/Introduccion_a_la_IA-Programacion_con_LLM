def generar_caso_de_uso_estudiantes(n=800, seed=42):
    np.random.seed(seed)
    
    horas_estudio = np.random.normal(5, 2, n)
    asistencia = np.random.normal(80, 10, n)
    edad = np.random.randint(17, 30, n)
    
    aprobado = ((horas_estudio > 5) & (asistencia > 75)).astype(int)
    
    df = pd.DataFrame({
        "horas_estudio": horas_estudio,
        "asistencia": asistencia,
        "edad": edad,
        "aprobado": aprobado
    })
    
    X = df.drop(columns=["aprobado"])
    y = df["aprobado"]
    
    return X, y
