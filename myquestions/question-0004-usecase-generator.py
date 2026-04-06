def generar_caso_de_uso_edificios(n=700, seed=42):
    np.random.seed(seed)
    
    consumo = np.random.normal(300, 50, n)
    area = np.random.normal(120, 30, n)
    anio = np.random.randint(1980, 2023, n)
    aislamiento = np.random.randint(0, 2, n)
    
    categoria = np.where(consumo < 250, "A",
                 np.where(consumo < 320, "B", "C"))
    
    df = pd.DataFrame({
        "consumo_kwh": consumo,
        "area_m2": area,
        "anio_construccion": anio,
        "aislamiento": aislamiento,
        "categoria_eficiencia": categoria
    })
    
    return df
