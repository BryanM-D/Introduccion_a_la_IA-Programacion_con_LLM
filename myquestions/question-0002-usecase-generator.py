import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def generar_caso_de_uso_estudiantes(df):
    
    # -----------------------------
    # 1. Limpieza de datos faltantes
    # -----------------------------
    df = df.dropna()
    
    # -----------------------------
    # 2. Separar variables
    # -----------------------------
    X = df[[
        "promedio",
        "inasistencias",
        "nivel_socioeconomico",
        "horas_estudio"
    ]]
    
    y = df["deserta"]
    
    # -----------------------------
    # 3. Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # -----------------------------
    # 4. Estandarización
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -----------------------------
    # 5. Modelo
    # -----------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    # -----------------------------
    # 6. Métricas
    # -----------------------------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0)
    }
    
    # -----------------------------
    # 7. Importancia de variables
    # -----------------------------
    importance = pd.Series(
        model.coef_[0],
        index=X.columns
    ).sort_values(ascending=False)
    
    # -----------------------------
    # 8. Retorno final
    # -----------------------------
    return {
        "metricas": metrics,
        "importancia_variables": importance.to_dict()
    }
