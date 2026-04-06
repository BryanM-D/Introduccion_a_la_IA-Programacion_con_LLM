import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

def generar_caso_de_uso_transacciones(df):
    
    # -----------------------------
    # 1. Separar variables
    # -----------------------------
    X = df[["monto", "hora", "pais", "frecuencia_transacciones"]]
    y = df["fraude"]
    
    # -----------------------------
    # 2. Codificación categórica
    # -----------------------------
    X = pd.get_dummies(X, columns=["pais"], drop_first=True)
    
    # -----------------------------
    # 3. Train / test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # -----------------------------
    # 4. Modelo de clasificación
    # -----------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # -----------------------------
    # 5. Matriz de confusión
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)
    
    # -----------------------------
    # 6. F1-score
    # -----------------------------
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # -----------------------------
    # 7. Detección de desbalance
    # -----------------------------
    distribucion = y.value_counts(normalize=True).to_dict()
    
    desbalance = abs(distribucion.get(0, 0) - distribucion.get(1, 0)) > 0.3
    
    # -----------------------------
    # 8. Resultado final
    # -----------------------------
    return {
        "matriz_confusion": cm.tolist(),
        "f1_score": f1,
        "distribucion_clases": distribucion,
        "existe_desbalance": desbalance
    }
