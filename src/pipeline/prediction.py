import pandas as pd
import joblib

# Cargar modelo y preprocesador
preprocessor = joblib.load("src/models/preprocessor.pkl")
model = joblib.load("src/models/RandomForest.pkl")

nuevo_cliente = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.2,
    "TotalCharges": 400.3
}])

# Aplicar preprocesamiento
nuevo_cliente_transf = preprocessor.transform(nuevo_cliente)

# Convertir a DataFrame con nombres de columnas (si están disponibles)
try:
    feature_names = preprocessor.get_feature_names_out()
    nuevo_cliente_transf = pd.DataFrame(
        nuevo_cliente_transf, columns=feature_names
    )
except Exception:
    # Si no se puede obtener nombres de features, continuar sin cambiar
    pass

# Predicción
pred = model.predict(nuevo_cliente_transf)
prob = model.predict_proba(nuevo_cliente_transf)[:, 1]

print(f"Predicción (Churn): {int(pred[0])}")
print(f"Probabilidad de churn: {prob[0]:.2%}")
