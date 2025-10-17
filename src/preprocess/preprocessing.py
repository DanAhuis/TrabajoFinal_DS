from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

def encode_and_scale(df):
    """Codifica variables categóricas y escala variables numéricas, devolviendo el preprocesador completo."""

    # Separar X e y
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Identificar tipos de columnas
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Definir transformadores
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combinar transformadores en un solo preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical)
        ]
    )

    # Ajustar y transformar los datos
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    return X_processed, y, preprocessor, feature_names
