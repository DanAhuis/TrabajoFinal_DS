import os
import pytest

from preprocess.cleaning import load_and_clean_data
from preprocess.preprocessing import encode_and_scale
from pipeline.processing import load_data, save_processed_data
from pipeline.training import train_and_evaluate


def create_sample_csv(path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "customerID": ["0001", "0002", "0003"],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "tenure": [1, 24, 5],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": [29.85, "", "269.3"],
        "Churn": ["No", "Yes", "No"],
    })
    df.to_csv(path, index=False)
    return df


def test_load_and_clean_data(tmp_path):
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "sample.csv"
    create_sample_csv(str(csv_path))

    df_clean = load_and_clean_data(str(csv_path))

    # customerID debe eliminarse y TotalCharges convertirse a numérico
    assert "customerID" not in df_clean.columns
    assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])
    # filas con TotalCharges vacías se eliminan -> en nuestro ejemplo 1 fila
    assert df_clean.shape[0] == 2


def test_encode_and_scale():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0],
        "B": ["x", "y", "x"],
        "Churn": ["No", "Yes", "No"],
    })

    X_processed, y, preprocessor, feature_names = encode_and_scale(df)

    # X_processed debe ser array con filas = 3
    assert X_processed.shape[0] == 3
    # y debe conservar las etiquetas originales
    assert list(y) == ["No", "Yes", "No"]
    # preprocessor debe exponer get_feature_names_out
    assert hasattr(preprocessor, "get_feature_names_out")


def test_load_and_save_processed_data(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    parquet_path = tmp_path / "out" / "test.parquet"

    # save_processed_data debe crear directorio y archivo parquet
    save_processed_data(df, str(parquet_path))
    assert parquet_path.exists()

    # load_data debe leer CSV; para probarlo creamos un CSV y usamos load_data
    csv_path = tmp_path / "in.csv"
    df.to_csv(str(csv_path), index=False)
    df_loaded = load_data(str(csv_path))
    assert df_loaded.equals(df)


def test_train_and_evaluate_synthetic():
    # Datos sintéticos para clasificación binaria
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    X = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])

    model = LogisticRegression()
    trained_model, metrics = train_and_evaluate(model, X, X, y, y, "LogReg")

    assert "Accuracy" in metrics
    assert metrics["Accuracy"] == 1.0
    # el modelo devuelto debe poder predecir
    assert (trained_model.predict(X) == y.values).all()


def test_end_to_end_pipeline(tmp_path):
    """Prueba de integración mínima.

    Flujo: carga -> limpieza -> preprocesamiento -> guardado.
    """
    pd = pytest.importorskip("pandas")
    # Crear CSV de entrada
    csv_path = tmp_path / "input.csv"
    create_sample_csv(str(csv_path))

    # 1. Carga y limpieza
    df_clean = load_and_clean_data(str(csv_path))
    assert "customerID" not in df_clean.columns

    # 2. Codificación y escalado
    X_processed, y, preprocessor, feature_names = encode_and_scale(df_clean)
    assert X_processed.shape[0] == df_clean.shape[0]

    # 3. Guardar resultado final (parquet o csv fallback)
    out_parquet = tmp_path / "out" / "prepared.parquet"
    df_final = pd.DataFrame(X_processed)
    df_final["Churn"] = y.values
    save_processed_data(df_final, str(out_parquet))
    # Aceptar parquet o fallback csv
    out_csv = str(out_parquet).replace(".parquet", ".csv")
    assert (
        out_parquet.exists()
        or os.path.exists(out_csv)
    )
