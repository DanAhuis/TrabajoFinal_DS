# 📊 Trabajo Final – Metodología para Data Science

## 🎯 Proyecto
**Predicción de Churn en Telecomunicaciones con Machine Learning**  

Este proyecto aplica la metodología de Data Science para predecir la **fuga de clientes (churn)** en el sector de telecomunicaciones, utilizando el dataset **Telco Customer Churn (IBM Sample Data Sets)**.  

---

## 📂 Estructura del repositorio

TrabajoFinal-DS/
├─ .github/              # Workflows de CI/CD (opcional, del curso)
├─ data/                 # datasets (ignorado en git)
│   └─ Telco-Customer-Churn.csv (local, no subir)
├─ docs/                 # Documentación
│   ├─ GUIA_TRABAJO_FINAL.md
│   ├─ Trabajo Parcial.docx
│   └─ report_final.md   # tu informe final
├─ notebooks/            
│   └─ churn_final.ipynb # notebook principal
├─ src/                  # código fuente modular
│   ├─ preprocessing.py
│   ├─ modeling.py
│   └─ evaluation.py
├─ tests/                # tests unitarios e integración
│   └─ test_pipeline.py
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ pyproject.toml        # (opcional, si sigues el estándar del curso)
└─ pytest.ini            # (para validaciones automáticas)


---

## 📑 Descripción del Problema

La **pérdida de clientes (churn)** es un problema crítico para empresas de telecomunicaciones.  
Retener clientes existentes resulta más económico que adquirir nuevos, por lo que predecir quiénes tienen mayor probabilidad de abandonar el servicio permite **diseñar estrategias de fidelización** más efectivas.  

- **Hipótesis**: Analizando variables demográficas, de facturación y de uso de servicios, es posible predecir con una precisión ≥ 80% qué clientes tienen riesgo de churn.  
- **Métricas de éxito**:  
  - Accuracy ≥ 80%  
  - Recall ≥ 70% en la clase churn  
  - ROC-AUC ≥ 0.85  

---

## 🛠️ Metodología (CRISP-DM adaptado)

1. **Recopilación de datos**: Dataset *Telco Customer Churn* (IBM, Kaggle).  
2. **Preprocesamiento**:  
   - Imputación de valores faltantes.  
   - Codificación One-Hot de variables categóricas.  
   - Escalado de variables numéricas.  
   - Eliminación de duplicados y corrección de outliers.  
3. **Análisis exploratorio**:  
   - Distribución de churn (≈ 26%).  
   - Relación entre tipo de contrato y churn.  
   - Relación entre cargos mensuales y churn.  
4. **Modelado**:  
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
5. **Evaluación**:  
   - Accuracy, Recall, ROC-AUC  
   - Matriz de confusión y clasificación  
   - Comparación de modelos  

---

## 📊 Resultados Obtenidos

### 🏆 Mejor Modelo: RandomForest
| Métrica | Valor |
|---------|-------|
| **Accuracy** | **95.73%** |
| **Precision** | **94.07%** |
| **Recall** | **89.66%** |
| **F1-Score** | **91.81%** |
| **ROC-AUC** | **93.80%** |

### 📈 Comparación de Modelos
| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| RandomForest | **95.73%** | **94.07%** | **89.66%** | **91.81%** | **93.80%** |
| XGBoost | 91.22% | 85.60% | 80.67% | 83.06% | 87.86% |
| LogisticRegression | 80.32% | 65.74% | 54.89% | 59.83% | 72.24% |

### ✅ Objetivos Alcanzados
- ✅ **Accuracy ≥ 80%**: RandomForest alcanzó 95.73%
- ✅ **Recall ≥ 70%**: RandomForest alcanzó 89.66%
- ✅ **ROC-AUC ≥ 0.85**: RandomForest alcanzó 0.938

---

## 🚀 API FastAPI

El proyecto incluye una **API REST** para realizar predicciones en tiempo real.

### 📡 Endpoints Disponibles

#### 1. Health Check
```http
GET /health
```
**Respuesta:**
```json
{
  "status": "ok"
}
```

#### 2. Predicción de Churn
```http
POST /predict
Content-Type: application/json
```
**Ejemplo de request:**
```json
[
  {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }
]
```

**Respuesta:**
```json
{
  "predictions": ["Yes"],
  "probabilities": [0.6235961317624209]
}
```

### 🔧 Ejecutar la API

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar API
uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# Acceder a documentación interactiva
# http://127.0.0.1:8000/docs
```

---

## ▶️ Reproducir el proyecto

### 1. Clonar repositorio
```bash
git clone https://github.com/DanAhuis/TrabajoFinal_DS.git
cd TrabajoFinal_DS
