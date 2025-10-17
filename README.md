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

## 📊 Resultados esperados

- Determinar el modelo con mejor desempeño predictivo.  
- Identificar las variables que más influyen en la fuga de clientes.  
- Proponer **estrategias de retención basadas en los hallazgos**.  

---

## ▶️ Reproducir el proyecto

### 1. Clonar repositorio
```bash
git clone https://github.com/DanAhuis/TrabajoFinal-DS.git
cd TrabajoFinal-DS
