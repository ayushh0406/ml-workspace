# Student Performance – End-to-End ML Project

Predict student performance using an end‑to‑end machine learning pipeline with robust EDA, feature engineering, model training, evaluation, and inference. The project is modular, config‑driven, and production‑friendly.

---

## 📑 Table of Contents
- About
- Dataset
- Key Insights (EDA)
- Features Engineered
- Project Structure
- Quick Start
- Configuration
- Workflow
- How to Run
- Results & Metrics
- Model Explainability
- Logging & Errors
- Testing
- Docker (optional)
- CI/CD (optional)
- Roadmap


---

## 📖 About
This repository implements a complete ML workflow on a “Student Performance” dataset to predict academic outcomes (e.g., final grade or pass/fail). It includes reproducible EDA, preprocessing pipelines, model training and evaluation, and a simple prediction interface for new data.

---

## 📦 Dataset
Common columns in student performance datasets include:
- Demographics: gender, age, parental education
- Academic context: study time, absences, prior scores
- Social factors: internet, family support, extracurriculars
- Target: final grade (continuous) or pass/fail (binary)

If using a public dataset, cite the source (e.g., UCI Student Performance). If using a private dataset, describe collection and privacy considerations.

---

## 🔎 Key Insights (EDA)
Highlights typically observed in Student Performance datasets:
- Strongest correlations often occur between previous period scores and final grades (e.g., G1/G2 with G3).  
- Absences, low study time, and lack of parental education tend to associate with lower performance.  
- Categorical factors like internet access and school support may interact with study time and prior scores.  

Example EDA steps performed:
- Data quality checks: missing values, duplicates, outliers
- Univariate distributions of key numeric features (scores, study time, absences)
- Bivariate plots: target vs. study time, target vs. absences
- Correlation heatmap; top correlated features with target
- Category impact analysis: mean target by parental education, school support, internet

Artifacts produced:
- eda/figures/*.png (histograms, boxplots, heatmaps)
- eda/summary.md (observations and decisions)

---

## 🧮 Features Engineered
- Numeric imputation: median for continuous features
- Categorical imputation: most frequent category
- Encoding: one‑hot for nominal categories
- Scaling: standard scaling for numeric features
- Optional domain features:
  - prior_avg = mean of previous term scores
  - attendance_flag = absences thresholded
  - study_effort = study_time normalized by prior_avg
  - interaction terms for key categorical × numeric pairs

---

## 📂 Project Structure
```

student-performance-ml/
├─ config/
│  ├─ config.yaml          \# data paths, artifacts, schema
│  └─ params.yaml          \# split, preprocessing, model hyperparams
├─ data/                   \# raw \& sample data (git-ignored for large files)
├─ artifacts/              \# generated outputs (created at runtime)
├─ eda/
│  ├─ figures/             \# EDA plots
│  └─ summary.md           \# EDA findings
├─ notebooks/              \# EDA \& research notebooks
├─ src/
│  └─ mlproject/
│     ├─ components/
│     │  ├─ data_ingestion.py
│     │  ├─ data_validation.py
│     │  ├─ data_transformation.py
│     │  ├─ model_trainer.py
│     │  └─ model_evaluation.py
│     ├─ pipeline/
│     │  ├─ training_pipeline.py
│     │  └─ prediction_pipeline.py
│     ├─ entity/           \# dataclasses for configs \& artifacts
│     ├─ config/           \# config loader utilities
│     ├─ constants/        \# filenames, keys, defaults
│     ├─ logger.py         \# centralized logging
│     └─ exception.py      \# custom exception wrapper
├─ tests/                  \# unit/integration tests (optional)
├─ requirements.txt
├─ setup.py
└─ README.md

```

---

## ⚡ Quick Start
```


# Create \& activate env

python -m venv .venv

# Linux/Mac

source .venv/bin/activate

# Windows (PowerShell)

.\.venv\Scripts\Activate.ps1

# Install

pip install -r requirements.txt
pip install -e .

```

---

## ⚙️ Configuration
- config/config.yaml
  - raw_data_path, processed directories, artifacts base
  - schema: required columns, dtypes, target name
- config/params.yaml
  - data.split: test_size, random_state, stratify (if classification)
  - preprocessing: numeric_imputer, categorical_imputer, encoder, scaler
  - model: type (e.g., RandomForestRegressor/Classifier, XGBoost), hyperparams

Example params.yaml (regression):
```

data:
target: final_grade
test_size: 0.2
random_state: 42

preprocessing:
numeric_imputer: median
categorical_imputer: most_frequent
encoder: onehot
scaler: standard

model:
type: RandomForestRegressor
n_estimators: 400
max_depth: 16
min_samples_leaf: 2
n_jobs: -1

```

---

## 🔄 Workflow
1) Data Ingestion
- Load CSV(s), perform train/test split, save:
  - artifacts/raw/
  - artifacts/train.csv
  - artifacts/test.csv

2) Data Validation
- Schema check: required columns, dtypes
- Missing & duplicate scans, basic ranges for numeric fields

3) Data Transformation
- ColumnTransformer pipelines:
  - Numeric: impute → scale
  - Categorical: impute → one‑hot encode
- Optional: feature engineering (domain features)
- Persist preprocessor: artifacts/preprocessor.pkl

4) Model Training
- Train candidate models per params
- Optional hyperparameter search
- Persist best model: artifacts/model.pkl

5) Model Evaluation
- Regression: RMSE, MAE, R²
- Classification (if using pass/fail): Accuracy, F1, ROC‑AUC
- Save metrics: artifacts/metrics.json
- Save evaluation plots (e.g., residuals, calibration): artifacts/reports/

6) Inference
- Batch prediction on new CSV or JSON
- Outputs predictions to file and/or stdout

---

## ▶️ How to Run
Full pipeline:
```

python -m src.mlproject.pipeline.training_pipeline

```

Stage‑wise (debugging):
```

python -m src.mlproject.components.data_ingestion
python -m src.mlproject.components.data_validation
python -m src.mlproject.components.data_transformation
python -m src.mlproject.components.model_trainer
python -m src.mlproject.components.model_evaluation

```

Inference:
```

python -m src.mlproject.pipeline.prediction_pipeline \
--input data/new_students.csv \
--output artifacts/predictions.csv

```

---

## 📊 Results & Metrics
- artifacts/metrics.json:
  - Regression: {"rmse": ..., "mae": ..., "r2": ...}
  - Classification: {"accuracy": ..., "f1": ..., "roc_auc": ...}
- artifacts/reports/: residual/error plots, confusion matrix (if classification)
- eda/summary.md: key insights guiding feature engineering and model choices

---

## 🧠 Model Explainability
- Feature importance from tree‑based models (saved under artifacts/reports/)
- Optional SHAP summary plots for global and local explanations
- Partial dependence or permutation importance for robust insights

---

## 🧾 Logging & Errors
- Centralized logs with timestamps and levels (src/mlproject/logger.py)
- Custom exception wrapper adds context for faster debugging (src/mlproject/exception.py)

---

## ✅ Testing
```

pip install pytest pytest-cov
pytest -q
pytest --cov=src --cov-report=term-missing

```

---

## 🐳 Docker (optional)
```

docker build -t student-perf-ml:latest .
docker run --rm -v "\$(pwd)/artifacts:/app/artifacts" student-perf-ml:latest \
python -m src.mlproject.pipeline.training_pipeline

```

---

## 🔁 CI/CD (optional)
- Add .github/workflows/ci.yml:
  - Install, lint, test, cache deps
  - Optional: build Docker, upload artifacts

---

## 🗺️ Roadmap
- MLflow tracking & model registry
- DVC for data/versioned pipelines
- FastAPI/Streamlit service for real‑time predictions
- Cross‑validation and automated hyperparameter tuning
- Fairness metrics and bias checks

---
## 🖥️ Run locally (PowerShell)

If you're on Windows PowerShell, use the following commands from the project root:

```powershell
# Create virtual environment
python -m venv .venv

# Activate the virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# (Optional) install package in editable mode
pip install -e .

# Run unit tests
pytest -q

# Run full training pipeline
python -m src.pipelines.train_pipeline
```

Tips:
- If PowerShell blocks script execution, enable it temporarily using:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

- Use `artifacts/` directory to find models, plots, and reports.

