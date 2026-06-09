# Credit Risk Intelligence Platform

An end-to-end machine learning system for predicting loan default risk with production-ready deployment, experiment tracking, explainability, and monitoring.

## Overview

This project demonstrates a complete ML lifecycle implementation:
- **Data Pipeline**: Data ingestion and preprocessing with schema validation
- **Model Development**: Multi-model training (Logistic Regression, Random Forest, XGBoost) with class imbalance handling
- **Experiment Tracking**: MLflow integration for reproducibility and model comparison
- **Explainability**: SHAP-based feature importance analysis for model interpretability
- **Deployment**: FastAPI REST API with real-time credit risk scoring
- **Containerization**: Docker support for easy deployment
- **Monitoring**: Data drift detection using Evidently AI

## Quick Start

### Prerequisites
- Python 3.12+
- Conda or pip
- Docker (optional, for containerization)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd credit-risk-system
```

2. Create and activate a virtual environment:
```bash
conda create -n mlops python=3.12
conda activate mlops
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Pipeline

**Train models and log experiments:**
```bash
python src/pipeline/train_pipeline.py
```

**View MLflow UI:**
```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

**Launch the API server:**
```bash
python -m uvicorn api.main:app --reload
# API will be available at http://localhost:8000
```

### Docker Deployment

Build and run the containerized application:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

## Project Structure

```
credit-risk-system/
├── api/                          # FastAPI application
│   └── main.py                  # API endpoints and prediction logic
├── src/
│   ├── components/              # Core pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py    # Data loading and validation
│   │   ├── data_preprocessing.py # Feature engineering and transformation
│   │   ├── model_trainer.py      # Model training with hyperparameters
│   │   └── model_evaluation.py   # Model evaluation metrics
│   ├── pipeline/                # End-to-end pipelines
│   │   ├── __init__.py
│   │   ├── train_pipeline.py    # Training and MLflow logging
│   │   └── predict_pipeline.py  # Batch prediction
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── common.py
├── config/                       # Configuration files
│   ├── config.yaml              # Model and training configuration
│   └── schema.yaml              # Data schema definition
├── data/
│   ├── raw/                     # Original dataset (synthetic_credit_risk.csv)
│   └── processed/               # Processed and transformed data
├── artifacts/                    # Model artifacts and outputs
│   ├── models/                  # Trained model pickles
│   ├── preprocessors/           # Data preprocessing artifacts
│   ├── transformed_data/        # Training/test data arrays
│   │   ├── X_train.npy
│   │   ├── X_test.npy
│   │   ├── y_train.npy
│   │   └── y_test.npy
│   └── reports/                 # Model evaluation and drift reports
├── monitoring/                   # Data drift monitoring
│   ├── drift_report.py          # Drift detection using Evidently
│   └── reports/
├── notebooks/                    # Jupyter notebooks
│   ├── eda.ipynb                # Exploratory data analysis
│   └── exp_notebook.ipynb       # Experiment notebook
├── mlruns/                       # MLflow experiment tracking
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Key Features

### Data Pipeline

The data ingestion and preprocessing pipeline handles:
- **Data Loading**: Loads synthetic credit risk dataset from CSV
- **Validation**: Checks for duplicates, missing values, and schema compliance
- **Feature Set** (12 features):
  - `age`: Customer age
  - `monthly_income`: Monthly income in currency units
  - `debt_ratio`: Total debt to income ratio
  - `credit_utilization`: Credit card utilization percentage
  - `transaction_count_30d`: Number of transactions in last 30 days
  - `avg_transaction_amount`: Average transaction size
  - `employment_type`: Categorical employment sector
  - `education_level`: Categorical education level
  - `region`: Geographic region
  - `device_type`: Device used for transactions
  - `last_payment_delay_days`: Days since last payment
  - `internal_score_v2`: Internal risk score

- **Preprocessing**:
  - Numeric missing values: Median imputation
  - Categorical missing values: Most frequent value imputation
  - Numeric scaling: StandardScaler
  - Categorical encoding: OneHotEncoder
- **Data Splitting**: 80-20 train-test split with stratification (random_state: 42)

### Model Development

Three machine learning models are trained with hyperparameter tuning:

#### 1. Logistic Regression
- **Algorithm**: GridSearchCV
- **Hyperparameters**:
  - C (regularization strength): 0.1
  - Solver: saga
  - Class weight: balanced
  - Max iterations: 200
- **Decision Threshold**: 0.80 (tuned for precision-recall balance)
- **Best Params Search**: Over C values [0.001, 0.01, 0.1, 1, 10, 100]

#### 2. Random Forest
- **Algorithm**: RandomizedSearchCV (50 iterations)
- **Hyperparameters**:
  - n_estimators: 172
  - max_depth: 5
  - max_features: None (auto)
  - min_samples_leaf: 3
  - min_samples_split: 2
  - Class weight: balanced
- **Decision Threshold**: 0.5
- **Primary Optimization**: Recall metric

#### 3. XGBoost
- **Algorithm**: RandomizedSearchCV (50 iterations)
- **Hyperparameters**:
  - n_estimators: 300
  - max_depth: 4
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8
  - scale_pos_weight: Auto-calculated for class imbalance
- **Decision Threshold**: 0.3 (optimized for recall)
- **Evaluation Metric**: Logloss

### Class Imbalance Handling

- **Logistic Regression**: `class_weight='balanced'`
- **Random Forest**: `class_weight='balanced'`
- **XGBoost**: `scale_pos_weight` (auto-calculated as neg_count / pos_count)

### Model Evaluation

Models are evaluated on the test set using:
- **Primary Metric**: Recall (minimize false negatives - missed defaults)
- **Secondary Metrics**:
  - Precision
  - F1 Score
  - ROC-AUC
  - Specificity (true negative rate)

**Threshold Optimization**:
- Strategy: Precision-Recall curve analysis
- Minimum Recall Target: 70% (catch at least 70% of defaults)
- Maximum False Positive Rate: 30%

### Experiment Tracking with MLflow

MLflow is integrated for reproducibility:
- **Tracking URI**: Local file-based (./mlruns)
- **Experiment Name**: credit-risk-models
- **Logged Artifacts**:
  - Model parameters
  - Evaluation metrics
  - Trained model pickle files
  - Preprocessor artifacts
- **Tags**: Project, environment, version information
- **Model Registry**: Enabled for model versioning

### Explainability

- **SHAP** (SHapley Additive exPlanations) for feature importance
- **LinearExplainer** for model interpretation
- Feature contribution analysis for each prediction
- Per-instance explanation available via API

### API Endpoints

FastAPI server (`api/main.py`) provides the following endpoints:

#### 1. Health Check
```
GET /health
```
Returns API health status and any startup errors.

Response:
```json
{
  "status": "ok"
}
```

#### 2. Predictions
```
POST /predict
```
Accepts credit application data and returns risk prediction with explainability.

Request Body:
```json
{
  "age": 35,
  "monthly_income": 5000,
  "debt_ratio": 0.35,
  "credit_utilization": 0.65,
  "transaction_count_30d": 15,
  "avg_transaction_amount": 250,
  "employment_type": "Full-time",
  "education_level": "Bachelor",
  "region": "North",
  "device_type": "Mobile",
  "last_payment_delay_days": 0,
  "internal_score_v2": 0.75
}
```

Response:
```json
{
  "default_probability": 0.25,
  "prediction": 0,
  "risk_tier": "low",
  "explainability": {
    "age": 0.05,
    "monthly_income": -0.08,
    "debt_ratio": 0.12,
    ...
  }
}
```

**Risk Tiers**:
- `low`: Probability < 0.3
- `medium`: Probability 0.3-0.7
- `high`: Probability >= 0.7

**Decision Threshold**: 0.8 (configurable)

### Containerization

- **Base Image**: Python 3.12-slim
- **Port**: 8000
- **Entrypoint**: Uvicorn server
- **Docker Compose**: Single service (credit-risk-api)

### Monitoring & Drift Detection

Implemented using **Evidently AI**:
- Generates data drift reports comparing reference and current data distributions
- Detects feature drift and statistical changes
- HTML report generation for visualization
- Available at: `monitoring/drift_report.py`

## Configuration

All settings are defined in `config/config.yaml`:

```yaml
data:
  raw_path: "data/raw/synthetic_credit_risk.csv"
  processed_path: "data/processed/ingested_data.csv"

preprocessing:
  test_size: 0.2
  random_state: 42
  stratified: true

mlflow:
  enabled: true
  tracking_uri: "file:./mlruns"
  experiment_name: "credit-risk-models"

training:
  random_state: 42
  cv_folds: 5
  n_jobs: -1
  class_weight: "balanced"

evaluation:
  primary_metric: "recall"
  threshold_optimization:
    enabled: true
    min_recall: 0.70
    max_false_positive_rate: 0.30
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Python | 3.12 |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-Learn, XGBoost |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| API Framework | FastAPI, Pydantic |
| Server | Uvicorn |
| Containerization | Docker, Docker Compose |
| Monitoring | Evidently AI |
| Version Control | Git & GitHub |

## API Documentation

Once the API is running, interactive documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Usage Example

### Python Client
```python
import requests

payload = {
    "age": 35,
    "monthly_income": 5000,
    "debt_ratio": 0.35,
    "credit_utilization": 0.65,
    "transaction_count_30d": 15,
    "avg_transaction_amount": 250,
    "employment_type": "Full-time",
    "education_level": "Bachelor",
    "region": "North",
    "device_type": "Mobile",
    "last_payment_delay_days": 0,
    "internal_score_v2": 0.75
}

response = requests.post("http://localhost:8000/predict", json=payload)
prediction = response.json()
print(f"Default Probability: {prediction['default_probability']}")
print(f"Risk Tier: {prediction['risk_tier']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "monthly_income": 5000,
    "debt_ratio": 0.35,
    "credit_utilization": 0.65,
    "transaction_count_30d": 15,
    "avg_transaction_amount": 250,
    "employment_type": "Full-time",
    "education_level": "Bachelor",
    "region": "North",
    "device_type": "Mobile",
    "last_payment_delay_days": 0,
    "internal_score_v2": 0.75
  }'
```

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
