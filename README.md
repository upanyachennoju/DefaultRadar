### Credit Risk Intelligence Platform

Built an end-to-end machine learning system to predict loan default risk and deploy the model in a production-like environment.

#### Data Pipeline

* Created a data ingestion component to load and validate credit risk data.
* Defined a schema-driven validation system using YAML configuration files.
* Built a preprocessing pipeline for:

  * Missing value handling
  * Categorical feature encoding
  * Numerical feature transformation
* Saved transformed datasets and preprocessing artifacts for reproducibility.

#### Model Development

* Trained and compared multiple machine learning models:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Addressed class imbalance using:

  * `class_weight='balanced'`
  * `scale_pos_weight`
* Conducted threshold tuning instead of relying on the default 0.5 decision threshold.
* Evaluated models using:

  * Precision
  * Recall
  * F1 Score
  * ROC-AUC
  * PR-AUC
  * Balanced Accuracy
  * Specificity

#### Model Selection

* Compared all trained models on the same test set.
* Selected the best-performing model based on business-relevant metrics.
* Persisted the final model as a reusable artifact.

#### Experiment Tracking

* Integrated MLflow for:

  * Parameter tracking
  * Metric tracking
  * Experiment comparison
  * Model artifact storage

#### Explainability

* Added SHAP-based model explainability.
* Analyzed feature contributions and prediction behavior.
* Generated explanations for model decisions.

#### Deployment

* Developed a FastAPI REST API for real-time credit risk prediction.
* Implemented:

  * Input validation using Pydantic
  * Probability scoring
  * Risk classification (Low / Medium / High)
  * Health-check endpoints

#### Containerization

* Dockerized the complete application.
* Created a reproducible deployment environment containing:

  * API
  * Model artifacts
  * Preprocessing pipeline
  * Dependencies

#### Monitoring

* Implemented data drift monitoring using Evidently AI.
* Compared incoming data distributions against training data.
* Generated automated drift reports for model monitoring.

#### Final System Workflow

```text
Raw Data
   ↓
Data Validation
   ↓
Preprocessing Pipeline
   ↓
Model Training
   ↓
Model Evaluation
   ↓
MLflow Tracking
   ↓
Best Model Selection
   ↓
SHAP Explainability
   ↓
FastAPI Deployment
   ↓
Docker Container
   ↓
Drift Monitoring (Evidently)
```

#### Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* MLflow
* SHAP
* FastAPI
* Docker
* Evidently AI
* Git & GitHub

This project demonstrates the complete machine learning lifecycle—from data preprocessing and model development to deployment, explainability, experiment tracking, and production monitoring.
