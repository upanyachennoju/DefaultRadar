import logging
import sys
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> Path:
	p = Path(path)
	if p.is_absolute() and p.exists():
		return p

	current = Path.cwd()
	while current != current.parent:
		if (current / "config").exists():
			candidate = current / p
			if candidate.exists():
				return candidate
		current = current.parent

	if p.exists():
		return p.resolve()

	raise FileNotFoundError(f"File not found: {path}")


def load_pickle(path: str):
	path = _resolve_path(path)
	with open(path, 'rb') as f:
		obj = pickle.load(f)
	logger.info(f"Loaded pickle from {path}")
	return obj


def load_config(config_path: str = "config/config.yaml") -> dict:
	try:
		path = _resolve_path(config_path)
		with open(path, 'r') as f:
			cfg = yaml.safe_load(f)
		logger.info(f"Loaded config from {path}")
		return cfg
	except Exception as e:
		logger.warning(f"Failed to load config {config_path}: {e}")
		return {}


def prepare_features(df: pd.DataFrame, preprocessor) -> np.ndarray:
	if preprocessor is None:
		raise ValueError("Preprocessor must not be None")

	try:
		X_transformed = preprocessor.transform(df)
		return X_transformed
	except Exception as e:
		logger.error(f"Error transforming input features: {e}")
		raise


def predict_from_df(
	df: pd.DataFrame,
	model_path: str = "artifacts/models/best_model.pkl",
	preprocessor_path: str = "artifacts/preprocessors/preprocessing.pkl",
	threshold: Optional[float] = None
):
	"""Return dataframe with `prediction_proba` and `prediction` columns.

	Args:
		df: Raw feature dataframe (no target column expected)
		model_path: Path to saved model pickle
		preprocessor_path: Path to saved preprocessor pickle
		threshold: Optional probability threshold for binary classification

	Returns:
		pd.DataFrame: Input dataframe with two added columns: `prediction_proba`, `prediction`
	"""
	model = load_pickle(model_path)
	preprocessor = load_pickle(preprocessor_path)

	X = prepare_features(df.copy(), preprocessor)

	# Ensure model supports predict_proba
	if not hasattr(model, 'predict_proba'):
		logger.error('Loaded model does not support `predict_proba`')
		raise AttributeError('Model has no method `predict_proba`')

	proba = model.predict_proba(X)[:, 1]
	if threshold is None:
		preds = model.predict(X)
	else:
		preds = (proba >= float(threshold)).astype(int)

	out = df.copy()
	out['prediction_proba'] = proba
	out['prediction'] = preds
	return out


def predict_from_csv(
	csv_path: str,
	output_path: Optional[str] = None,
	model_path: str = "artifacts/models/best_model.pkl",
	preprocessor_path: str = "artifacts/preprocessors/preprocessing.pkl",
	threshold: Optional[float] = None
):
	csv_path = _resolve_path(csv_path)
	df = pd.read_csv(csv_path)
	logger.info(f"Loaded input CSV {csv_path} with shape {df.shape}")

	result_df = predict_from_df(df, model_path=model_path, preprocessor_path=preprocessor_path, threshold=threshold)

	if output_path:
		outp = Path(output_path)
		outp.parent.mkdir(parents=True, exist_ok=True)
		result_df.to_csv(outp, index=False)
		logger.info(f"Predictions saved to {outp}")

	return result_df


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Run prediction pipeline on CSV input")
	parser.add_argument('input_csv', help='Path to input CSV with feature columns')
	parser.add_argument('--output', '-o', help='Path to save predictions CSV', default=None)
	parser.add_argument('--model', help='Path to model pickle', default='artifacts/models/best_model.pkl')
	parser.add_argument('--preprocessor', help='Path to preprocessor pickle', default='artifacts/preprocessors/preprocessing.pkl')
	parser.add_argument('--threshold', help='Optional probability threshold (float)', default=None)

	args = parser.parse_args()

	result = predict_from_csv(
		args.input_csv,
		output_path=args.output,
		model_path=args.model,
		preprocessor_path=args.preprocessor,
		threshold=args.threshold
	)

	print(result.head().to_string(index=False))