from datetime import datetime
import json
import pickle
from typing import Any, Dict, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    MODELS_DIR,
    PIPELINE_CONFIG,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from src.features import ChurnFeatureEngineer
from src.preprocessor import Preprocessor
from src.target_encoder import TargetEncoder


def create_parameter_grids() -> Dict[str, Dict[str, list]]:
    """
    Create parameter grids for different levels of hyperparameter tuning.

    Returns:
    --------
    Dict containing different parameter grids for quick, standard, and comprehensive searches
    """

    quick_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", "log2"],
        "model__bootstrap": [True],
        "model__random_state": [42],
    }

    standard_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [10, 15, 20, 25, None],
        "model__min_samples_split": [2, 5, 10, 15],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        "model__bootstrap": [True, False],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
        "model__random_state": [42],
    }

    comprehensive_grid = {
        "model__n_estimators": [50, 100, 200, 300, 500, 800],
        "model__max_depth": [5, 10, 15, 20, 25, 30, None],
        "model__min_samples_split": [2, 5, 10, 15, 20],
        "model__min_samples_leaf": [1, 2, 4, 8, 12],
        "model__max_features": ["sqrt", "log2", 0.2, 0.3, 0.5, 0.7],
        "model__bootstrap": [True, False],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
        "model__criterion": ["gini", "entropy"],
        "model__max_leaf_nodes": [None, 50, 100, 200],
        "model__random_state": [42],
    }

    return {"quick": quick_grid, "standard": standard_grid, "comprehensive": comprehensive_grid}


def create_scoring_metrics() -> Dict[str, Any]:
    """
    Create custom scoring metrics including business-focused F-beta score.

    Returns:
    --------
    Dict containing scoring metrics for GridSearchCV
    """

    # Custom F-beta scorer with beta=0.5 (emphasizes precision for business value)
    fbeta_05_scorer = make_scorer(fbeta_score, beta=0.5)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "fbeta_05": fbeta_05_scorer,
    }

    return scoring


def evaluate_model_comprehensive(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test_encoded: np.ndarray,
    target_encoder: TargetEncoder,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the best model on test data.

    Parameters:
    -----------
    model : Pipeline
        Trained model pipeline
    X_test : pd.DataFrame
        Test features
    y_test_encoded : np.ndarray
        Encoded test targets
    target_encoder : TargetEncoder
        Target encoder for decoding predictions

    Returns:
    --------
    Dict containing comprehensive evaluation metrics
    """

    y_pred_encoded = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    y_pred = target_encoder.inverse_transform(y_pred_encoded)
    y_test_original = target_encoder.inverse_transform(y_test_encoded)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test_encoded, y_pred_encoded),
        "precision": precision_score(y_test_encoded, y_pred_encoded),
        "recall": recall_score(y_test_encoded, y_pred_encoded),
        "f1_score": f1_score(y_test_encoded, y_pred_encoded),
        "roc_auc": roc_auc_score(y_test_encoded, y_pred_proba),
        "fbeta_05": fbeta_score(y_test_encoded, y_pred_encoded, beta=0.5),
        "fbeta_20": fbeta_score(y_test_encoded, y_pred_encoded, beta=2.0),
    }

    class_report = classification_report(y_test_original, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)

    return {
        "metrics": metrics,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "feature_importance": dict(
            zip(X_test.columns, model.named_steps["model"].feature_importances_)
        )
        if hasattr(model.named_steps["model"], "feature_importances_")
        else None,
    }


def run_hyperparameter_tuning(
    search_type: str = "standard",
    cv_folds: int = 5,
    scoring_metric: str = "fbeta_05",
    n_jobs: int = -1,
    verbose: int = 2,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Run hyperparameter tuning using GridSearchCV.

    Parameters:
    -----------
    search_type : str, default='standard'
        Type of search: 'quick', 'standard', or 'comprehensive'
    cv_folds : int, default=5
        Number of cross-validation folds
    scoring_metric : str, default='fbeta_05'
        Primary metric for model selection
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all processors)
    verbose : int, default=2
        Verbosity level for GridSearchCV

    Returns:
    --------
    Tuple containing the best model pipeline and results dictionary
    """

    logger.info(f"Starting hyperparameter tuning with {search_type} search")
    start_time = datetime.now()

    logger.info("Loading training data...")
    train_df = pd.read_csv(RAW_DATA_DIR / "train.csv")
    val_df = pd.read_csv(RAW_DATA_DIR / "val.csv")

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    logger.info(f"Combined dataset shape: {combined_df.shape}")

    X = combined_df.drop(columns=["Churn"])
    y = combined_df["Churn"]

    target_encoder = TargetEncoder()
    y_encoded = target_encoder.fit_transform(y)
    logger.info(f"Target distribution: {np.bincount(y_encoded)}")

    feature_engineer = ChurnFeatureEngineer()
    preprocessor = Preprocessor(pipeline_config=PIPELINE_CONFIG)
    rf_model = RandomForestClassifier()

    pipeline = Pipeline(
        [
            ("feature_engineering", feature_engineer),
            ("preprocessing", preprocessor),
            ("model", rf_model),
        ]
    )

    param_grids = create_parameter_grids()
    param_grid = param_grids[search_type]

    logger.info(f"Parameter grid contains {len(param_grid)} parameters")
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"Total parameter combinations: {total_combinations:,}")
    logger.info(f"Expected total fits: {total_combinations * cv_folds:,}")

    scoring = create_scoring_metrics()

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    logger.info("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=make_scorer(fbeta_score, beta=0.5),
        cv=cv,
        refit=scoring_metric,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )

    grid_search.fit(X, y_encoded)

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Grid search completed in {duration}")

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    logger.info(f"Best {scoring_metric} score: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")

    test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]
    y_test_encoded = target_encoder.transform(y_test)

    logger.info("Evaluating best model on test set...")
    test_results = evaluate_model_comprehensive(best_model, X_test, y_test_encoded, target_encoder)

    results = {
        "search_config": {
            "search_type": search_type,
            "cv_folds": cv_folds,
            "scoring_metric": scoring_metric,
            "total_combinations": total_combinations,
            "duration_seconds": duration.total_seconds(),
        },
        "best_params": best_params,
        "cv_results": {
            "best_score": best_score,
            "mean_test_scores": {
                metric: grid_search.cv_results_[f"mean_test_{metric}"][grid_search.best_index_]
                for metric in scoring.keys()
            },
            "std_test_scores": {
                metric: grid_search.cv_results_[f"std_test_{metric}"][grid_search.best_index_]
                for metric in scoring.keys()
            },
        },
        "test_evaluation": test_results,
        "target_encoding_mapping": target_encoder.get_encoding_mapping(),
        "timestamp": datetime.now().isoformat(),
    }

    return best_model, results, target_encoder


def save_tuning_results(
    model: Pipeline, results: Dict[str, Any], target_encoder: TargetEncoder, search_type: str
) -> None:
    """
    Save the tuned model and results to files.

    Parameters:
    -----------
    model : Pipeline
        Best model from hyperparameter tuning
    results : Dict
        Results dictionary from tuning
    target_encoder : TargetEncoder
        Target encoder
    search_type : str
        Type of search performed
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model artifacts
    model_filename = f"best_model_{search_type}_{timestamp}.pkl"
    model_path = MODELS_DIR / model_filename

    artifacts = {
        "model_pipeline": model,
        "target_encoder": target_encoder,
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    logger.info(f"Best model saved to {model_path}")

    results_filename = f"hyperparameter_results_{search_type}_{timestamp}.json"
    results_path = REPORTS_DIR / results_filename

    json_results = results.copy()
    if "feature_importance" in json_results["test_evaluation"]:
        feature_importance = json_results["test_evaluation"]["feature_importance"]
        if feature_importance:
            # Sort by importance for better readability
            sorted_features = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            json_results["test_evaluation"]["feature_importance"] = sorted_features

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")

    summary_filename = f"tuning_summary_{search_type}_{timestamp}.txt"
    summary_path = REPORTS_DIR / summary_filename

    with open(summary_path, "w") as f:
        f.write("Hyperparameter Tuning Summary\n")
        f.write("=" * 50 + "\n")
        f.write("Search Type: {search_type}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Duration: {results['search_config']['duration_seconds']:.2f} seconds\n")
        f.write(f"Total Combinations: {results['search_config']['total_combinations']:,}\n\n")

        f.write("Best Parameters:\n")
        f.write("-" * 20 + "\n")
        for param, value in results["best_params"].items():
            f.write(f"{param}: {value}\n")

        f.write("\nCross-Validation Results:\n")
        f.write("-" * 30 + "\n")
        for metric, score in results["cv_results"]["mean_test_scores"].items():
            std = results["cv_results"]["std_test_scores"][metric]
            f.write(f"{metric}: {score:.4f} (+/- {std * 2:.4f})\n")

        f.write("\nTest Set Evaluation:\n")
        f.write("-" * 25 + "\n")
        for metric, score in results["test_evaluation"]["metrics"].items():
            f.write(f"{metric}: {score:.4f}\n")

        if results["test_evaluation"]["feature_importance"]:
            f.write("\nTop 10 Feature Importances:\n")
            f.write("-" * 35 + "\n")
            sorted_features = dict(
                sorted(
                    results["test_evaluation"]["feature_importance"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            for i, (feature, importance) in enumerate(sorted_features.items()):
                if i >= 10:
                    break
                f.write(f"{feature}: {importance:.4f}\n")

    logger.info(f"Summary report saved to {summary_path}")


def main():
    """
    Main function to run hyperparameter tuning.
    """

    search_type = "standard"
    cv_folds = 5
    scoring_metric = "fbeta_05"

    logger.info("Starting hyperparameter tuning process")
    logger.info(
        f"Configuration: {search_type} search, {cv_folds}-fold CV, {scoring_metric} scoring"
    )

    try:
        best_model, results, target_encoder = run_hyperparameter_tuning(
            search_type=search_type,
            cv_folds=cv_folds,
            scoring_metric=scoring_metric,
            n_jobs=-1,
            verbose=2,
        )

        save_tuning_results(best_model, results, target_encoder, search_type)

        logger.info("Hyperparameter tuning completed successfully!")
        logger.info(f"Best {scoring_metric} score: {results['cv_results']['best_score']:.4f}")
        logger.info(f"Test accuracy: {results['test_evaluation']['metrics']['accuracy']:.4f}")
        logger.info(f"Test F-beta (0.5): {results['test_evaluation']['metrics']['fbeta_05']:.4f}")

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        raise


if __name__ == "__main__":
    main()
