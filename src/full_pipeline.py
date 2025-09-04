import pickle

from loguru import logger
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PIPELINE_CONFIG,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SAVE_INTERIM,
    SAVE_METRICS_REPORT,
    SAVE_PROCESSED,
)
from src.features import ChurnFeatureEngineer
from src.preprocessor import Preprocessor
from src.target_encoder import TargetEncoder


def full_pipeline():
    # 1. read the raw training data
    train_df = pd.read_csv(RAW_DATA_DIR / "train.csv")
    if train_df.empty:
        logger.error("The training data is empty.")
        return
    logger.info(f"Training data shape: {train_df.shape}")

    # 2. split the data into features and target
    X = train_df.drop(columns=["Churn"])
    y = train_df["Churn"]

    # 3. encode the target variable
    target_encoder = TargetEncoder()
    y_encoded = target_encoder.fit_transform(y)
    logger.info(f"Target variable encoded. Mapping: {target_encoder.get_encoding_mapping()}")

    # 4. engineer features
    feature_engineer = ChurnFeatureEngineer()
    X_fe = feature_engineer.fit_transform(X)
    if SAVE_INTERIM:
        interim_path = INTERIM_DATA_DIR / "train_fe.csv"
        X_fe.to_csv(interim_path, index=False)
        logger.info(f"Interim data with engineered features saved to {interim_path}")

    logger.info(f"Features after engineering shape: {X_fe.shape}")

    # 5. preprocess the features
    preprocessor = Preprocessor(pipeline_config=PIPELINE_CONFIG)
    X_preprocessed = preprocessor.fit_transform(X_fe)
    if SAVE_PROCESSED:
        processed_path = INTERIM_DATA_DIR / "train_preprocessed.csv"
        X_preprocessed.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")
    logger.info(f"Features after preprocessing shape: {X_preprocessed.shape}")

    # 6. training
    params = {
        "max_depth": 7,
        "min_samples_leaf": 1,
        "min_samples_split": 5,
        "n_estimators": 500,
    }
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_preprocessed, y_encoded)
    logger.info("Model training completed.")

    # 7. generate evaluation metrics
    y_pred_encoded = model.predict(X_preprocessed)

    # Decode predictions back to original labels for interpretability
    y_pred = target_encoder.inverse_transform(y_pred_encoded)
    y_original = target_encoder.inverse_transform(y_encoded)

    accuracy = accuracy_score(y_encoded, y_pred_encoded)
    f1 = f1_score(y_encoded, y_pred_encoded)
    precision = precision_score(y_encoded, y_pred_encoded)
    recall = recall_score(y_encoded, y_pred_encoded)
    fbeta = fbeta_score(y_encoded, y_pred_encoded, beta=0.5)
    report = classification_report(y_original, y_pred)
    conf_matrix = confusion_matrix(y_encoded, y_pred_encoded)

    if SAVE_METRICS_REPORT:
        metrics_path = REPORTS_DIR / "metrics_report.txt"
        with open(metrics_path, "w") as f:
            f.write(f"Target Encoding Mapping: {target_encoder.get_encoding_mapping()}\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F-beta Score (beta=0.5): {fbeta}\n")
            f.write("\nClassification Report (Original Labels):\n")
            f.write(report)
            f.write("\nConfusion Matrix (Encoded Labels):\n")
            f.write(str(conf_matrix))
        logger.info(f"Metrics report saved to {metrics_path}")

    # 8. Form the complete pipeline including target encoding
    full_pipeline = Pipeline(
        steps=[
            ("feature_engineering", feature_engineer),
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    # 9. load validation data and evaluate
    val_df = pd.read_csv(RAW_DATA_DIR / "val.csv")
    if val_df.empty:
        logger.error("The validation data is empty.")
        return
    logger.info(f"Validation data shape: {val_df.shape}")
    X_val = val_df.drop(columns=["Churn"])
    y_val = val_df["Churn"]

    # Encode validation target
    y_val_encoded = target_encoder.transform(y_val)

    y_val_pred_encoded = full_pipeline.predict(X_val)

    # Decode predictions for interpretability
    y_val_pred = target_encoder.inverse_transform(y_val_pred_encoded)
    y_val_original = target_encoder.inverse_transform(y_val_encoded)

    val_accuracy = accuracy_score(y_val_encoded, y_val_pred_encoded)
    logger.info(f"Validation Accuracy: {val_accuracy}")
    val_f1 = f1_score(y_val_encoded, y_val_pred_encoded)
    logger.info(f"Validation F1 Score: {val_f1}")
    val_precision = precision_score(y_val_encoded, y_val_pred_encoded)
    logger.info(f"Validation Precision: {val_precision}")
    val_recall = recall_score(y_val_encoded, y_val_pred_encoded)
    logger.info(f"Validation Recall: {val_recall}")
    val_fbeta = fbeta_score(y_val_encoded, y_val_pred_encoded, beta=0.5)
    logger.info(f"Validation F-beta Score (beta=0.5): {val_fbeta}")
    val_report = classification_report(y_val_original, y_val_pred)
    logger.info(f"Validation Classification Report:\n{val_report}")
    val_conf_matrix = confusion_matrix(y_val_encoded, y_val_pred_encoded)
    logger.info(f"Validation Confusion Matrix:\n{val_conf_matrix}")

    if SAVE_METRICS_REPORT:
        val_metrics_path = REPORTS_DIR / "validation_metrics_report.txt"
        with open(val_metrics_path, "w") as f:
            f.write(f"Target Encoding Mapping: {target_encoder.get_encoding_mapping()}\n")
            f.write(f"Validation Accuracy: {val_accuracy}\n")
            f.write(f"Validation F1 Score: {val_f1}\n")
            f.write(f"Validation Precision: {val_precision}\n")
            f.write(f"Validation Recall: {val_recall}\n")
            f.write(f"Validation F-beta Score (beta=0.5): {val_fbeta}\n")
            f.write("\nValidation Classification Report (Original Labels):\n")
            f.write(val_report)
            f.write("\nValidation Confusion Matrix (Encoded Labels):\n")
            f.write(str(val_conf_matrix))
        logger.info(f"Validation metrics report saved to {val_metrics_path}")

    # Return both the pipeline and target encoder for future use
    return full_pipeline, target_encoder


if __name__ == "__main__":
    pipeline, encoder = full_pipeline()
    artifacts = {
        "model_pipeline": pipeline,
        "target_encoder": encoder,
    }
    with open(MODELS_DIR / "artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
