# train.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, RANDOM_STATE, TEST_SIZE, ARTIFACTS_DIR
from data_loader import load_imdb_data

class SentimentAnalysisTrainer:
    def __init__(self):
        self.vectorizer = None
        self.models = {}
        
    def setup_mlflow(self):
        """Set up MLflow tracking"""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
    def load_and_prepare_data(self):
        """Load and prepare the data"""
        print("Loading and preparing data...")
        train_df, test_df = load_imdb_data(save_as_csv=True)
        
        # Use a subset for faster training (remove this for full dataset)
        train_df = train_df.sample(5000, random_state=RANDOM_STATE)
        test_df = test_df.sample(1000, random_state=RANDOM_STATE)
        
        X_train = train_df['text'].values
        y_train = train_df['label'].values
        X_test = test_df['text'].values
        y_test = test_df['label'].values
        
        return X_train, X_test, y_train, y_test
    
    def create_vectorizer(self, max_features=5000):
        """Create and fit TF-IDF vectorizer"""
        print("Creating TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        return self.vectorizer
    
    def get_models(self):
        """Define the models to train"""
        return {
            "LogisticRegression": LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "SGDClassifier": SGDClassifier(
                random_state=RANDOM_STATE,
                max_iter=1000,
                tol=1e-3
            )
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Create and save confusion matrix plot"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = ARTIFACTS_DIR / f'confusion_matrix_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }
        
        # Create confusion matrix plot
        plot_path = self.plot_confusion_matrix(y_test, y_pred, model_name)
        
        return metrics, plot_path
    
    def train_models(self):
        """Main training function with MLflow tracking"""
        self.setup_mlflow()
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        
        # Vectorize text data
        vectorizer = self.create_vectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        print("Starting model training with MLflow tracking...")
        
        models = self.get_models()
        
        for model_name, model in models.items():
            print(f"\n=== Training {model_name} ===")
            
            with mlflow.start_run(run_name=model_name):
                # Log parameters
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("vectorizer_max_features", 5000)
                mlflow.log_param("ngram_range", "1,2")
                mlflow.log_param("random_state", RANDOM_STATE)
                
                # Add model-specific parameters
                if hasattr(model, 'get_params'):
                    model_params = model.get_params()
                    for param_name, param_value in model_params.items():
                        if isinstance(param_value, (int, float, str)):
                            mlflow.log_param(f"model_{param_name}", param_value)
                
                # Train model
                print(f"Training {model_name}...")
                model.fit(X_train_vec, y_train)
                
                # Evaluate model
                print(f"Evaluating {model_name}...")
                metrics, plot_path = self.evaluate_model(model, X_test_vec, y_test, model_name)
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                    print(f"{metric_name}: {metric_value:.4f}")
                
                # Log artifacts
                mlflow.log_artifact(plot_path)
                mlflow.log_artifact(str(__file__))  # Log the training script itself
                
                # Log model
                mlflow.sklearn.log_model(model, f"{model_name.lower()}_model")
                
                # Save vectorizer
                vectorizer_path = ARTIFACTS_DIR / f"vectorizer_{model_name}.pkl"
                joblib.dump(vectorizer, vectorizer_path)
                mlflow.log_artifact(vectorizer_path)
                
                print(f"Completed {model_name} - Logged to MLflow")
        
        print("\n=== Training Complete ===")
        print(f"MLflow UI can be launched with: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

def main():
    trainer = SentimentAnalysisTrainer()
    trainer.train_models()

if __name__ == "__main__":
    main()