# evaluate.py
import mlflow
import pandas as pd
from sklearn.metrics import classification_report
from data_loader import load_imdb_data
from config import MLFLOW_TRACKING_URI

def compare_models():
    """Compare all trained models and generate a report"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load test data
    _, test_df = load_imdb_data(save_as_csv=False)
    test_df = test_df.sample(500, random_state=42)  # Smaller subset for quick evaluation
    
    # Get all runs
    experiments = mlflow.search_experiments()
    results = []
    
    for exp in experiments:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        
        for _, run in runs.iterrows():
            if run['status'] == 'FINISHED':
                results.append({
                    'run_id': run['run_id'],
                    'model_name': run['tags.mlflow.runName'],
                    'accuracy': run['metrics.accuracy'],
                    'f1_score': run['metrics.f1_score'],
                    'precision': run['metrics.precision'],
                    'recall': run['metrics.recall']
                })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    if not comparison_df.empty:
        print("=== Model Comparison ===")
        print(comparison_df.round(4).to_string(index=False))
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['f1_score'].idxmax()]
        print(f"\n=== Best Model ===")
        print(f"Model: {best_model['model_name']}")
        print(f"F1-Score: {best_model['f1_score']:.4f}")
        print(f"Accuracy: {best_model['accuracy']:.4f}")
        
        return best_model
    else:
        print("No completed runs found.")
        return None

if __name__ == "__main__":
    compare_models()