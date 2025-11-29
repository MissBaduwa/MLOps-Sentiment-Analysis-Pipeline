# data_loader.py
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from config import DATA_DIR

def load_imdb_data(save_as_csv=True):
    """
    Load IMDb movie reviews dataset and convert to pandas DataFrame
    """
    print("Loading IMDb dataset...")
    
    # Load dataset
    train_ds, test_ds = tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        as_supervised=True,
        shuffle_files=True
    )
    
    # Convert to lists
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    
    for text, label in train_ds:
        train_texts.append(text.numpy().decode('utf-8'))
        train_labels.append(label.numpy())
    
    for text, label in test_ds:
        test_texts.append(text.numpy().decode('utf-8'))
        test_labels.append(label.numpy())
    
    # Create DataFrames
    train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
    
    # Map labels to meaningful names
    train_df['sentiment'] = train_df['label'].map({0: 'negative', 1: 'positive'})
    test_df['sentiment'] = test_df['label'].map({0: 'negative', 1: 'positive'})
    
    if save_as_csv:
        train_df.to_csv(DATA_DIR / 'train.csv', index=False)
        test_df.to_csv(DATA_DIR / 'test.csv', index=False)
        print(f"Data saved to {DATA_DIR}")
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Class distribution in training set:\n{train_df['sentiment'].value_counts()}")
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_imdb_data()