import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

def main():
    print("Loading data...")
    train_path = "<DATASET_ROOT>/mle-bench/cache/commonlitreadabilityprize/prepared/public/train.csv"
    test_path = "<DATASET_ROOT>/mle-bench/cache/commonlitreadabilityprize/prepared/public/test.csv"
    sample_sub_path = "<DATASET_ROOT>/mle-bench/cache/commonlitreadabilityprize/prepared/public/sample_submission.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Set up 5-fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(train))
    test_predictions = np.zeros(len(test))

    # Tfidf Vectorizer
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words=None)
    
    # We fit the vectorizer on the full corpus (train + test excerpts)
    all_texts = pd.concat([train['excerpt'], test['excerpt']], axis=0)
    vectorizer.fit(all_texts)
    
    X_train_full = vectorizer.transform(train['excerpt'])
    X_test = vectorizer.transform(test['excerpt'])
    y_train_full = train['target'].values

    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        oof_predictions[val_idx] = val_preds
        
        # Calculate RMSE for this fold
        fold_rmse = np.sqrt(np.mean((y_val - val_preds)**2))
        fold_scores.append(fold_rmse)
        print(f"Fold {fold} RMSE: {fold_rmse:.5f}")

        # Predict on test
        test_predictions += model.predict(X_test) / 5.0

    cv_rmse = np.sqrt(np.mean((y_train_full - oof_predictions)**2))
    print(f"\nMean Fold RMSE: {np.mean(fold_scores):.5f}")
    print(f"Overall CV RMSE: {cv_rmse:.5f}")

    # Create submission file
    submission = pd.DataFrame({
        'id': test['id'],
        'target': test_predictions
    })
    
    submission_path = "<DATASET_ROOT>/planning-research/agent_data/runners/_active/gemini/runs/run_20260801_203455_flash25_rec_1h_commonlitreadabilityprize/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")
    print(submission.head())

if __name__ == '__main__':
    main()
