#!/usr/bin/env python3
"""
CommonLit Readability Prize - lightweight feature engineering + ensemble.
No deep learning embeddings to save time; focus on robust classical ML.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import re

CACHE_DIR = '<DATASET_ROOT>/mle-bench/cache/commonlitreadabilityprize/prepared/public'
WORK_DIR = '<DATASET_ROOT>/planning-research/agent_data/runners/_active/claudecode/runs/run_20260801_190617_haiku45_rec_1h_commonlitreadabilityprize'

def load_data():
    train = pd.read_csv(f'{CACHE_DIR}/train.csv')
    test = pd.read_csv(f'{CACHE_DIR}/test.csv')
    return train, test

def count_syllables(word):
    """Rough syllable count using vowel groups."""
    word = word.lower()
    count = 0
    vowels = 'aeiou'
    previous_was_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    # Adjust for silent e
    if word.endswith('e'):
        count -= 1
    return max(1, count)

def extract_features(texts):
    """Extract comprehensive linguistic features."""
    features = []
    for text in texts:
        feat = {}

        # Basic counts
        words = text.split()
        feat['word_count'] = len(words)
        feat['char_count'] = len(text)
        feat['unique_words'] = len(set(w.lower() for w in words))
        feat['unique_ratio'] = feat['unique_words'] / max(1, feat['word_count'])

        # Word length
        feat['avg_word_len'] = feat['char_count'] / max(1, feat['word_count'])
        word_lens = [len(w) for w in words]
        feat['max_word_len'] = max(word_lens) if word_lens else 0
        feat['min_word_len'] = min(word_lens) if word_lens else 0
        feat['std_word_len'] = np.std(word_lens) if len(word_lens) > 1 else 0

        # Sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        feat['sentence_count'] = len(sentences)
        feat['avg_sentence_len'] = feat['word_count'] / max(1, feat['sentence_count'])
        sent_lens = [len(s.split()) for s in sentences]
        feat['max_sent_len'] = max(sent_lens) if sent_lens else 0
        feat['min_sent_len'] = min(sent_lens) if sent_lens else 0

        # Syllables (Flesch-Kincaid inspired)
        total_syllables = sum(count_syllables(w) for w in words)
        feat['avg_syllables_per_word'] = total_syllables / max(1, feat['word_count'])
        feat['syllable_count'] = total_syllables

        # Punctuation
        feat['comma_count'] = text.count(',')
        feat['semicolon_count'] = text.count(';')
        feat['quote_count'] = text.count('"') + text.count("'")
        feat['paren_count'] = text.count('(') + text.count(')')

        # Flesch Reading Ease components
        feat['flesch_kincaid_grade'] = (
            0.39 * (feat['word_count'] / max(1, feat['sentence_count'])) +
            11.8 * (feat['avg_syllables_per_word']) - 15.59
        )

        # Complex words (3+ syllables)
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        feat['complex_word_ratio'] = complex_words / max(1, feat['word_count'])

        # Gunning Fog Index
        feat['gunning_fog'] = (
            0.4 * (feat['word_count'] / max(1, feat['sentence_count']) + 100 * (complex_words / max(1, feat['word_count'])))
        )

        features.append(feat)

    return pd.DataFrame(features)

def get_tfidf_features(texts, max_features=100):
    """Get sparse TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                  lowercase=True, stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    return tfidf.toarray(), vectorizer

def train_lgb_fold(X_train, y_train, X_val, y_val):
    """Train single LightGBM fold."""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=300)
    return model

def train_xgb_fold(X_train, y_train, X_val, y_val):
    """Train single XGBoost fold."""
    params = {
        'objective': 'reg:squarederror',
        'metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'verbosity': 0
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=300, evals=[(dval, 'val')])
    return model

def train_rf_fold(X_train, y_train, X_val, y_val):
    """Train single Random Forest fold."""
    model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    print("Loading data...")
    train, test = load_data()
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Extract hand-crafted features
    print("Extracting linguistic features...")
    train_feat = extract_features(train['excerpt'].values)
    test_feat = extract_features(test['excerpt'].values)

    # Get TF-IDF
    print("Computing TF-IDF features...")
    train_tfidf, vectorizer = get_tfidf_features(train['excerpt'].tolist(), max_features=100)
    test_tfidf = vectorizer.transform(test['excerpt'].tolist()).toarray()

    # Combine all features
    X_train = np.hstack([train_feat.values, train_tfidf])
    X_test = np.hstack([test_feat.values, test_tfidf])
    y_train = train['target'].values

    print(f"Feature matrix shape: {X_train.shape}")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ensemble with 5-fold CV
    print("Cross-validation training (LightGBM + XGBoost + RandomForest)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_lgb = np.zeros((len(X_test), 5))
    test_xgb = np.zeros((len(X_test), 5))
    test_rf = np.zeros((len(X_test), 5))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold+1}/5...")
        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        y_tr, y_va = y_train[train_idx], y_train[val_idx]

        # LightGBM
        model_lgb = train_lgb_fold(X_tr, y_tr, X_va, y_va)
        preds_lgb_val = model_lgb.predict(X_va)
        preds_lgb_test = model_lgb.predict(X_test)

        # XGBoost
        model_xgb = train_xgb_fold(X_tr, y_tr, X_va, y_va)
        preds_xgb_val = model_xgb.predict(xgb.DMatrix(X_va))
        preds_xgb_test = model_xgb.predict(xgb.DMatrix(X_test))

        # Random Forest
        model_rf = train_rf_fold(X_tr, y_tr, X_va, y_va)
        preds_rf_val = model_rf.predict(X_va)
        preds_rf_test = model_rf.predict(X_test)

        # Blend predictions
        oof_preds[val_idx] = (preds_lgb_val + preds_xgb_val + preds_rf_val) / 3
        test_lgb[:, fold] = preds_lgb_test
        test_xgb[:, fold] = preds_xgb_test
        test_rf[:, fold] = preds_rf_test

    # CV score
    rmse = np.sqrt(np.mean((oof_preds - y_train)**2))
    print(f"CV RMSE: {rmse:.6f}")

    # Final submission: average across folds and models
    test_final = (test_lgb.mean(axis=1) + test_xgb.mean(axis=1) + test_rf.mean(axis=1)) / 3

    # Clip to reasonable range
    test_final = np.clip(test_final, -4, 2)

    # Save
    sub = pd.DataFrame({'id': test['id'], 'target': test_final})
    sub.to_csv(f'{WORK_DIR}/submission.csv', index=False)
    print(f"Submission saved: {WORK_DIR}/submission.csv")
    print(sub.head(10))

    return sub

if __name__ == '__main__':
    main()
