# File: ml_pipeline.py (REPLACES model.py)

"""
ML Pipeline: Resume Category Classification

This script trains a text classifier on resume data, compares multiple
algorithms, selects the best one, and saves both the model and its
performance metrics for transparency.

Algorithm: Multinomial Naive Bayes (selected after comparison)
Feature Extraction: TF-IDF Vectorization
Evaluation: Stratified 80/20 train-test split

Usage: python ml_pipeline.py
Output: model.pkl, tfidf.pkl, model_metrics.json
"""

import json
import os
import re
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def clean_text(text: str) -> str:
    """Clean raw resume text for ML vectorization."""
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def explore_data(df: pd.DataFrame) -> dict:
    """
    Explore and summarize the dataset before training.

    This function exists so that YOU can answer interview questions
    about your training data. If someone asks 'what dataset did you use?'
    you need to know these numbers.
    """
    print("\n" + "=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)

    total_rows = len(df)
    num_categories = df['Category'].nunique()
    categories = df['Category'].value_counts()

    print(f"\n  Total resumes:     {total_rows}")
    print(f"  Total categories:  {num_categories}")
    print(f"  Avg per category:  {total_rows // num_categories}")

    print(f"\n  Category Distribution:")
    print(f"  {'Category':<35} {'Count':>6} {'Percent':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*8}")
    for cat, count in categories.items():
        pct = count / total_rows * 100
        print(f"  {cat:<35} {count:>6} {pct:>7.1f}%")

    # Baseline accuracy (most common class)
    baseline = categories.iloc[0] / total_rows * 100
    print(f"\n  Baseline accuracy (majority class): {baseline:.1f}%")
    print(f"  (If model is not ABOVE this, it's useless)")

    # Text length stats
    df['text_length'] = df['Resume'].apply(lambda x: len(str(x).split()))
    print(f"\n  Average resume length: {df['text_length'].mean():.0f} words")
    print(f"  Shortest resume:       {df['text_length'].min()} words")
    print(f"  Longest resume:        {df['text_length'].max()} words")

    return {
        'total_samples': total_rows,
        'num_categories': num_categories,
        'baseline_accuracy': round(baseline, 2),
        'categories': categories.to_dict(),
        'avg_resume_length': round(df['text_length'].mean()),
    }


def compare_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Train and compare multiple classifiers.

    This is what separates 'I used sklearn' from
    'I evaluated three classifiers and selected the best one.'
    """
    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=1.0),
        'Linear SVM (SGD)': SGDClassifier(
            loss='hinge',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
    }

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results = {}
    best_accuracy = 0
    best_name = None
    best_model = None

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"\n  {name}")
        print(f"    Accuracy:   {accuracy * 100:.2f}%")
        print(f"    Train time: {train_time:.3f}s")

        results[name] = {
            'accuracy': round(accuracy * 100, 2),
            'train_time_seconds': round(train_time, 3),
        }

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_name = name
            best_model = model

    print(f"\n  ★ BEST MODEL: {best_name} ({best_accuracy * 100:.2f}%)")

    return results, best_name, best_model


def build_model():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("  RESUME CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading dataset...")
    csv_path = os.path.join('data', 'resume_data.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Fix BOM character in column name if present
    df.columns = [col.replace('\ufeff', '') for col in df.columns]
    if 'job_position_name' in df.columns:
        df.rename(columns={'job_position_name': 'Category'}, inplace=True)

    # Combine text fields
    text_columns = ['career_objective', 'skills', 'degree_names', 'responsibilities']
    for col in text_columns:
        if col not in df.columns:
            df[col] = ''
    df['Resume'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    df = df.dropna(subset=['Category'])

    # Step 2: Explore
    print("\n[2/5] Exploring dataset...")
    data_stats = explore_data(df)

    # Step 3: Preprocess
    print("\n[3/5] Preprocessing text...")
    df['Cleaned_Resume'] = df['Resume'].apply(clean_text)

    # Remove very short resumes (less than 10 words after cleaning)
    df = df[df['Cleaned_Resume'].apply(lambda x: len(x.split()) >= 10)]
    print(f"  Resumes after filtering: {len(df)}")

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(
        max_features=3000,          # Increased from 1500 for better coverage
        stop_words='english',
        ngram_range=(1, 2),          # Capture 2-word phrases like 'machine learning'
        sublinear_tf=True,           # Log normalization
        min_df=2,                    # Word must appear in at least 2 resumes
    )
    X = tfidf.fit_transform(df['Cleaned_Resume'])
    y = df['Category']

    print(f"  Vocabulary size: {len(tfidf.vocabulary_)} features")

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")

    # Step 5: Compare models
    print("\n[4/5] Training and comparing models...")
    comparison, best_name, best_model = compare_models(
        X_train, X_test, y_train, y_test
    )

    # Detailed metrics for best model
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    # Extract top features per category (for explainability)
    top_features = {}
    if hasattr(best_model, 'feature_log_prob_') or hasattr(best_model, 'coef_'):
        feature_names = tfidf.get_feature_names_out()
        if hasattr(best_model, 'feature_log_prob_'):
            # Naive Bayes
            for i, category in enumerate(best_model.classes_):
                log_probs = best_model.feature_log_prob_[i]
                top_indices = np.argsort(log_probs)[-8:][::-1]
                top_features[category] = [
                    feature_names[idx] for idx in top_indices
                ]

    print(f"\n  Top predictive features per category (sample):")
    for cat, features in list(top_features.items())[:5]:
        print(f"    {cat}: {', '.join(features[:5])}")

    # Step 6: Save everything
    print("\n[5/5] Saving model artifacts...")

    pickle.dump(best_model, open('model.pkl', 'wb'))
    pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

    # Save metrics for the frontend to display
    metrics = {
        'accuracy': round(accuracy * 100, 2),
        'total_samples': data_stats['total_samples'],
        'num_categories': data_stats['num_categories'],
        'baseline_accuracy': data_stats['baseline_accuracy'],
        'model_type': best_name,
        'vocabulary_size': len(tfidf.vocabulary_),
        'categories': list(best_model.classes_),
        'model_comparison': comparison,
        'top_features_per_category': top_features,
        'precision_macro': round(report.get('macro avg', {}).get('precision', 0) * 100, 1),
        'recall_macro': round(report.get('macro avg', {}).get('recall', 0) * 100, 1),
        'f1_macro': round(report.get('macro avg', {}).get('f1-score', 0) * 100, 1),
        'trained_at': datetime.now().isoformat(),
    }

    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Model:      {best_name}")
    print(f"  Accuracy:   {accuracy * 100:.2f}%")
    print(f"  Baseline:   {data_stats['baseline_accuracy']:.1f}%")
    print(f"  Lift:       +{accuracy * 100 - data_stats['baseline_accuracy']:.1f}% over random")
    print(f"  Files:      model.pkl, tfidf.pkl, model_metrics.json")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    build_model()