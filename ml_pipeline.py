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
from sklearn.metrics import accuracy_score, classification_report


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_category_column(df):
    """
    Auto-detect which column contains job categories.
    Your CSV might call it different things depending on
    how Kaggle exported it.
    """
    possible_names = [
        'category', 'job_position_name', 'job_title',
        'position', 'role', 'label', 'target', 'class',
        'job_category', 'job_role', 'occupation'
    ]

    # Clean column names (remove BOM, whitespace, lowercase)
    clean_cols = {}
    for col in df.columns:
        cleaned = col.replace('\ufeff', '').strip().lower().replace(' ', '_')
        clean_cols[cleaned] = col

    # Try to find a match
    for name in possible_names:
        if name in clean_cols:
            original_col = clean_cols[name]
            print(f"  Found category column: '{original_col}'")
            return original_col

    # If no match, pick the column with fewest unique values
    # (that's probably the label column, not free text)
    object_cols = df.select_dtypes(include='object').columns
    if len(object_cols) > 0:
        unique_counts = {col: df[col].nunique() for col in object_cols}
        # Category column typically has 5-50 unique values
        candidates = {k: v for k, v in unique_counts.items() if 3 <= v <= 100}
        if candidates:
            best = min(candidates, key=candidates.get)
            print(f"  Auto-detected category column: '{best}' ({candidates[best]} unique values)")
            return best

    print("  ERROR: Could not find category column!")
    print(f"  Your columns are: {df.columns.tolist()}")
    raise ValueError("Cannot find category column. Check your CSV.")


def find_text_columns(df, category_col):
    """
    Auto-detect which columns contain resume text.
    Combines all text columns into one big text field.
    """
    text_cols = []
    for col in df.columns:
        if col == category_col:
            continue
        if df[col].dtype == 'object':
            # Check if it has substantial text (average > 20 chars)
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > 20:
                text_cols.append(col)
                print(f"  Using text column: '{col}' (avg {avg_len:.0f} chars)")

    if not text_cols:
        # Fallback: use ALL object columns except category
        text_cols = [c for c in df.select_dtypes(include='object').columns if c != category_col]
        print(f"  Fallback: using all text columns: {text_cols}")

    return text_cols


def build_model():
    print("=" * 60)
    print("  RESUME CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    # ---- STEP 1: Load ----
    print("\n[1/5] Loading dataset...")
    csv_path = os.path.join('data', 'resume_data.csv')

    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found!")
        print(f"  Make sure your CSV is at: {os.path.abspath(csv_path)}")
        return

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # ---- STEP 2: Find columns ----
    print("\n[2/5] Detecting columns...")
    category_col = find_category_column(df)
    text_cols = find_text_columns(df, category_col)

    # Combine text columns
    df['Resume'] = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
    df['Category'] = df[category_col].astype(str).str.strip()

    # Drop empty
    df = df.dropna(subset=['Category'])
    df = df[df['Category'].str.len() > 0]
    df = df[df['Resume'].str.len() > 50]

    print(f"  Usable rows: {len(df)}")
    print(f"  Categories: {df['Category'].nunique()}")

    # Show category distribution
    print(f"\n  Category Distribution:")
    cat_counts = df['Category'].value_counts()
    for cat, count in cat_counts.head(15).items():
        pct = count / len(df) * 100
        print(f"    {cat:<40} {count:>5} ({pct:.1f}%)")
    if len(cat_counts) > 15:
        print(f"    ... and {len(cat_counts) - 15} more categories")

    baseline = cat_counts.iloc[0] / len(df) * 100
    print(f"\n  Baseline accuracy (majority class): {baseline:.1f}%")

    # ---- STEP 3: Preprocess ----
    print("\n[3/5] Preprocessing...")
    df['Cleaned'] = df['Resume'].apply(clean_text)
    df = df[df['Cleaned'].str.split().str.len() >= 10]
    print(f"  After filtering short texts: {len(df)} rows")

    # Remove categories with too few samples
    min_samples = 3
    category_counts = df['Category'].value_counts()
    valid_categories = category_counts[category_counts >= min_samples].index
    df = df[df['Category'].isin(valid_categories)]
    print(f"  After removing rare categories (<{min_samples} samples): {len(df)} rows, {df['Category'].nunique()} categories")

    if len(df) < 20:
        print("  ERROR: Not enough data to train a model!")
        return

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X = tfidf.fit_transform(df['Cleaned'])
    y = df['Category']

    print(f"  TF-IDF vocabulary: {len(tfidf.vocabulary_)} features")

    # ---- STEP 4: Train ----
    print("\n[4/5] Training models...")

    # Stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # If some categories have only 1 sample, stratify fails
        print("  Warning: Using non-stratified split (some categories too small)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Compare models
      # Compare models using CROSS-VALIDATION for realistic accuracy
    from sklearn.model_selection import cross_val_score

    models = {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'Linear SVM': SGDClassifier(loss='modified_huber', max_iter=1000, random_state=42),
    }
    # NOTE: changed SVM from 'hinge' to 'modified_huber'
    # because 'hinge' does NOT support predict_proba
    # 'modified_huber' gives probability estimates AND is still an SVM

    best_acc = 0
    best_cv_acc = 0
    best_name = None
    best_model = None
    comparison = {}

    for name, m in models.items():
        start = time.perf_counter()

        # Cross-validation on FULL data (gives realistic accuracy)
        cv_folds = min(5, len(df) // 10)  # Don't use more folds than data allows
        cv_folds = max(2, cv_folds)       # At least 2 folds
        cv_scores = cross_val_score(m, X, y, cv=cv_folds, scoring='accuracy')
        cv_mean = cv_scores.mean()

        # Also train on train split for test accuracy
        m.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        preds = m.predict(X_test)
        test_acc = accuracy_score(y_test, preds)

        print(f"\n  {name}")
        print(f"    Test Accuracy:  {test_acc * 100:.2f}%")
        print(f"    CV Accuracy:    {cv_mean * 100:.2f}% (±{cv_scores.std() * 100:.1f}%)")
        print(f"    Train time:     {elapsed:.3f}s")

        # If test accuracy is 100% but CV is lower, flag it
        if test_acc == 1.0 and cv_mean < 0.95:
            print(f"    ⚠ WARNING: 100% test accuracy but {cv_mean*100:.1f}% CV → likely overfitting")

        comparison[name] = {
            'accuracy_test': round(test_acc * 100, 2),
            'accuracy_cv': round(cv_mean * 100, 2),
            'accuracy_cv_std': round(cv_scores.std() * 100, 2),
            'train_time': round(elapsed, 3),
        }

        # Use CV accuracy for selection (more reliable than test accuracy)
        if cv_mean > best_cv_acc:
            best_cv_acc = cv_mean
            best_acc = test_acc
            best_name = name
            best_model = m

    # Use CV accuracy as the "real" accuracy (not the potentially inflated test accuracy)
    reported_accuracy = best_cv_acc

    print(f"\n  ★ WINNER: {best_name}")
    print(f"  ★ Test Accuracy: {best_acc * 100:.2f}%")
    print(f"  ★ CV Accuracy:   {best_cv_acc * 100:.2f}% (this is the real number)")
    print(f"  ★ vs Baseline:   +{best_cv_acc * 100 - baseline:.1f}% improvement")
    # ---- STEP 5: Save ----
    print("\n[5/5] Saving...")

    pickle.dump(best_model, open('model.pkl', 'wb'))
    pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

    # Get detailed metrics
    final_preds = best_model.predict(X_test)
    report = classification_report(y_test, final_preds, output_dict=True, zero_division=0)

    metrics = {
        'accuracy': round(reported_accuracy * 100, 2),
        'accuracy_test': round(best_acc * 100, 2),
        'baseline_accuracy': round(baseline, 2),
        'improvement_over_baseline': round(best_acc * 100 - baseline, 2),
        'model_type': best_name,
        'total_samples': len(df),
        'num_categories': int(df['Category'].nunique()),
        'vocabulary_size': len(tfidf.vocabulary_),
        'categories': sorted(df['Category'].unique().tolist()),
        'model_comparison': comparison,
        'precision_macro': round(report.get('macro avg', {}).get('precision', 0) * 100, 1),
        'recall_macro': round(report.get('macro avg', {}).get('recall', 0) * 100, 1),
        'f1_macro': round(report.get('macro avg', {}).get('f1-score', 0) * 100, 1),
        'trained_at': datetime.now().isoformat(),
    }

    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"  Accuracy (CV):   {reported_accuracy * 100:.2f}%")
    print(f"  Accuracy (Test): {best_acc * 100:.2f}%")
    print(f"  Baseline:  {baseline:.1f}%")
    print(f"  Model:     {best_name}")
    print(f"  Saved:     model.pkl, tfidf.pkl, model_metrics.json")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    build_model()