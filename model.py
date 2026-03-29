import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def clean_text(text):
    """Cleans raw resume text for the AI model."""
    text = str(text).lower()
    # Keep only letters and spaces
    text = re.sub(r'[^a-z ]', ' ', text) 
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_model():
    print("1. Loading dataset...")
    df = pd.read_csv('data/resume_data.csv')
    
    print("2. Preprocessing data...")
    # Fix the hidden character in the column name
    df.rename(columns={'\ufeffjob_position_name': 'Category'}, inplace=True)
    
    # Combine all text fields into one massive text block per resume
    df['Resume'] = (df['career_objective'].fillna('') + ' ' + 
                    df['skills'].fillna('') + ' ' + 
                    df['degree_names'].fillna('') + ' ' + 
                    df['responsibilities'].fillna(''))
    
    # Drop rows without a target category
    df = df.dropna(subset=['Category'])
    
    print("3. Cleaning text (this takes a few seconds)...")
    df['Cleaned_Resume'] = df['Resume'].apply(clean_text)
    
    print("4. Training the Naive Bayes Classifier...")
    # Convert words to numbers (max 1500 features to prevent memory crash)
    tfidf = TfidfVectorizer(max_features=1500, stop_words='english')
    X = tfidf.fit_transform(df['Cleaned_Resume'])
    y = df['Category']
    
    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("-" * 50)
    print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
    print("-" * 50)
    
    print("5. Saving model files...")
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
    print("SUCCESS: model.pkl and tfidf.pkl generated.")

if __name__ == "__main__":
    build_model()