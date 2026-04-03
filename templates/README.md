# Resume Screener

A Flask-based resume screening web application that analyzes resumes and predicts relevant job categories using machine learning.

## Features

- Resume text analysis
- Job category prediction
- Clean Flask-based UI
- ML model integration using saved `.pkl` files
- Simple recruiter-style screening workflow

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS

## Project Structure

```text
resume-screener/
├── app.py
├── model.py
├── model.pkl
├── tfidf.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── resume_data.csv
└── templates/
    └── index.html