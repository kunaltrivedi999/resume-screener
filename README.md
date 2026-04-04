# Resume Screener

**Live Deployment:** https://resume-screener-livid-ten.vercel.app/

A machine learning-powered web application that objectively evaluates candidate resumes against job descriptions using Natural Language Processing and strict skill-matching algorithms.

---

## Overview

This system replaces manual resume parsing with an automated scoring pipeline. It calculates a composite fit score based on hard skill overlap, semantic context, and experience requirements, while actively penalizing missing core competencies.

---

## Core Features

- **Semantic Matching Engine:** Utilizes TF-IDF and Cosine Similarity with custom stop-word filtering to map contextual relevance and ignore generic filler terminology.
- **Algorithmic Penalty Logic:** Extracts technical skills via word-boundary mapping and applies strict percentage deductions for missing job requirements.
- **ML Classification:** Uses a Logistic Regression model trained via stratified cross-validation to predict candidate career tracks and calculate confidence intervals.
- **Data Extraction:** Automatically parses years of experience, education levels, and generates tiered salary estimates based on market multipliers.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML / NLP | scikit-learn, Pandas, NumPy, Regex |
| Frontend | HTML5, CSS3, Jinja2 |
| Deployment | Render / Vercel |

---

## Local Development Setup

**1. Clone the repository**

```bash
git clone https://github.com/kunaltrivedi999/resume-screener.git
cd resume-screener
2. Create and activate a virtual environment

Bash

python -m venv venv
Windows:

Bash

venv\Scripts\activate
macOS / Linux:

Bash

source venv/bin/activate
3. Install dependencies

Bash

pip install -r requirements.txt
4. Run the ML pipeline

Bash

python ml_pipeline.py
5. Start the Flask server

Bash

python app.py
6. Open in browser

text

http://127.0.0.1:5000
Architecture
text

resume-screener/
├── app.py               # Flask routing layer and composite score calculator
├── matching_engine.py   # SimilarityEngine class, TF-IDF vectorization, penalty logic
├── ml_pipeline.py       # Offline training script — generates model.pkl and tfidf.pkl
├── requirements.txt
└── templates/
How It Works
text

Resume Input
     ↓
Text Extraction
     ↓
TF-IDF Vectorization → Cosine Similarity Score
     ↓
Skill Extraction → Penalty Deductions for Missing Skills
     ↓
Logistic Regression → Career Track + Confidence
     ↓
Experience + Education Parsing → Salary Estimate
     ↓
Composite Fit Score Output
License
MIT

text


---