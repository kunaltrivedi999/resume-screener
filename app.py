# File: app.py (REFACTORED — replaces your current app.py)

"""
Resume Screening Dashboard — Flask Application

Architecture:
    GET  /              → Web UI (form + results)
    POST /              → Process form submission
    POST /api/v1/match  → Headless JSON API
    GET  /health        → Health check

Scoring Pipeline:
    1. Extract skills from both texts (word-boundary matching)
    2. Compute TF-IDF cosine similarity between documents
    3. Blend into composite score: (skill × 0.55) + (cosine × 0.45)
    4. ML model predicts resume category with confidence
    5. Generate career path recommendations
"""

import json
import os
import pickle
import logging

from flask import Flask, render_template, request, jsonify

from matching_engine import (
    extract_skills,
    calculate_skill_match,
    calculate_composite_score,
    extract_experience,
    extract_experience_from_jd,
    calculate_experience_score,
    estimate_salary_range,
    extract_education,
    SimilarityEngine,
    SKILL_DATABASE,
)

# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# LOAD ML MODEL + METRICS (once at startup)
# ============================================================

model = None
tfidf = None
model_metrics = {}

try:
    if os.path.exists("model.pkl") and os.path.exists("tfidf.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)
        logger.info("ML model loaded successfully.")
    else:
        logger.warning("model.pkl or tfidf.pkl not found. Running without ML.")
except Exception as e:
    logger.error(f"Error loading ML model: {e}")

try:
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            model_metrics = json.load(f)
        logger.info(f"Model metrics loaded: {model_metrics.get('accuracy', '?')}% accuracy")
except Exception as e:
    logger.error(f"Error loading metrics: {e}")

# Initialize similarity engine once
similarity_engine = SimilarityEngine()

# ============================================================
# CAREER PATHS (domain knowledge)
# ============================================================

CAREER_PATHS = {
    'Backend Developer': {
        'required': ['python', 'sql', 'git', 'rest api'],
        'good_to_have': ['flask', 'django', 'docker', 'postgresql', 'redis', 'linux', 'aws'],
        'salary_range': '₹20,000 - ₹40,000',
        'demand': 'Very High',
    },
    'Frontend Developer': {
        'required': ['html', 'css', 'javascript', 'react', 'git'],
        'good_to_have': ['typescript', 'nextjs', 'tailwind', 'vue', 'angular', 'figma', 'webpack'],
        'salary_range': '₹18,000 - ₹35,000',
        'demand': 'Very High',
    },
    'Full Stack Developer': {
        'required': ['python', 'javascript', 'sql', 'html', 'css', 'git', 'rest api'],
        'good_to_have': ['react', 'flask', 'django', 'mongodb', 'docker', 'aws', 'nodejs'],
        'salary_range': '₹25,000 - ₹50,000',
        'demand': 'High',
    },
    'Data Analyst': {
        'required': ['python', 'sql', 'pandas', 'excel'],
        'good_to_have': ['numpy', 'data science', 'machine learning', 'tableau', 'power bi', 'r'],
        'salary_range': '₹20,000 - ₹45,000',
        'demand': 'High',
    },
    'AI/ML Engineer': {
        'required': ['python', 'machine learning', 'pandas', 'numpy', 'scikit-learn'],
        'good_to_have': ['deep learning', 'tensorflow', 'pytorch', 'nlp', 'opencv', 'docker', 'aws'],
        'salary_range': '₹30,000 - ₹60,000',
        'demand': 'Growing',
    },
    'DevOps Engineer': {
        'required': ['linux', 'docker', 'git', 'ci/cd', 'aws'],
        'good_to_have': ['kubernetes', 'terraform', 'jenkins', 'azure', 'python', 'nginx'],
        'salary_range': '₹25,000 - ₹55,000',
        'demand': 'High',
    },
    'QA/Testing Engineer': {
        'required': ['testing', 'python', 'sql', 'git'],
        'good_to_have': ['unit testing', 'agile', 'jira', 'postman', 'rest api', 'ci/cd'],
        'salary_range': '₹18,000 - ₹35,000',
        'demand': 'High',
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_resume_category(resume_text: str) -> dict:
    """
    Predict resume category with confidence and top alternatives.
    """
    if model is None or tfidf is None:
        return {
            'category': 'ML Model Not Loaded',
            'confidence': 0,
            'confidence_percent': 0,
            'model_accuracy': model_metrics.get('accuracy', 0),
            'top_3': [],
        }

    try:
        import re as _re
        import numpy as np
        cleaned = _re.sub(r'[^a-z ]', ' ', resume_text.lower())
        cleaned = _re.sub(r'\s+', ' ', cleaned).strip()

        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        confidence = 0
        top_3 = []

        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(vectorized)[0]
            confidence = float(max(probas))
            classes = model.classes_
            top_indices = probas.argsort()[-3:][::-1]
            top_3 = [
                {
                    'category': str(classes[i]),
                    'confidence': round(float(probas[i]) * 100, 1)
                }
                for i in top_indices
                if float(probas[i]) > 0.01
            ]
        elif hasattr(model, 'decision_function'):
            decisions = model.decision_function(vectorized)[0]
            if hasattr(decisions, '__len__'):
                exp_d = np.exp(decisions - np.max(decisions))
                pseudo_probs = exp_d / exp_d.sum()
                confidence = float(np.max(pseudo_probs))
                classes = model.classes_
                top_indices = pseudo_probs.argsort()[-3:][::-1]
                top_3 = [
                    {
                        'category': str(classes[i]),
                        'confidence': round(float(pseudo_probs[i]) * 100, 1)
                    }
                    for i in top_indices
                    if float(pseudo_probs[i]) > 0.01
                ]
            else:
                confidence = 0.5

        return {
            'category': str(prediction),
            'confidence': round(confidence, 3),
            'confidence_percent': round(confidence * 100),
            'model_accuracy': model_metrics.get('accuracy', 0),
            'top_3': top_3,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            'category': 'Prediction Error',
            'confidence': 0,
            'confidence_percent': 0,
            'model_accuracy': model_metrics.get('accuracy', 0),
            'top_3': [],
        }

def predict_career_paths(resume_skills: list) -> list:
    """Predict best career paths based on resume skills."""
    career_scores = []
    resume_set = set(s.lower() for s in resume_skills)

    for career, data in CAREER_PATHS.items():
        required = set(data['required'])
        good_to_have = set(data['good_to_have'])

        req_match = len(required & resume_set) / len(required) if required else 0
        good_match = len(good_to_have & resume_set) / len(good_to_have) if good_to_have else 0

        total_score = round((req_match * 70) + (good_match * 30))

        career_scores.append({
            'career': career,
            'score': total_score,
            'salary_range': data['salary_range'],
            'demand': data['demand'],
            'matched_required': sorted(required & resume_set),
            'missing_required': sorted(required - resume_set),
        })

    career_scores.sort(key=lambda x: x['score'], reverse=True)
    return career_scores


def generate_learning_path(missing_skills: list) -> list:
    """Generate prioritized learning roadmap."""
    priority_map = {
        'python': 1, 'sql': 1, 'git': 1,
        'html': 1, 'css': 1, 'javascript': 1,
        'flask': 2, 'django': 2, 'react': 2,
        'rest api': 2, 'linux': 2, 'pandas': 2,
        'mysql': 2, 'postgresql': 2, 'mongodb': 2,
        'testing': 2, 'agile': 2,
        'docker': 3, 'aws': 3, 'kubernetes': 3,
        'ci/cd': 3, 'terraform': 3,
        'machine learning': 3, 'deep learning': 3,
        'microservices': 3, 'system design': 3,
    }

    time_map = {1: '1-2 weeks', 2: '2-4 weeks', 3: '1-2 months', 4: '2-4 weeks'}
    label_map = {1: 'Learn First', 2: 'Learn Next', 3: 'Learn Later', 4: 'Optional'}

    result = []
    for skill in missing_skills:
        p = priority_map.get(skill, 4)
        result.append({
            'skill': skill,
            'priority': p,
            'time': time_map.get(p, '2-4 weeks'),
            'priority_label': label_map.get(p, 'Optional'),
        })

    result.sort(key=lambda x: x['priority'])
    return result[:10]


def generate_summary(
    match_score, skill_score, cosine_percent,
    matched_skills, missing_skills,
    experience, education, top_career, prediction
):
    """Generate executive summary."""
    if match_score >= 75:
        assessment = "EXCELLENT FIT"
        recommendation = (
            "This candidate is highly suitable for the role. "
            "Recommend moving forward to interview."
        )
    elif match_score >= 50:
        assessment = "STRONG FIT"
        recommendation = (
            "This candidate shows good alignment with the role. "
            "Consider interview with focus on missing skills."
        )
    elif match_score >= 30:
        assessment = "MODERATE FIT"
        recommendation = (
            "This candidate has some relevant skills but also "
            "noticeable gaps. Suitable if training is possible."
        )
    else:
        assessment = "LOW FIT"
        recommendation = (
            "This candidate does not strongly align with this role "
            "based on the current analysis."
        )

    strengths = []
    if len(matched_skills) >= 3:
        strengths.append(f"Has {len(matched_skills)} matching technical skills")
    if experience >= 0 and experience <= 2:
        strengths.append("Experience level suitable for fresher/junior roles")
    if education in ['Bachelors', 'Masters', 'PhD/Doctorate']:
        strengths.append(f"Education: {education}")
    if cosine_percent >= 40:
        strengths.append(f"High semantic similarity ({cosine_percent}%) with job description")
    if skill_score >= 60:
        strengths.append("Strong skill alignment with requirements")
    if prediction['category'] not in ["ML Model Not Loaded", "Prediction Error"]:
        strengths.append(
            f"ML prediction: {prediction['category']} "
            f"({prediction['confidence_percent']}% confidence)"
        )

    concerns = []
    if len(missing_skills) > 3:
        concerns.append(f"Missing {len(missing_skills)} required or preferred skills")
    if experience == -1:
        concerns.append("Experience level not clearly detected")
    if skill_score < 40:
        concerns.append("Skill match below recommended level")
    if education == 'Not Detected':
        concerns.append("Education details unclear or missing")
    if cosine_percent < 20:
        concerns.append("Low semantic overlap with job description")

    return {
        'assessment': assessment,
        'recommendation': recommendation,
        'strengths': strengths if strengths else ['Resume submitted for analysis'],
        'concerns': concerns if concerns else ['No major concerns identified'],
        'top_career': top_career,
    }


def run_full_analysis(job_description: str, resume_text: str) -> dict:
    """
    Core analysis pipeline. Used by BOTH web UI and REST API.
    """
    # 1. Skill extraction
    job_skills_by_cat, job_skills_flat = extract_skills(job_description)
    resume_skills_by_cat, resume_skills_flat = extract_skills(resume_text)

    # 2. Skill match
    skill_score, matched_skills, missing_skills, extra_skills = calculate_skill_match(
        job_skills_flat, resume_skills_flat
    )

    # 3. Semantic similarity
    similarity = similarity_engine.compute_similarity(job_description, resume_text)

    # 4. Experience matching (NEW)
    resume_experience = extract_experience(resume_text)
    jd_experience = extract_experience_from_jd(job_description)
    experience_match = calculate_experience_score(resume_experience, jd_experience)

    # 5. Education
    education = extract_education(resume_text)

    # 6. Composite score (now includes experience)
    match_score, match_level = calculate_composite_score(
        skill_score,
        similarity['cosine_percent'],
        experience_match['score'],
    )

    # 7. ML prediction
    prediction = predict_resume_category(resume_text)

    # 8. Career paths
    career_paths = predict_career_paths(resume_skills_flat)
    top_career = career_paths[0] if career_paths else None

    # 9. Salary estimation (NEW)
    salary = estimate_salary_range(
        resume_experience if resume_experience >= 0 else 0,
        matched_skills,
        education,
        top_career['career'] if top_career else ''
    )

    # 10. Learning path
    learning_path = generate_learning_path(missing_skills)

    # 11. Summary
    summary = generate_summary(
        match_score, skill_score, similarity['cosine_percent'],
        matched_skills, missing_skills,
        resume_experience, education, top_career, prediction
    )

    return {
        'match_score': match_score,
        'match_level': match_level,
        'skill_score': skill_score,
        'keyword_score': similarity['cosine_percent'],
        'cosine_similarity': similarity['cosine_similarity'],
        'cosine_percent': similarity['cosine_percent'],
        'similarity_method': similarity['method'],
        'shared_terms': similarity['shared_important_terms'],
        'job_distinctive_terms': similarity['job_distinctive_terms'],
        'resume_distinctive_terms': similarity['resume_distinctive_terms'],
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'extra_skills': extra_skills,
        'resume_skills': resume_skills_by_cat,
        'job_skills': job_skills_flat,
        'total_job_skills': len(job_skills_flat),
        'total_matched': len(matched_skills),
        'career_paths': career_paths[:3],
        'learning_path': learning_path,
        'experience': resume_experience,
        'experience_match': experience_match,
        'education': education,
        'salary': salary,
        'prediction': prediction,
        'summary': summary,
        'model_metrics': model_metrics,
    }

# ============================================================
# WEB ROUTES
# ============================================================

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '').strip()
        resume_text = request.form.get('resume', '').strip()

        if not resume_text:
            return render_template('index.html', error="Please paste your resume text.")

        if not job_description:
            return render_template(
                'index.html',
                error="Please paste the job description.",
                resume_text=resume_text
            )

        if len(resume_text.split()) < 30:
            return render_template(
                'index.html',
                error=f"Resume too short ({len(resume_text.split())} words). Minimum 30.",
                resume_text=resume_text,
                job_description=job_description
            )

        if len(job_description.split()) < 10:
            return render_template(
                'index.html',
                error=f"Job description too short ({len(job_description.split())} words). Minimum 10.",
                resume_text=resume_text,
                job_description=job_description
            )

        try:
            result = run_full_analysis(job_description, resume_text)
            return render_template(
                'index.html',
                resume_text=resume_text,
                job_description=job_description,
                **result
            )
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return render_template(
                'index.html',
                error=f"Analysis error: {str(e)}",
                resume_text=resume_text,
                job_description=job_description
            )

    return render_template('index.html')


# ============================================================
# REST API
# ============================================================

@app.route('/api/v1/match', methods=['POST'])
def api_match():
    """
    Headless JSON API for resume-job matching.

    Request:
        POST /api/v1/match
        Content-Type: application/json
        {
            "job_description": "...",
            "resume": "..."
        }

    Response:
        200: Full analysis as JSON
        400: Validation error
    """
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json', 'status': 400}), 400

    data = request.get_json()
    job_desc = data.get('job_description', '').strip()
    resume = data.get('resume', '').strip()

    if not job_desc or not resume:
        return jsonify({
            'error': 'Both job_description and resume are required',
            'status': 400
        }), 400

    if len(resume.split()) < 10:
        return jsonify({'error': 'Resume must be at least 10 words', 'status': 400}), 400

    try:
        result = run_full_analysis(job_desc, resume)
        return jsonify({'status': 200, 'data': result})
    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return jsonify({'error': str(e), 'status': 500}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check for monitoring."""
    return jsonify({
        'status': 'healthy',
        'service': 'resume-screener',
        'ml_model': 'loaded' if model else 'not loaded',
        'model_accuracy': model_metrics.get('accuracy', 'N/A'),
    })


if __name__ == '__main__':
    app.run(debug=True)