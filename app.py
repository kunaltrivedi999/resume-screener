from flask import Flask, render_template, request
import re
import pickle
import os

app = Flask(__name__)

# ============================================
# SECTION 1: LOAD ML MODEL
# ============================================

model = None
tfidf = None

try:
    if os.path.exists("model.pkl") and os.path.exists("tfidf.pkl"):
        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("tfidf.pkl", "rb") as tfidf_file:
            tfidf = pickle.load(tfidf_file)
        print("ML model loaded successfully.")
    else:
        print("model.pkl or tfidf.pkl not found. Running without ML prediction.")
except Exception as e:
    print(f"Error loading ML model: {e}")


# ============================================
# SECTION 2: SKILL DATABASE
# ============================================

SKILL_DATABASE = {
    'programming_languages': {
        'skills': [
            'python', 'java', 'javascript', 'typescript',
            'c++', 'c#', 'go', 'rust', 'ruby', 'php',
            'swift', 'kotlin', 'r', 'scala', 'sql',
            'html', 'css'
        ],
        'weight': 3
    },
    'web_frameworks': {
        'skills': [
            'flask', 'django', 'react', 'angular', 'vue',
            'nextjs', 'express', 'spring', 'fastapi',
            'bootstrap', 'tailwind', 'jquery', 'nodejs',
            'laravel', 'dotnet'
        ],
        'weight': 3
    },
    'databases': {
        'skills': [
            'mysql', 'postgresql', 'mongodb', 'redis',
            'sqlite', 'firebase', 'oracle', 'cassandra',
            'dynamodb', 'elasticsearch'
        ],
        'weight': 2
    },
    'devops_cloud': {
        'skills': [
            'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'linux', 'jenkins', 'terraform', 'nginx',
            'ci/cd', 'github actions', 'heroku', 'vercel'
        ],
        'weight': 2
    },
    'ai_ml': {
        'skills': [
            'machine learning', 'deep learning', 'nlp',
            'tensorflow', 'pytorch', 'pandas', 'numpy',
            'scikit-learn', 'opencv', 'keras',
            'data science', 'neural network'
        ],
        'weight': 2
    },
    'tools': {
        'skills': [
            'git', 'github', 'gitlab', 'postman', 'jira',
            'figma', 'vs code', 'vim', 'webpack', 'babel',
            'npm', 'pip', 'conda', 'excel', 'tableau', 'power bi'
        ],
        'weight': 1
    },
    'concepts': {
        'skills': [
            'rest api', 'graphql', 'microservices', 'agile',
            'scrum', 'testing', 'unit testing', 'tdd',
            'oop', 'design patterns', 'data structures',
            'algorithms', 'system design', 'mvc'
        ],
        'weight': 2
    }
}


# ============================================
# SECTION 3: CAREER PATHS
# ============================================

CAREER_PATHS = {
    'Backend Developer': {
        'required': ['python', 'sql', 'git', 'rest api'],
        'good_to_have': [
            'flask', 'django', 'docker', 'postgresql',
            'redis', 'linux', 'aws'
        ],
        'salary_range': '₹20,000 - ₹40,000',
        'demand': 'Very High',
        'description': 'Build server-side logic and APIs'
    },
    'Frontend Developer': {
        'required': [
            'html', 'css', 'javascript', 'react', 'git'
        ],
        'good_to_have': [
            'typescript', 'nextjs', 'tailwind', 'vue',
            'angular', 'figma', 'webpack'
        ],
        'salary_range': '₹18,000 - ₹35,000',
        'demand': 'Very High',
        'description': 'Build user interfaces and web pages'
    },
    'Full Stack Developer': {
        'required': [
            'python', 'javascript', 'sql', 'html',
            'css', 'git', 'rest api'
        ],
        'good_to_have': [
            'react', 'flask', 'django', 'mongodb',
            'docker', 'aws', 'nodejs'
        ],
        'salary_range': '₹25,000 - ₹50,000',
        'demand': 'High',
        'description': 'Build complete web applications'
    },
    'Data Analyst': {
        'required': ['python', 'sql', 'pandas', 'excel'],
        'good_to_have': [
            'numpy', 'data science', 'machine learning',
            'tableau', 'power bi', 'r'
        ],
        'salary_range': '₹20,000 - ₹45,000',
        'demand': 'High',
        'description': 'Analyze data and generate insights'
    },
    'AI/ML Engineer': {
        'required': [
            'python', 'machine learning', 'pandas',
            'numpy', 'scikit-learn'
        ],
        'good_to_have': [
            'deep learning', 'tensorflow', 'pytorch',
            'nlp', 'opencv', 'docker', 'aws',
            'neural network'
        ],
        'salary_range': '₹30,000 - ₹60,000',
        'demand': 'Growing',
        'description': 'Build AI and ML models'
    },
    'DevOps Engineer': {
        'required': [
            'linux', 'docker', 'git', 'ci/cd', 'aws'
        ],
        'good_to_have': [
            'kubernetes', 'terraform', 'jenkins',
            'azure', 'python', 'nginx',
            'github actions'
        ],
        'salary_range': '₹25,000 - ₹55,000',
        'demand': 'High',
        'description': 'Manage deployment and infrastructure'
    },
    'QA/Testing Engineer': {
        'required': [
            'testing', 'python', 'sql', 'git'
        ],
        'good_to_have': [
            'unit testing', 'agile', 'jira',
            'selenium', 'postman', 'rest api',
            'ci/cd'
        ],
        'salary_range': '₹18,000 - ₹35,000',
        'demand': 'High',
        'description': 'Test software and ensure quality'
    }
}


# ============================================
# SECTION 4: TEXT CLEANING
# ============================================

def clean_text_logic(text):
    """Clean text for rule-based logic."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s/#+.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text_ml(text):
    """Clean text for ML model."""
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================
# SECTION 5: ML PREDICTION
# ============================================

def predict_resume_category(resume_text):
    """Predict resume category using trained ML model."""
    if model is None or tfidf is None:
        return "ML Model Not Loaded"

    try:
        cleaned_resume = clean_text_ml(resume_text)
        vectorized_resume = tfidf.transform([cleaned_resume])
        prediction = model.predict(vectorized_resume)[0]
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction Error"


# ============================================
# SECTION 6: SKILL EXTRACTION
# ============================================

def extract_skills(text):
    """Extract all skills found in text."""
    text_lower = clean_text_logic(text)
    found_skills = {}
    all_found = []

    for category, data in SKILL_DATABASE.items():
        category_found = []

        for skill in data['skills']:
            if skill in text_lower:
                category_found.append(skill)
                all_found.append(skill)
            elif skill == 'javascript' and 'js' in text_lower.split():
                category_found.append(skill)
                all_found.append(skill)
            elif skill == 'typescript' and 'ts' in text_lower.split():
                category_found.append(skill)
                all_found.append(skill)
            elif skill == 'machine learning' and 'ml' in text_lower.split():
                category_found.append(skill)
                all_found.append(skill)
            elif skill == 'nodejs' and ('node' in text_lower.split() or 'node.js' in text_lower):
                category_found.append(skill)
                all_found.append(skill)

        if category_found:
            found_skills[category] = category_found

    return found_skills, list(set(all_found))


def extract_experience(text):
    """Detect years of experience from text."""
    text_lower = text.lower()

    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*[:\-]?\s*(\d+)\s*years?',
        r'(\d+)\+?\s*years?\s*(?:of)?\s*(?:work|industry)',
        r'(\d+)\+?\s*yrs?\s*(?:of)?\s*exp',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return int(match.group(1))

    fresher_keywords = [
        'fresher', 'fresh graduate', 'entry level',
        'no experience', 'seeking first',
        'recent graduate', 'just graduated'
    ]
    for keyword in fresher_keywords:
        if keyword in text_lower:
            return 0

    return -1


def extract_education(text):
    """Detect education level from text."""
    text_lower = text.lower()

    education_levels = {
        'PhD/Doctorate': [
            'phd', 'doctorate', 'doctoral', 'ph.d'
        ],
        'Masters': [
            'master', 'mtech', 'm.tech', 'mca', 'm.c.a',
            'mba', 'm.b.a', 'msc', 'm.sc', 'ms '
        ],
        'Bachelors': [
            'btech', 'b.tech', 'bsc', 'b.sc', 'bca',
            'b.c.a', 'be ', 'b.e.', 'bachelor',
            'bsc it', 'bsc cs'
        ],
        'Diploma': [
            'diploma', 'polytechnic'
        ],
        'Higher Secondary': [
            '12th', 'hsc', 'higher secondary',
            'intermediate', '+2'
        ]
    }

    detected = []
    for level, keywords in education_levels.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(level)
                break

    if detected:
        priority = [
            'PhD/Doctorate', 'Masters', 'Bachelors',
            'Diploma', 'Higher Secondary'
        ]
        for level in priority:
            if level in detected:
                return level

    return 'Not Detected'


# ============================================
# SECTION 7: MATCHING ENGINE
# ============================================

def calculate_skill_match(job_skills, resume_skills):
    """Calculate skill match score."""
    if not job_skills:
        return 0, [], []

    job_set = set(s.lower() for s in job_skills)
    resume_set = set(s.lower() for s in resume_skills)

    matched = job_set & resume_set
    missing = job_set - resume_set

    if len(job_set) == 0:
        return 0, [], []

    score = (len(matched) / len(job_set)) * 100

    return round(score), list(matched), list(missing)


def calculate_keyword_similarity(job_text, resume_text):
    """Calculate keyword similarity."""
    job_words = set(clean_text_logic(job_text).split())
    resume_words = set(clean_text_logic(resume_text).split())

    stop_words = {
        'the', 'is', 'a', 'an', 'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'of', 'it',
        'that', 'this', 'are', 'was', 'be', 'have',
        'with', 'can', 'not', 'they', 'we', 'you',
        'will', 'should', 'would', 'could', 'may',
        'must', 'shall', 'do', 'does', 'did', 'has',
        'had', 'been', 'being', 'having', 'each',
        'which', 'their', 'there', 'from', 'what',
        'who', 'how', 'when', 'where', 'why', 'all',
        'about', 'our', 'your', 'more', 'also',
        'very', 'just', 'than', 'then', 'into',
        'some', 'such', 'only', 'other', 'over',
        'after', 'before', 'between', 'under',
        'above', 'any', 'both', 'through', 'during',
        'own', 'same', 'so', 'too', 'no', 'nor',
        'up', 'out', 'if', 'as', 'by'
    }

    job_words -= stop_words
    resume_words -= stop_words

    if not job_words:
        return 0

    overlap = job_words & resume_words
    score = (len(overlap) / len(job_words)) * 100

    return round(min(score, 100))


def predict_career_paths(resume_skills):
    """Predict best career paths based on resume skills."""
    career_scores = []

    for career, data in CAREER_PATHS.items():
        required = set(data['required'])
        good_to_have = set(data['good_to_have'])
        resume_set = set(s.lower() for s in resume_skills)

        req_match = len(required & resume_set) / len(required) if required else 0
        good_match = len(good_to_have & resume_set) / len(good_to_have) if good_to_have else 0

        total_score = (req_match * 70) + (good_match * 30)

        matched_required = list(required & resume_set)
        matched_good = list(good_to_have & resume_set)
        missing_required = list(required - resume_set)
        missing_good = list(good_to_have - resume_set)

        career_scores.append({
            'career': career,
            'score': round(total_score),
            'description': data['description'],
            'salary_range': data['salary_range'],
            'demand': data['demand'],
            'matched_required': matched_required,
            'matched_good': matched_good,
            'missing_required': missing_required,
            'missing_good': missing_good[:5],
            'total_required': len(required),
            'total_matched': len(matched_required)
        })

    career_scores.sort(key=lambda x: x['score'], reverse=True)
    return career_scores


def generate_learning_path(missing_skills):
    """Generate learning path for missing skills."""
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
        'microservices': 3, 'system design': 3
    }

    prioritized = []
    for skill in missing_skills:
        priority = priority_map.get(skill, 4)
        time_estimate = {
            1: '1-2 weeks',
            2: '2-4 weeks',
            3: '1-2 months',
            4: '2-4 weeks'
        }.get(priority, '2-4 weeks')

        prioritized.append({
            'skill': skill,
            'priority': priority,
            'time': time_estimate,
            'priority_label': {
                1: 'Learn First',
                2: 'Learn Next',
                3: 'Learn Later',
                4: 'Optional'
            }.get(priority, 'Optional')
        })

    prioritized.sort(key=lambda x: x['priority'])
    return prioritized[:10]


def generate_summary(
    match_score, skill_score, keyword_score,
    matched_skills, missing_skills,
    experience, education, top_career, predicted_category
):
    """Generate summary report."""
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
        strengths.append("Experience level is suitable for fresher/junior roles")
    if education in ['Bachelors', 'Masters', 'PhD/Doctorate']:
        strengths.append(f"Education qualification detected: {education}")
    if skill_score >= 60:
        strengths.append("Strong skill alignment with the job description")
    if predicted_category not in ["ML Model Not Loaded", "Prediction Error"]:
        strengths.append(f"ML model predicts candidate category as: {predicted_category}")

    concerns = []
    if len(missing_skills) > 3:
        concerns.append(f"Missing {len(missing_skills)} required or preferred skills")
    if experience == -1:
        concerns.append("Experience level not clearly detected from resume")
    if skill_score < 40:
        concerns.append("Skill match is below the recommended level")
    if education == 'Not Detected':
        concerns.append("Education details are unclear or missing")

    return {
        'assessment': assessment,
        'recommendation': recommendation,
        'strengths': strengths if strengths else ['Resume submitted for analysis'],
        'concerns': concerns if concerns else ['No major concerns identified'],
        'top_career': top_career
    }


# ============================================
# SECTION 8: FLASK ROUTE
# ============================================

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '').strip()
        resume_text = request.form.get('resume', '').strip()

        if not resume_text:
            return render_template(
                'index.html',
                error="Please paste your resume text."
            )

        if not job_description:
            return render_template(
                'index.html',
                error="Please paste the job description.",
                resume_text=resume_text
            )

        resume_words = len(resume_text.split())
        if resume_words < 30:
            return render_template(
                'index.html',
                error=f"Resume too short ({resume_words} words). Paste complete resume (minimum 30 words).",
                resume_text=resume_text,
                job_description=job_description
            )

        job_words = len(job_description.split())
        if job_words < 10:
            return render_template(
                'index.html',
                error=f"Job description too short ({job_words} words). Paste complete job description.",
                resume_text=resume_text,
                job_description=job_description
            )

        # ML category prediction
        predicted_category = predict_resume_category(resume_text)

        # Rule-based analysis
        job_skills_by_cat, job_skills_flat = extract_skills(job_description)
        resume_skills_by_cat, resume_skills_flat = extract_skills(resume_text)

        skill_score, matched_skills, missing_skills = calculate_skill_match(
            job_skills_flat, resume_skills_flat
        )
        keyword_score = calculate_keyword_similarity(job_description, resume_text)

        match_score = round((skill_score * 0.6) + (keyword_score * 0.4))

        experience = extract_experience(resume_text)
        education = extract_education(resume_text)

        career_paths = predict_career_paths(resume_skills_flat)
        top_career = career_paths[0] if career_paths else None

        learning_path = generate_learning_path(missing_skills)

        extra_skills = list(set(resume_skills_flat) - set(job_skills_flat))

        summary = generate_summary(
            match_score, skill_score, keyword_score,
            matched_skills, missing_skills,
            experience, education, top_career, predicted_category
        )

        if match_score >= 75:
            match_level = 'strong'
        elif match_score >= 50:
            match_level = 'good'
        elif match_score >= 30:
            match_level = 'partial'
        else:
            match_level = 'weak'

        return render_template(
            'index.html',
            match_score=match_score,
            skill_score=skill_score,
            keyword_score=keyword_score,
            match_level=match_level,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            extra_skills=extra_skills,
            resume_skills=resume_skills_by_cat,
            job_skills=job_skills_flat,
            total_job_skills=len(job_skills_flat),
            total_matched=len(matched_skills),
            career_paths=career_paths[:3],
            learning_path=learning_path,
            experience=experience,
            education=education,
            summary=summary,
            predicted_category=predicted_category,
            resume_text=resume_text,
            job_description=job_description
        )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)