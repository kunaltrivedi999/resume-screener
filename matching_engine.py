import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Set

from sklearn.feature_extraction.text import TfidfVectorizer




# ============================================================
# SECTION 1: SKILL EXTRACTION (Fixed)
# ============================================================

# The complete skill database with weights
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

# Common aliases that should map to canonical skill names
SKILL_ALIASES = {
    'js': 'javascript',
    'ts': 'typescript',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'node': 'nodejs',
    'node.js': 'nodejs',
    'react.js': 'react',
    'reactjs': 'react',
    'vue.js': 'vue',
    'vuejs': 'vue',
    'angular.js': 'angular',
    'angularjs': 'angular',
    'next.js': 'nextjs',
    'express.js': 'express',
    'mongo': 'mongodb',
    'postgres': 'postgresql',
    'k8s': 'kubernetes',
    'sklearn': 'scikit-learn',
    'sk-learn': 'scikit-learn',
    'tf': 'tensorflow',
    'np': 'numpy',
    'pd': 'pandas',
}


def _clean_for_extraction(text: str) -> str:
    """
    Clean text while preserving special characters needed for
    skill detection (++, #, /, .).
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s/#+.\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _skill_matches(skill: str, text: str) -> bool:
    """
    Check if a skill appears in text as a STANDALONE term.

    This fixes the v1.0 false positive bug where:
      - 'r' matched 'programmer' (substring match)
      - 'go' matched 'google' (substring match)
      - 'java' matched 'javascript' (substring match)

    Fix: Use word boundary regex (\\b) for alphabetic skills.
    For skills with special characters (c++, c#, ci/cd), use
    context-aware matching with whitespace/boundary checks.
    """
    # Skills containing special chars: match literally with boundary context
    if not skill.replace(' ', '').isalnum():
        # Build pattern: skill preceded by start/whitespace, followed by end/whitespace
        escaped = re.escape(skill)
        pattern = r'(?:^|[\s,;:(]|(?<=\s))' + escaped + r'(?:[\s,;:)]|$)'
        return bool(re.search(pattern, text))

    # Pure alphabetic skills (possibly multi-word): use word boundaries
    pattern = r'\b' + re.escape(skill) + r'\b'
    return bool(re.search(pattern, text))


def extract_skills(text: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Extract all recognized skills from text using word-boundary matching.

    Returns:
        Tuple of (categorized_skills_dict, flat_skills_list)

    Example:
        categorized = {'programming_languages': ['python', 'java'], ...}
        flat = ['python', 'java', 'flask', 'git']
    """
    cleaned = _clean_for_extraction(text)
    found_skills = {}
    all_found = set()

    # First, check for aliases and expand them
    words = cleaned.split()
    for word in words:
        if word in SKILL_ALIASES:
            canonical = SKILL_ALIASES[word]
            all_found.add(canonical)

    # Then check each skill in the database
    for category, data in SKILL_DATABASE.items():
        category_found = []
        for skill in data['skills']:
            if skill in all_found or _skill_matches(skill, cleaned):
                category_found.append(skill)
                all_found.add(skill)

        if category_found:
            found_skills[category] = sorted(category_found)

    return found_skills, sorted(list(all_found))


def calculate_skill_match(
    job_skills: List[str],
    resume_skills: List[str]
) -> Tuple[int, List[str], List[str], List[str]]:
    """
    Calculate skill overlap between job requirements and resume.

    Returns:
        (score_0_to_100, matched_list, missing_list, extra_list)
    """
    if not job_skills:
        return 0, [], [], list(resume_skills)

    job_set = set(s.lower() for s in job_skills)
    resume_set = set(s.lower() for s in resume_skills)

    matched = sorted(job_set & resume_set)
    missing = sorted(job_set - resume_set)
    extra = sorted(resume_set - job_set)

    score = round((len(matched) / len(job_set)) * 100) if job_set else 0

    return score, matched, missing, extra


# ============================================================
# SECTION 2: TF-IDF COSINE SIMILARITY (The Math Upgrade)
# ============================================================
class SimilarityEngine:
    """
    Dual-engine similarity: SBERT (primary) + TF-IDF (fallback).
    
    Why SBERT is better:
        TF-IDF:  "Built predictive models" vs "Developed ML systems"  → 0.05
        SBERT:   "Built predictive models" vs "Developed ML systems"  → 0.62
        
    SBERT encodes meaning into vectors. TF-IDF only matches exact words.
    """

    def __init__(self):
        self.use_sbert = False
        self.sbert_model = None
        self.method = "TF-IDF Cosine Similarity"

        try:
            from sentence_transformers import SentenceTransformer
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_sbert = True
            self.method = "Sentence-BERT (all-MiniLM-L6-v2)"
            print("[Similarity] SBERT loaded successfully")
        except ImportError:
            print("[Similarity] sentence-transformers not installed → using TF-IDF")
        except Exception as e:
            print(f"[Similarity] SBERT failed: {e} → using TF-IDF")

        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            norm='l2'
        )

    def compute_similarity(self, text_a: str, text_b: str) -> dict:
        clean_a = self._clean(text_a)
        clean_b = self._clean(text_b)

        if not clean_a.strip() or not clean_b.strip():
            return {
                'cosine_similarity': 0.0,
                'cosine_percent': 0,
                'method': self.method,
                'shared_important_terms': [],
                'job_distinctive_terms': [],
                'resume_distinctive_terms': [],
            }

        if self.use_sbert:
            score = self._sbert_score(clean_a, clean_b)
        else:
            score = self._tfidf_score(clean_a, clean_b)

        terms = self._get_terms(clean_a, clean_b)

        return {
            'cosine_similarity': round(score, 4),
            'cosine_percent': round(score * 100),
            'method': self.method,
            'shared_important_terms': terms['shared'],
            'job_distinctive_terms': terms['job_only'],
            'resume_distinctive_terms': terms['resume_only'],
        }

    def _sbert_score(self, text_a: str, text_b: str) -> float:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        embeddings = self.sbert_model.encode([text_a, text_b])
        score = float(cos_sim([embeddings[0]], [embeddings[1]])[0][0])
        return max(0.0, min(1.0, score))

    def _tfidf_score(self, text_a: str, text_b: str) -> float:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        matrix = self.tfidf.fit_transform([text_a, text_b])
        score = float(cos_sim(matrix[0:1], matrix[1:2])[0][0])
        return max(0.0, min(1.0, score))

    def _get_terms(self, text_a: str, text_b: str) -> dict:
        try:
            analyzer = TfidfVectorizer(
                max_features=3000,
                stop_words='english',
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            matrix = analyzer.fit_transform([text_a, text_b])
            names = analyzer.get_feature_names_out()
            jv = matrix[0].toarray()[0]
            rv = matrix[1].toarray()[0]

            shared = []
            job_only = []
            resume_only = []

            for i, term in enumerate(names):
                if jv[i] > 0.05 and rv[i] > 0.05:
                    shared.append((term, (jv[i] + rv[i]) / 2))
                elif jv[i] > 0.1 and rv[i] < 0.01:
                    job_only.append((term, jv[i]))
                elif rv[i] > 0.1 and jv[i] < 0.01:
                    resume_only.append((term, rv[i]))

            shared.sort(key=lambda x: x[1], reverse=True)
            job_only.sort(key=lambda x: x[1], reverse=True)
            resume_only.sort(key=lambda x: x[1], reverse=True)

            return {
                'shared': [t[0] for t in shared[:10]],
                'job_only': [t[0] for t in job_only[:8]],
                'resume_only': [t[0] for t in resume_only[:8]],
            }
        except Exception:
            return {'shared': [], 'job_only': [], 'resume_only': []}

    def _clean(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ============================================================
# SECTION 3: COMPOSITE SCORING
# ============================================================

def calculate_composite_score(
    skill_score: int,
    cosine_percent: int,
    experience_score: int = 50,
) -> Tuple[int, str]:
    """
    Compute final match score blending all signals.
    
    Formula:
        composite = (skill × 0.40) + (semantic × 0.30) + (experience × 0.20) + (education_base × 0.10)
    
    Weight Justification:
        Skills (40%):      Direct, verifiable match
        Semantic (30%):    Contextual relevance beyond keywords
        Experience (20%):  A fresher applying for a senior role should score lower
        Base (10%):        Reserved for education (passed as part of experience_score)
    """
    composite = round(
        (skill_score * 0.40) +
        (cosine_percent * 0.30) +
        (experience_score * 0.30)
    )
    composite = max(0, min(100, composite))

    if composite >= 75:
        level = 'strong'
    elif composite >= 50:
        level = 'good'
    elif composite >= 30:
        level = 'partial'
    else:
        level = 'weak'

    return composite, level
# ============================================================
# SECTION 4: EXPERIENCE & EDUCATION EXTRACTION
# ============================================================

def extract_experience(text: str) -> int:
    """
    Detect years of experience using regex pattern matching.

    Returns:
        int: Years of experience. 0 = fresher, -1 = not detected.
    """
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
            years = int(match.group(1))
            return min(years, 50)  # Cap at 50 to prevent garbage data

    fresher_keywords = [
        'fresher', 'fresh graduate', 'entry level',
        'no experience', 'seeking first',
        'recent graduate', 'just graduated'
    ]
    for keyword in fresher_keywords:
        if keyword in text_lower:
            return 0

    return -1


def extract_education(text: str) -> str:
    """
    Detect highest education level from resume text.

    Returns:
        str: Education level name (e.g., 'Bachelors', 'Masters')
    """
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

    priority = [
        'PhD/Doctorate', 'Masters', 'Bachelors',
        'Diploma', 'Higher Secondary'
    ]

    for level in priority:
        for keyword in education_levels[level]:
            if keyword in text_lower:
                return level

    return 'Not Detected'
# ============================================================
# SECTION 5: EXPERIENCE SCORING
# ============================================================

def extract_experience_from_jd(text: str) -> dict:
    """
    Extract experience requirements from job description.
    
    Detects patterns like:
        "2-4 years experience"
        "minimum 3 years"
        "0-1 years (fresher friendly)"
        "5+ years required"
    
    Returns:
        {'min_years': 2, 'max_years': 4, 'fresher_ok': False}
    """
    text_lower = text.lower()
    
    # Pattern: "X-Y years" or "X to Y years"
    range_patterns = [
        r'(\d+)\s*[-–to]+\s*(\d+)\s*\+?\s*years?',
        r'(\d+)\s*[-–to]+\s*(\d+)\s*\+?\s*yrs?',
    ]
    for pattern in range_patterns:
        match = re.search(pattern, text_lower)
        if match:
            min_y = int(match.group(1))
            max_y = int(match.group(2))
            return {
                'min_years': min_y,
                'max_years': max_y,
                'fresher_ok': min_y == 0,
                'detected': True,
            }
    
    # Pattern: "X+ years" or "minimum X years" or "at least X years"
    min_patterns = [
        r'(\d+)\+\s*years?\s*(?:of)?\s*experience',
        r'(?:minimum|at\s*least|min)\s*(\d+)\s*years?',
        r'(\d+)\s*years?\s*(?:of)?\s*(?:minimum|required)',
    ]
    for pattern in min_patterns:
        match = re.search(pattern, text_lower)
        if match:
            min_y = int(match.group(1))
            return {
                'min_years': min_y,
                'max_years': min_y + 3,
                'fresher_ok': min_y == 0,
                'detected': True,
            }
    
    # Pattern: just "X years experience"
    simple_patterns = [
        r'(\d+)\s*years?\s*(?:of)?\s*experience',
        r'experience\s*[:\-]?\s*(\d+)\s*years?',
    ]
    for pattern in simple_patterns:
        match = re.search(pattern, text_lower)
        if match:
            years = int(match.group(1))
            return {
                'min_years': max(0, years - 1),
                'max_years': years + 2,
                'fresher_ok': years <= 1,
                'detected': True,
            }
    
    # Check for fresher-friendly keywords
    fresher_keywords = [
        'fresher', 'entry level', 'entry-level', 'junior',
        'graduate', 'no experience required', '0 experience',
        'internship', 'trainee', 'beginner'
    ]
    for keyword in fresher_keywords:
        if keyword in text_lower:
            return {
                'min_years': 0,
                'max_years': 2,
                'fresher_ok': True,
                'detected': True,
            }
    
    # Check for senior keywords
    senior_keywords = [
        'senior', 'lead', 'principal', 'architect',
        'staff engineer', 'director', 'manager'
    ]
    for keyword in senior_keywords:
        if keyword in text_lower:
            return {
                'min_years': 5,
                'max_years': 15,
                'fresher_ok': False,
                'detected': True,
            }
    
    # Nothing detected
    return {
        'min_years': 0,
        'max_years': 99,
        'fresher_ok': True,
        'detected': False,
    }


def calculate_experience_score(
    resume_years: int,
    jd_requirements: dict
) -> dict:
    """
    Score how well the candidate's experience matches the job.
    
    Scoring logic:
        Perfect fit (within range)     → 100
        Slightly below (1 year short)  → 60
        Way below (2+ years short)     → 30
        Overqualified (above range)    → 80 (still good, slight penalty)
        Fresher for fresher role       → 100
        Fresher for senior role        → 10
        No data from either side       → 50 (neutral)
    
    Returns:
        {
            'score': 0-100,
            'verdict': 'Perfect Match' / 'Underqualified' / etc,
            'resume_years': 3,
            'required_range': '2-4 years',
            'details': 'Candidate has 3 years, job requires 2-4 years'
        }
    """
    min_req = jd_requirements['min_years']
    max_req = jd_requirements['max_years']
    detected = jd_requirements['detected']
    fresher_ok = jd_requirements['fresher_ok']
    
    # If experience not detected from resume
    if resume_years == -1:
        return {
            'score': 50,
            'verdict': 'Experience Not Detected',
            'resume_years': -1,
            'required_range': f"{min_req}-{max_req} years" if detected else "Not specified",
            'details': 'Could not detect experience level from resume',
        }
    
    # If JD doesn't specify experience
    if not detected:
        return {
            'score': 70,
            'verdict': 'No Requirement Specified',
            'resume_years': resume_years,
            'required_range': 'Not specified',
            'details': f'Candidate has {resume_years} years, job has no specific requirement',
        }
    
    # Fresher applying
    if resume_years == 0:
        if fresher_ok:
            score = 100
            verdict = 'Perfect Match — Fresher Welcome'
        elif min_req <= 2:
            score = 50
            verdict = 'Slightly Underqualified'
        else:
            score = 10
            verdict = 'Significantly Underqualified'
        
        return {
            'score': score,
            'verdict': verdict,
            'resume_years': 0,
            'required_range': f"{min_req}-{max_req} years",
            'details': f'Fresher applying for role requiring {min_req}-{max_req} years',
        }
    
    # Within range — perfect
    if min_req <= resume_years <= max_req:
        return {
            'score': 100,
            'verdict': 'Perfect Match',
            'resume_years': resume_years,
            'required_range': f"{min_req}-{max_req} years",
            'details': f'{resume_years} years experience fits {min_req}-{max_req} year requirement',
        }
    
    # Slightly below
    if resume_years < min_req:
        gap = min_req - resume_years
        if gap == 1:
            score = 60
            verdict = 'Slightly Below Requirement'
        elif gap == 2:
            score = 40
            verdict = 'Below Requirement'
        else:
            score = max(10, 30 - (gap * 5))
            verdict = 'Significantly Underqualified'
        
        return {
            'score': score,
            'verdict': verdict,
            'resume_years': resume_years,
            'required_range': f"{min_req}-{max_req} years",
            'details': f'{resume_years} years but job needs {min_req}-{max_req} years ({gap} year gap)',
        }
    
    # Overqualified
    if resume_years > max_req:
        over = resume_years - max_req
        if over <= 2:
            score = 85
            verdict = 'Slightly Overqualified'
        else:
            score = 70
            verdict = 'Overqualified'
        
        return {
            'score': score,
            'verdict': verdict,
            'resume_years': resume_years,
            'required_range': f"{min_req}-{max_req} years",
            'details': f'{resume_years} years exceeds {min_req}-{max_req} year range by {over} years',
        }
    
    # Fallback
    return {
        'score': 50,
        'verdict': 'Unable to Assess',
        'resume_years': resume_years,
        'required_range': f"{min_req}-{max_req} years",
        'details': 'Could not determine experience fit',
    }


def estimate_salary_range(
    resume_years: int,
    matched_skills: list,
    education: str,
    career_match: str
) -> dict:
    """
    Estimate salary range based on experience + skills + education.
    
    Base ranges (Mumbai market, monthly):
        Fresher:     ₹15,000 - ₹30,000
        1-2 years:   ₹25,000 - ₹50,000
        3-5 years:   ₹40,000 - ₹80,000
        5+ years:    ₹60,000 - ₹1,50,000
    
    Multipliers:
        In-demand skills (Docker, AWS, React): +10-15%
        Masters/PhD: +10-20%
        AI/ML specialization: +15-25%
    """
    # Base salary by experience (monthly, in INR)
    if resume_years <= 0:
        base_min, base_max = 15000, 30000
        level = 'Fresher'
    elif resume_years <= 2:
        base_min, base_max = 25000, 50000
        level = 'Junior'
    elif resume_years <= 5:
        base_min, base_max = 40000, 80000
        level = 'Mid-Level'
    elif resume_years <= 8:
        base_min, base_max = 60000, 120000
        level = 'Senior'
    else:
        base_min, base_max = 80000, 150000
        level = 'Lead/Principal'
    
    # Skill multipliers
    premium_skills = {
        'docker', 'kubernetes', 'aws', 'azure', 'gcp',
        'react', 'nextjs', 'typescript',
        'machine learning', 'deep learning', 'pytorch', 'tensorflow',
        'system design', 'microservices',
    }
    
    premium_count = len(set(s.lower() for s in matched_skills) & premium_skills)
    skill_multiplier = 1.0 + (premium_count * 0.05)  # +5% per premium skill, max ~25%
    skill_multiplier = min(skill_multiplier, 1.25)
    
    # Education multiplier
    edu_multiplier = 1.0
    if education == 'Masters':
        edu_multiplier = 1.10
    elif education == 'PhD/Doctorate':
        edu_multiplier = 1.20
    
    # Career-specific adjustment
    career_multiplier = 1.0
    high_pay_careers = ['AI/ML Engineer', 'DevOps Engineer', 'Full Stack Developer']
    if career_match in high_pay_careers:
        career_multiplier = 1.10
    
    # Calculate final range
    total_multiplier = skill_multiplier * edu_multiplier * career_multiplier
    final_min = round(base_min * total_multiplier / 1000) * 1000  # Round to nearest 1000
    final_max = round(base_max * total_multiplier / 1000) * 1000
    
    return {
        'level': level,
        'salary_min': final_min,
        'salary_max': final_max,
        'salary_range': f"₹{final_min:,} - ₹{final_max:,}/month",
        'salary_annual': f"₹{final_min * 12 / 100000:.1f}L - ₹{final_max * 12 / 100000:.1f}L/year",
        'multipliers': {
            'skill_boost': f"+{(skill_multiplier - 1) * 100:.0f}%",
            'education_boost': f"+{(edu_multiplier - 1) * 100:.0f}%",
            'career_boost': f"+{(career_multiplier - 1) * 100:.0f}%",
        }
    }