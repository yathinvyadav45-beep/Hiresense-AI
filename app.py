"""
HireSense AI - Smart Recruitment Backend
=========================================
Stack: Flask + pdfplumber + scikit-learn (TF-IDF) + spaCy (NLP)
Run : python app.py
Deps: pip install flask pdfplumber scikit-learn spacy python-docx flask-cors
      python -m spacy download en_core_web_sm
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import docx
import os
import re
import random
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: use spaCy for better NLP if installed
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    print("[INFO] spaCy not found — using TF-IDF only mode.")

app = Flask(__name__)
CORS(app)  # Allow frontend requests

UPLOAD_FOLDER = "resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

# Skills tailored for BCA / MCA graduates
REQUIRED_SKILLS = [
    "python", "java", "html", "css", "javascript",
    "sql", "data structures", "dbms", "react", "angular",
    "problem solving", "communication", "c++", "php",
    "mysql", "mongodb", "git", "linux", "networking",
    "operating systems", "object oriented programming"
]

# Qualifications for BCA / MCA
ELIGIBLE_QUALIFICATIONS = ["bca", "mca", "b.c.a", "m.c.a",
                            "bachelor of computer applications",
                            "master of computer applications"]

# Partner companies with roles and test info
COMPANIES = {
    "TCS": {
        "role": "Software Engineer",
        "package": "3.5–6 LPA",
        "test_type": "TCS NQT (Aptitude + Coding)",
        "interview_rounds": ["Technical Round", "HR Round"]
    },
    "Infosys": {
        "role": "Systems Engineer",
        "package": "3.6–5 LPA",
        "test_type": "Infosys Online Assessment",
        "interview_rounds": ["Technical Round", "HR Round"]
    },
    "Wipro": {
        "role": "Project Engineer",
        "package": "3.5–5.5 LPA",
        "test_type": "WILP Assessment",
        "interview_rounds": ["Technical Round", "Managerial Round", "HR Round"]
    },
    "HCL": {
        "role": "Technical Associate",
        "package": "3–4.5 LPA",
        "test_type": "HCL Aptitude Test",
        "interview_rounds": ["Technical Round", "HR Round"]
    },
    "Cognizant": {
        "role": "Programmer Analyst",
        "package": "4–6 LPA",
        "test_type": "GenC Assessment",
        "interview_rounds": ["Technical Round", "HR Round"]
    },
    "Tech Mahindra": {
        "role": "Software Developer",
        "package": "3.5–5 LPA",
        "test_type": "Online Aptitude Test",
        "interview_rounds": ["Technical Round", "HR Round"]
    }
}

ATS_MINIMUM = 60  # Minimum ATS score to be eligible


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
    return text.lower().strip()


def extract_text_from_docx(path: str) -> str:
    """Extract text from Word document."""
    text = ""
    try:
        doc = docx.Document(path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"[ERROR] DOCX extraction failed: {e}")
    return text.lower().strip()


def extract_resume_text(path: str) -> str:
    """Auto-detect file type and extract text."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in [".doc", ".docx"]:
        return extract_text_from_docx(path)
    else:
        return ""


def check_qualification(text: str) -> dict:
    """Check if candidate has BCA or MCA qualification."""
    for qual in ELIGIBLE_QUALIFICATIONS:
        if qual in text:
            if "mca" in text or "master of computer" in text:
                return {"found": True, "degree": "MCA"}
            return {"found": True, "degree": "BCA"}
    return {"found": False, "degree": None}


def extract_skills(text: str) -> dict:
    """Extract matched and missing skills from resume text."""
    matched = []
    missing = []
    for skill in REQUIRED_SKILLS:
        if skill in text:
            matched.append(skill)
        else:
            missing.append(skill)
    return {"matched": matched, "missing": missing}


def calculate_ats_score(resume_text: str) -> float:
    """
    Calculate ATS score using TF-IDF cosine similarity + skill match bonus.
    Returns score 0–100.
    """
    if not resume_text.strip():
        return 0.0

    job_description = " ".join(REQUIRED_SKILLS)

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([job_description, resume_text])
        similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        base_score = round(similarity * 100, 2)
    except Exception as e:
        print(f"[ERROR] TF-IDF failed: {e}")
        base_score = 0.0

    # Skill match bonus
    matched = [s for s in REQUIRED_SKILLS if s in resume_text]
    skill_bonus = (len(matched) / len(REQUIRED_SKILLS)) * 20  # up to +20 points

    # Qualification bonus
    qual = check_qualification(resume_text)
    qual_bonus = 5 if qual["found"] else 0

    # NLP entity bonus using spaCy
    nlp_bonus = 0
    if SPACY_AVAILABLE:
        try:
            doc = nlp(resume_text[:5000])  # limit for performance
            org_count = sum(1 for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"])
            nlp_bonus = min(org_count * 0.5, 5)  # max +5 from NLP
        except Exception:
            pass

    final_score = min(base_score + skill_bonus + qual_bonus + nlp_bonus, 100)
    return round(final_score, 2)


def generate_schedule(company_name: str) -> dict:
    """Generate test and interview dates based on current date."""
    today = datetime.today()
    test_date = today + timedelta(days=7)
    interview_date = today + timedelta(days=21)

    fmt = "%A, %d %B %Y"
    return {
        "test_date": test_date.strftime(fmt),
        "test_time": "10:00 AM IST",
        "test_type": COMPANIES.get(company_name, {}).get("test_type", "Online Assessment"),
        "interview_date": interview_date.strftime(fmt),
        "interview_time": "2:00 PM IST",
        "interview_rounds": COMPANIES.get(company_name, {}).get("interview_rounds", ["Technical", "HR"]),
        "mode": "Online"
    }


def generate_improvement_tips(missing_skills: list) -> list:
    """Generate personalised improvement tips."""
    tips = []
    if missing_skills:
        tips.append(f"Add missing skills to your resume: {', '.join(missing_skills[:5])}")
    tips += [
        "Use clear section headings: Skills, Projects, Education, Experience",
        "Quantify achievements e.g. 'Built a web app with 500+ active users'",
        "Avoid tables, columns, or images — ATS parsers struggle with these",
        "Include your BCA/MCA project details with tech stacks used",
        "Use keywords from the job description naturally throughout your resume"
    ]
    return tips


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "HireSense AI Backend Running",
        "version": "2.0",
        "endpoints": ["/upload", "/companies", "/apply", "/health"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "spacy": SPACY_AVAILABLE})


@app.route("/companies", methods=["GET"])
def get_companies():
    """Return list of available companies."""
    company_list = []
    for name, info in COMPANIES.items():
        company_list.append({
            "name": name,
            "role": info["role"],
            "package": info["package"],
            "test_type": info["test_type"]
        })
    return jsonify({"companies": company_list})


@app.route("/upload", methods=["POST"])
def upload():
    """
    Main ATS scoring endpoint.
    Accepts: multipart/form-data with 'resume' file
    Returns: JSON with score, skills, eligibility, tips
    """
    if "resume" not in request.files:
        return jsonify({"error": "No resume file provided"}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Validate file type
    allowed_ext = {".pdf", ".doc", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": "Invalid file type. Upload PDF or DOCX."}), 400

    # Save file
    safe_name = re.sub(r"[^\w\-_.]", "_", file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(save_path)

    # Extract text
    resume_text = extract_resume_text(save_path)
    if not resume_text.strip():
        return jsonify({"error": "Could not extract text from resume. Try a text-based PDF."}), 422

    # Analysis
    score = calculate_ats_score(resume_text)
    skills = extract_skills(resume_text)
    qualification = check_qualification(resume_text)
    eligible = score >= ATS_MINIMUM

    response = {
        "ats_score": score,
        "minimum_score": ATS_MINIMUM,
        "eligible": eligible,
        "qualification": qualification,
        "skills": {
            "matched": skills["matched"],
            "missing": skills["missing"],
            "match_count": len(skills["matched"]),
            "total_skills": len(REQUIRED_SKILLS)
        },
        "message": (
            f"Congratulations! Your ATS score is {score}%. You are eligible to apply."
            if eligible else
            f"Your ATS score is {score}%, which is below the minimum of {ATS_MINIMUM}%. "
            f"Please improve your resume to qualify."
        ),
        "improvement_tips": [] if eligible else generate_improvement_tips(skills["missing"])
    }

    if eligible:
        response["available_companies"] = list(COMPANIES.keys())

    return jsonify(response)


@app.route("/apply", methods=["POST"])
def apply():
    """
    Apply to a company and get test/interview schedule.
    Requires: JSON body with { "company": "TCS", "ats_score": 74 }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    company_name = data.get("company", "").strip()
    ats_score = data.get("ats_score", 0)

    if ats_score < ATS_MINIMUM:
        return jsonify({
            "error": f"ATS score {ats_score}% is below minimum {ATS_MINIMUM}%. Not eligible to apply.",
            "eligible": False
        }), 403

    if company_name not in COMPANIES:
        return jsonify({"error": f"Company '{company_name}' not found. Valid: {list(COMPANIES.keys())}"}), 404

    company_info = COMPANIES[company_name]
    schedule = generate_schedule(company_name)

    return jsonify({
        "success": True,
        "company": company_name,
        "role": company_info["role"],
        "package": company_info["package"],
        "ats_score": ats_score,
        "schedule": schedule,
        "next_steps": [
            f"Test Date: {schedule['test_date']} at {schedule['test_time']}",
            f"Test Type: {schedule['test_type']}",
            f"Interview Date (if test passed): {schedule['interview_date']} at {schedule['interview_time']}",
            f"Interview Rounds: {', '.join(schedule['interview_rounds'])}",
            "Mode: Online"
        ],
        "message": (
            f"Successfully applied to {company_name}! "
            f"Your aptitude test is on {schedule['test_date']}. "
            f"If you pass, HR interview will be on {schedule['interview_date']}."
        )
    })


@app.route("/test/result", methods=["POST"])
def test_result():
    """
    Submit aptitude test result and decide interview eligibility.
    Requires: JSON { "company": "TCS", "score": 75, "ats_score": 74 }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    test_score = data.get("score", 0)
    company_name = data.get("company", "")
    ats_score = data.get("ats_score", 0)
    passed = test_score >= 60

    if passed:
        schedule = generate_schedule(company_name)
        return jsonify({
            "passed": True,
            "test_score": test_score,
            "message": f"Congratulations! Test score {test_score}%. HR Interview scheduled.",
            "interview_date": schedule["interview_date"],
            "interview_time": schedule["interview_time"],
            "interview_rounds": schedule["interview_rounds"],
            "mode": "Online"
        })
    else:
        return jsonify({
            "passed": False,
            "test_score": test_score,
            "message": (
                f"Test score {test_score}% is below 60%. "
                f"You are not eligible for the interview this time. "
                f"Please prepare and try again in the next hiring cycle."
            ),
            "tips": [
                "Revise Data Structures and Algorithms",
                "Practice aptitude questions on IndiaBix, PrepInsta",
                "Review SQL queries and DBMS concepts",
                "Improve verbal reasoning and quantitative aptitude"
            ]
        })


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  HireSense AI Backend v2.0")
    print("  Target: BCA / MCA Graduates")
    print(f"  ATS Minimum Score: {ATS_MINIMUM}%")
    print(f"  spaCy NLP: {'Enabled' if SPACY_AVAILABLE else 'Disabled (TF-IDF only)'}")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
