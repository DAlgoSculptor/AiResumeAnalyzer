"""
Enhanced AI Resume Analyzer with Advanced NLP and Machine Learning Capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
import io
from pathlib import Path

# Core ML and NLP imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
except ImportError:
    nltk = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    import spacy
    # Load spacy model with error handling
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        st.warning("⚠️ SpaCy model not found. Some advanced features may be limited.")
except ImportError:
    spacy = None
    nlp = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    # Initialize sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest")
except ImportError:
    sentiment_analyzer = None

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Document processing imports
try:
    import PyPDF2
    import pdfplumber
except ImportError:
    PyPDF2 = None
    pdfplumber = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

class DatabaseManager:
    """Manages SQLite database operations for resume analysis history"""
    
    def __init__(self, db_path: str = "resume_analyzer.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resume_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE,
                filename TEXT,
                analysis_date TIMESTAMP,
                overall_score REAL,
                experience_score REAL,
                education_score REAL,
                skills_score REAL,
                ats_score REAL,
                sentiment_score REAL,
                analysis_data TEXT,
                recommendations TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS skill_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_name TEXT,
                category TEXT,
                frequency INTEGER,
                last_seen TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, file_hash: str, filename: str, analysis_results: Dict):
        """Save analysis results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO resume_analyses 
                (file_hash, filename, analysis_date, overall_score, experience_score, 
                 education_score, skills_score, ats_score, sentiment_score, 
                 analysis_data, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_hash, filename, datetime.now(),
                analysis_results.get('overall_score', 0),
                analysis_results.get('experience_score', 0),
                analysis_results.get('education_score', 0),
                analysis_results.get('skills_score', 0),
                analysis_results.get('ats_score', 0),
                analysis_results.get('sentiment_score', 0),
                json.dumps(analysis_results),
                json.dumps(analysis_results.get('recommendations', []))
            ))
            conn.commit()
        except Exception as e:
            st.error(f"Database error: {e}")
        finally:
            conn.close()
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve analysis history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM resume_analyses 
            ORDER BY analysis_date DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        columns = ['id', 'file_hash', 'filename', 'analysis_date', 'overall_score',
                  'experience_score', 'education_score', 'skills_score', 'ats_score',
                  'sentiment_score', 'analysis_data', 'recommendations']
        
        return [dict(zip(columns, row)) for row in results]

class AdvancedResumeAnalyzer:
    """Enhanced Resume Analyzer with advanced AI capabilities"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.lemmatizer = WordNetLemmatizer() if nltk else None
        self.stop_words = set(stopwords.words('english')) if nltk else set()
        
        # Enhanced skill categories with weights
        self.skill_categories = {
            'programming': {
                'skills': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'kotlin', 'swift', 'scala', 'dart', 'r'],
                'weight': 0.25
            },
            'web_development': {
                'skills': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'nextjs', 'nuxt', 'svelte'],
                'weight': 0.20
            },
            'data_science': {
                'skills': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib', 'seaborn', 'jupyter', 'keras', 'spark', 'hadoop'],
                'weight': 0.20
            },
            'cloud_devops': {
                'skills': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'heroku', 'vercel', 'gitlab-ci', 'github-actions'],
                'weight': 0.15
            },
            'databases': {
                'skills': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'oracle', 'cassandra', 'dynamodb', 'neo4j'],
                'weight': 0.10
            },
            'soft_skills': {
                'skills': ['leadership', 'communication', 'teamwork', 'problem-solving', 'analytical', 'creative', 'management', 'agile', 'scrum'],
                'weight': 0.10
            }
        }
        
        # ATS-friendly keywords
        self.ats_keywords = [
            'experience', 'skills', 'education', 'certification', 'project', 'achievement',
            'responsibility', 'accomplishment', 'result', 'impact', 'improvement', 'optimization'
        ]
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    def extract_text_advanced(self, uploaded_file) -> str:
        """Enhanced text extraction with multiple fallback methods"""
        text = ""
        file_content = uploaded_file.getvalue()
        
        try:
            if uploaded_file.type == "application/pdf":
                # Try pdfplumber first (better for complex layouts)
                if pdfplumber:
                    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                
                # Fallback to PyPDF2
                if not text and PyPDF2:
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                        
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                if Document:
                    doc = Document(io.BytesIO(file_content))
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                        
            elif uploaded_file.type == "text/plain":
                text = file_content.decode("utf-8")
                
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            
        return text.strip()
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing"""
        if not text:
            return ""
            
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if nltk and self.lemmatizer:
            # Tokenize and lemmatize
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            text = ' '.join(tokens)
            
        return text

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of resume text"""
        sentiment_data = {
            'score': 0.5,
            'label': 'neutral',
            'confidence': 0.0,
            'details': {}
        }

        try:
            if sentiment_analyzer:
                # Use transformer-based sentiment analysis
                result = sentiment_analyzer(text[:512])  # Limit text length
                sentiment_data['label'] = result[0]['label'].lower()
                sentiment_data['confidence'] = result[0]['score']
                sentiment_data['score'] = result[0]['score'] if result[0]['label'] == 'POSITIVE' else 1 - result[0]['score']

            elif TextBlob:
                # Fallback to TextBlob
                blob = TextBlob(text)
                sentiment_data['score'] = (blob.sentiment.polarity + 1) / 2  # Convert to 0-1 scale
                sentiment_data['label'] = 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
                sentiment_data['confidence'] = abs(blob.sentiment.polarity)

        except Exception as e:
            st.warning(f"Sentiment analysis error: {e}")

        return sentiment_data

    def extract_entities(self, text: str) -> Dict:
        """Extract named entities using spaCy"""
        entities = {
            'organizations': [],
            'locations': [],
            'persons': [],
            'dates': [],
            'skills': [],
            'technologies': []
        }

        try:
            if nlp:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        entities['organizations'].append(ent.text)
                    elif ent.label_ in ["GPE", "LOC"]:
                        entities['locations'].append(ent.text)
                    elif ent.label_ == "PERSON":
                        entities['persons'].append(ent.text)
                    elif ent.label_ == "DATE":
                        entities['dates'].append(ent.text)

        except Exception as e:
            st.warning(f"Entity extraction error: {e}")

        return entities

    def calculate_ats_score(self, text: str) -> float:
        """Calculate ATS (Applicant Tracking System) compatibility score"""
        score = 0
        total_checks = 10

        # Check for standard sections
        sections = ['experience', 'education', 'skills', 'summary', 'objective']
        for section in sections:
            if section in text.lower():
                score += 1

        # Check for quantifiable achievements
        numbers_pattern = r'\d+(?:\.\d+)?(?:%|percent|k|thousand|million|billion)?'
        if re.search(numbers_pattern, text):
            score += 1

        # Check for action verbs
        action_verbs = ['managed', 'led', 'developed', 'created', 'implemented', 'improved', 'increased', 'decreased']
        for verb in action_verbs:
            if verb in text.lower():
                score += 0.5

        # Check for keywords density
        keyword_count = sum(1 for keyword in self.ats_keywords if keyword in text.lower())
        if keyword_count >= 5:
            score += 1

        # Check text length (not too short, not too long)
        word_count = len(text.split())
        if 300 <= word_count <= 800:
            score += 1

        return min(score / total_checks * 100, 100)

    def extract_skills_advanced(self, text: str) -> Dict:
        """Advanced skill extraction with categorization and confidence scoring"""
        text_lower = text.lower()
        extracted_skills = {}

        for category, data in self.skill_categories.items():
            category_skills = []
            for skill in data['skills']:
                # Use regex for better matching
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                matches = re.findall(pattern, text_lower)
                if matches:
                    # Calculate confidence based on frequency and context
                    frequency = len(matches)
                    confidence = min(frequency * 0.2 + 0.6, 1.0)
                    category_skills.append({
                        'name': skill,
                        'frequency': frequency,
                        'confidence': confidence
                    })

            if category_skills:
                extracted_skills[category] = {
                    'skills': category_skills,
                    'count': len(category_skills),
                    'weight': data['weight']
                }

        return extracted_skills

    def calculate_experience_score_advanced(self, text: str) -> Tuple[float, Dict]:
        """Advanced experience scoring with detailed analysis"""
        experience_data = {
            'total_years': 0,
            'positions': [],
            'companies': [],
            'achievements': [],
            'score': 0
        }

        # Extract years of experience
        year_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'over\s*(\d+)\s*years?',
            r'more\s*than\s*(\d+)\s*years?'
        ]

        years = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text.lower())
            years.extend([int(match) for match in matches])

        if years:
            experience_data['total_years'] = max(years)

        # Extract job positions and companies using NER
        entities = self.extract_entities(text)
        experience_data['companies'] = list(set(entities['organizations']))

        # Calculate score based on multiple factors
        base_score = min(experience_data['total_years'] * 10, 70)  # Max 70 for experience years

        # Bonus for multiple companies
        company_bonus = min(len(experience_data['companies']) * 5, 20)

        # Bonus for quantifiable achievements
        achievement_patterns = [
            r'increased.*?(\d+)%',
            r'improved.*?(\d+)%',
            r'reduced.*?(\d+)%',
            r'managed.*?(\d+)',
            r'led.*?team.*?(\d+)'
        ]

        achievements = 0
        for pattern in achievement_patterns:
            if re.search(pattern, text.lower()):
                achievements += 1

        achievement_bonus = min(achievements * 5, 10)

        experience_data['score'] = min(base_score + company_bonus + achievement_bonus, 100)

        return experience_data['score'], experience_data

    def calculate_education_score_advanced(self, text: str) -> Tuple[float, Dict]:
        """Advanced education scoring with degree recognition"""
        education_data = {
            'degrees': [],
            'institutions': [],
            'certifications': [],
            'score': 0
        }

        # Degree patterns with scoring
        degree_patterns = {
            r'\b(?:phd|ph\.d|doctorate|doctoral)\b': 40,
            r'\b(?:master|masters|msc|ms|ma|mba)\b': 30,
            r'\b(?:bachelor|bachelors|bsc|bs|ba)\b': 20,
            r'\b(?:associate|diploma)\b': 10,
            r'\b(?:certificate|certification)\b': 5
        }

        score = 0
        for pattern, points in degree_patterns.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                education_data['degrees'].extend(matches)
                score += points

        # Extract institutions
        entities = self.extract_entities(text)
        education_data['institutions'] = list(set(entities['organizations']))

        # Bonus for prestigious institutions (simplified check)
        prestigious_keywords = ['university', 'institute', 'college', 'school']
        institution_bonus = 0
        for keyword in prestigious_keywords:
            if keyword in text.lower():
                institution_bonus += 5

        education_data['score'] = min(score + min(institution_bonus, 20), 100)

        return education_data['score'], education_data

    def calculate_job_match_advanced(self, resume_text: str, job_description: str) -> Tuple[float, Dict]:
        """Advanced job matching with detailed analysis"""
        match_data = {
            'score': 50,
            'matched_skills': [],
            'missing_skills': [],
            'similarity_score': 0,
            'keyword_overlap': 0
        }

        if not job_description.strip():
            return match_data['score'], match_data

        try:
            # Preprocess texts
            resume_processed = self.preprocess_text(resume_text)
            job_processed = self.preprocess_text(job_description)

            if not resume_processed or not job_processed:
                return match_data['score'], match_data

            # Calculate TF-IDF similarity
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            match_data['similarity_score'] = similarity[0][0] * 100

            # Extract skills from both texts
            resume_skills = self.extract_skills_advanced(resume_text)
            job_skills = self.extract_skills_advanced(job_description)

            # Find matched and missing skills
            all_resume_skills = set()
            for category_data in resume_skills.values():
                for skill_data in category_data['skills']:
                    all_resume_skills.add(skill_data['name'].lower())

            all_job_skills = set()
            for category_data in job_skills.values():
                for skill_data in category_data['skills']:
                    all_job_skills.add(skill_data['name'].lower())

            match_data['matched_skills'] = list(all_resume_skills.intersection(all_job_skills))
            match_data['missing_skills'] = list(all_job_skills - all_resume_skills)

            # Calculate keyword overlap
            resume_words = set(resume_processed.split())
            job_words = set(job_processed.split())
            overlap = len(resume_words.intersection(job_words))
            total_job_words = len(job_words)
            match_data['keyword_overlap'] = (overlap / total_job_words * 100) if total_job_words > 0 else 0

            # Calculate final score
            skill_match_score = (len(match_data['matched_skills']) / len(all_job_skills) * 100) if all_job_skills else 50
            final_score = (match_data['similarity_score'] * 0.4 + skill_match_score * 0.4 + match_data['keyword_overlap'] * 0.2)
            match_data['score'] = min(final_score, 100)

        except Exception as e:
            st.warning(f"Job matching error: {e}")

        return match_data['score'], match_data

    def generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate AI-powered recommendations based on analysis"""
        recommendations = []

        # Experience recommendations
        if analysis_results['experience_score'] < 60:
            recommendations.append("💼 Consider adding more quantifiable achievements and specific metrics to demonstrate your impact.")
            recommendations.append("📈 Include more details about your responsibilities and accomplishments in previous roles.")

        # Education recommendations
        if analysis_results['education_score'] < 50:
            recommendations.append("🎓 Consider adding relevant certifications or online courses to strengthen your educational background.")
            recommendations.append("📚 Highlight any professional development or training programs you've completed.")

        # Skills recommendations
        skills_data = analysis_results.get('skills_detailed', {})
        total_skills = sum(data['count'] for data in skills_data.values())
        if total_skills < 10:
            recommendations.append("🛠️ Add more technical skills relevant to your field to improve your profile.")
            recommendations.append("💻 Consider learning trending technologies in your industry.")

        # ATS recommendations
        if analysis_results['ats_score'] < 70:
            recommendations.append("🤖 Improve ATS compatibility by using standard section headers (Experience, Education, Skills).")
            recommendations.append("📝 Use more action verbs and industry-specific keywords.")
            recommendations.append("📊 Add quantifiable achievements with specific numbers and percentages.")

        # Sentiment recommendations
        if analysis_results['sentiment_score'] < 60:
            recommendations.append("😊 Use more positive and confident language to improve the overall tone.")
            recommendations.append("✨ Highlight your achievements and contributions more prominently.")

        # Job match recommendations
        if 'job_match_detailed' in analysis_results:
            missing_skills = analysis_results['job_match_detailed'].get('missing_skills', [])
            if missing_skills:
                recommendations.append(f"🎯 Consider developing these skills mentioned in the job description: {', '.join(missing_skills[:5])}")

        # General recommendations
        word_count = analysis_results.get('resume_length', 0)
        if word_count < 200:
            recommendations.append("📄 Your resume appears to be quite short. Consider adding more details about your experience and achievements.")
        elif word_count > 1000:
            recommendations.append("✂️ Your resume might be too long. Consider condensing information to keep it concise and focused.")

        return recommendations[:8]  # Limit to top 8 recommendations

    def analyze_resume_comprehensive(self, resume_text: str, job_description: str = "") -> Dict:
        """Comprehensive resume analysis with all advanced features"""

        # Basic analysis
        sentiment_data = self.analyze_sentiment(resume_text)
        entities = self.extract_entities(resume_text)
        ats_score = self.calculate_ats_score(resume_text)

        # Advanced skill analysis
        skills_detailed = self.extract_skills_advanced(resume_text)

        # Calculate skills score
        skills_score = 0
        if skills_detailed:
            total_weight = sum(data['weight'] for data in skills_detailed.values())
            weighted_score = sum(data['count'] * data['weight'] * 10 for data in skills_detailed.values())
            skills_score = min(weighted_score / total_weight if total_weight > 0 else 0, 100)

        # Experience analysis
        experience_score, experience_detailed = self.calculate_experience_score_advanced(resume_text)

        # Education analysis
        education_score, education_detailed = self.calculate_education_score_advanced(resume_text)

        # Job matching analysis
        job_match_score, job_match_detailed = self.calculate_job_match_advanced(resume_text, job_description)

        # Calculate overall score with weights
        weights = {
            'experience': 0.25,
            'education': 0.15,
            'skills': 0.20,
            'ats': 0.15,
            'sentiment': 0.10,
            'job_match': 0.15 if job_description.strip() else 0
        }

        # Adjust weights if no job description
        if not job_description.strip():
            weights['experience'] = 0.30
            weights['education'] = 0.20
            weights['skills'] = 0.25
            weights['ats'] = 0.15
            weights['sentiment'] = 0.10

        overall_score = (
            experience_score * weights['experience'] +
            education_score * weights['education'] +
            skills_score * weights['skills'] +
            ats_score * weights['ats'] +
            sentiment_data['score'] * 100 * weights['sentiment'] +
            job_match_score * weights['job_match']
        )

        # Compile results
        results = {
            'overall_score': overall_score,
            'experience_score': experience_score,
            'education_score': education_score,
            'skills_score': skills_score,
            'ats_score': ats_score,
            'sentiment_score': sentiment_data['score'] * 100,
            'job_match_score': job_match_score,
            'resume_length': len(resume_text.split()),
            'skills_detailed': skills_detailed,
            'experience_detailed': experience_detailed,
            'education_detailed': education_detailed,
            'job_match_detailed': job_match_detailed,
            'sentiment_detailed': sentiment_data,
            'entities': entities,
            'weights_used': weights
        }

        # Generate recommendations
        results['recommendations'] = self.generate_recommendations(results)

        return results
