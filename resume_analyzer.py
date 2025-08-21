import streamlit as st
import io
import plotly.express as px
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from collections import Counter
import time

# Conditional imports for file processing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None

from utils import download_nltk_data, load_css, SKILL_CATEGORIES, EXPERIENCE_KEYWORDS, EDUCATION_KEYWORDS
from ui_elements import create_skills_chart, create_gauge_chart, create_radar_chart, display_metrics, display_skills, display_recommendations, display_features, display_sidebar, display_hero_section, display_about_section, display_services_section, display_contact_section, display_navbar

# 🚀 MUST BE FIRST STREAMLIT COMMAND
# ✅ MUST BE FIRST Streamlit command
st.set_page_config(
page_title="AI Resume Analyzer",
page_icon="🚀",
layout="wide",
initial_sidebar_state="expanded"
)

# Initialize NLTK data
download_nltk_data()

class ResumeAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Predefined skill categories and job requirements
        self.skill_categories = SKILL_CATEGORIES
        
        self.experience_keywords = EXPERIENCE_KEYWORDS
        
        self.education_keywords = EDUCATION_KEYWORDS
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        if PyPDF2 is None:
            st.error("PyPDF2 is not available. Please install it to process PDF files.")
            return ""
        
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file):
        """Extract text from DOCX file"""
        if Document is None:
            st.error("python-docx is not available. Please install it to process DOCX files.")
            return ""
        
        try:
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def extract_skills(self, text):
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skill_categories.items():
            found_skills[category] = []
            for skill in skills:
                if skill in text_lower:
                    found_skills[category].append(skill.title())
        
        return found_skills
    
    def calculate_experience_score(self, text):
        """Calculate experience score based on keywords"""
        text_lower = text.lower()
        experience_count = sum(1 for keyword in self.experience_keywords if keyword in text_lower)
        
        # Look for years of experience
        years_pattern = r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)'
        years_matches = re.findall(years_pattern, text_lower)
        
        total_years = sum(int(year) for year in years_matches)
        
        # Calculate score (normalize to 0-100)
        experience_score = min(100, (experience_count * 5) + (total_years * 8))
        
        return experience_score, total_years
    
    def calculate_education_score(self, text):
        """Calculate education score"""
        text_lower = text.lower()
        education_count = sum(1 for keyword in self.education_keywords if keyword in text_lower)
        
        # Bonus points for advanced degrees
        if 'phd' in text_lower or 'doctorate' in text_lower:
            education_count += 4
        elif 'master' in text_lower or 'mba' in text_lower:
            education_count += 3
        elif 'bachelor' in text_lower:
            education_count += 2
        
        education_score = min(100, education_count * 8)
        return education_score
    
    def calculate_job_match(self, resume_text, job_description):
        """Calculate job match score using cosine similarity"""
        if not job_description.strip():
            return 50  # Default score when no job description
        
        try:
            resume_processed = self.preprocess_text(resume_text)
            job_processed = self.preprocess_text(job_description)
            
            if not resume_processed or not job_processed:
                return 50
            
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return similarity[0][0] * 100
            
        except Exception as e:
            st.warning(f"Could not calculate job match: {str(e)}")
            return 50
    
    def analyze_resume(self, resume_text, job_description=""):
        """Main analysis function"""
        # Extract skills
        skills = self.extract_skills(resume_text)
        
        # Calculate scores
        experience_score, total_years = self.calculate_experience_score(resume_text)
        education_score = self.calculate_education_score(resume_text)
        job_match_score = self.calculate_job_match(resume_text, job_description)
        
        # Calculate overall score with weights
        if job_description.strip():
            overall_score = (experience_score * 0.35 + education_score * 0.25 + job_match_score * 0.4)
        else:
            overall_score = (experience_score * 0.5 + education_score * 0.5)
        
        return {
            'skills': skills,
            'experience_score': experience_score,
            'education_score': education_score,
            'job_match_score': job_match_score,
            'overall_score': overall_score,
            'total_years_experience': total_years,
            'resume_length': len(resume_text.split())
        }

def main():
    """Main function to run the Streamlit app"""
    load_css()  # Load custom CSS for styling
    display_navbar()
    st.markdown('<div id="hero-section"></div>', unsafe_allow_html=True)
    display_hero_section()  # Display hero section instead of header
    st.markdown('<div id="about-section"></div>', unsafe_allow_html=True)
    display_about_section()
    st.markdown('<div id="services-section"></div>', unsafe_allow_html=True)
    display_services_section()
    
    # Initialize analyzer ✅ FIXED
    analyzer = ResumeAnalyzer()  # Initialize analyzer
    
    # Sidebar
    uploaded_file, job_description, analyze_button = display_sidebar()
    
    if analyze_button and uploaded_file:
        with st.spinner("Analyzing resume... 🚀"):
            # Extract text depending on file type
            resume_text = ""
            if uploaded_file.type == "application/pdf":
                resume_text = analyzer.extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = analyzer.extract_text_from_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                resume_text = stringio.read()
            
            if resume_text:
                results = analyzer.analyze_resume(resume_text, job_description)
                
                # Display metrics
                display_metrics(results)
                
                # Charts
                st.markdown('<div class="section-header">📊 Resume Analytics</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_gauge_chart(results['overall_score'], "Overall Score"), use_container_width=True)
                with col2:
                    st.plotly_chart(create_radar_chart(results), use_container_width=True)
                
                # Skills section
                display_skills(results['skills'])
                skills_chart = create_skills_chart(results['skills'])
                if skills_chart:
                    st.plotly_chart(skills_chart, use_container_width=True)
                
                # Recommendations
                display_recommendations(results)
                
                # Features
                st.markdown('<div id="features-section"></div>', unsafe_allow_html=True)
                display_features()
            else:
                st.error("⚠️ Could not extract text from the uploaded file. Please try another format.")
    elif analyze_button and not uploaded_file:
        st.warning("⚠️ Please upload a resume file before analysis.")
    
    st.markdown('<div id="contact-section"></div>', unsafe_allow_html=True)
    display_contact_section() # Display contact section at the end

if __name__ == "__main__":
    main()
