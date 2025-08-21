import streamlit as st
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
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import PyPDF2
from docx import Document
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Custom CSS for modern UI
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
    }
    
    .main-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(31, 38, 135, 0.5);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Upload Area */
    .upload-area {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Progress Bar */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        height: 8px;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Skills Tags */
    .skill-tag {
        display: inline-block;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        text-align: center;
        position: relative;
    }
    
    .section-header:after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 3px;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border-radius: 2px;
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4ecdc4;
        box-shadow: 0 5px 20px rgba(31, 38, 135, 0.3);
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(31, 38, 135, 0.5);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Animated Background */
    .floating-shapes {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .shape {
        position: absolute;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Text Areas and Inputs */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-top: 3px solid #4ecdc4;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    
    <div class="floating-shapes">
        <div class="shape" style="width: 100px; height: 100px; top: 10%; left: 10%; animation-delay: 0s;"></div>
        <div class="shape" style="width: 150px; height: 150px; top: 20%; right: 10%; animation-delay: 1s;"></div>
        <div class="shape" style="width: 80px; height: 80px; bottom: 20%; left: 15%; animation-delay: 2s;"></div>
        <div class="shape" style="width: 120px; height: 120px; bottom: 10%; right: 20%; animation-delay: 3s;"></div>
    </div>
    """, unsafe_allow_html=True)

class ResumeAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Predefined skill categories and job requirements
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'kotlin', 'swift'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib', 'seaborn', 'jupyter'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'oracle'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform'],
            'analytics': ['excel', 'tableau', 'powerbi', 'looker', 'google analytics', 'sql', 'r'],
            'project_management': ['agile', 'scrum', 'kanban', 'jira', 'trello', 'asana', 'confluence'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem-solving', 'analytical', 'creative']
        }
        
        self.experience_keywords = ['experience', 'worked', 'developed', 'managed', 'led', 'created', 'implemented', 
                                  'designed', 'built', 'maintained', 'optimized', 'improved', 'collaborated']
        
        self.education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'certification', 
                                 'course', 'training', 'certified']
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
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
                    found_skills[category].append(skill)
        
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
        experience_score = min(100, (experience_count * 5) + (total_years * 10))
        
        return experience_score, total_years
    
    def calculate_education_score(self, text):
        """Calculate education score"""
        text_lower = text.lower()
        education_count = sum(1 for keyword in self.education_keywords if keyword in text_lower)
        
        # Bonus points for advanced degrees
        if 'phd' in text_lower or 'doctorate' in text_lower:
            education_count += 3
        elif 'master' in text_lower or 'mba' in text_lower:
            education_count += 2
        elif 'bachelor' in text_lower:
            education_count += 1
        
        education_score = min(100, education_count * 10)
        return education_score
    
    def calculate_job_match(self, resume_text, job_description):
        """Calculate job match score using cosine similarity"""
        if not job_description:
            return 0
        
        # Preprocess texts
        resume_processed = self.preprocess_text(resume_text)
        job_processed = self.preprocess_text(job_description)
        
        # Vectorize
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
        
        # Calculate similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        
        return similarity[0][0] * 100
    
    def analyze_resume(self, resume_text, job_description=""):
        """Main analysis function"""
        # Extract skills
        skills = self.extract_skills(resume_text)
        
        # Calculate scores
        experience_score, total_years = self.calculate_experience_score(resume_text)
        education_score = self.calculate_education_score(resume_text)
        job_match_score = self.calculate_job_match(resume_text, job_description)
        
        # Calculate overall score
        overall_score = (experience_score * 0.4 + education_score * 0.3 + job_match_score * 0.3)
        
        return {
            'skills': skills,
            'experience_score': experience_score,
            'education_score': education_score,
            'job_match_score': job_match_score,
            'overall_score': overall_score,
            'total_years_experience': total_years,
            'resume_length': len(resume_text.split())
        }

def create_modern_skills_chart(skills_data):
    """Create modern skills visualization with glassmorphism effect"""
    skill_counts = {}
    for category, skills in skills_data.items():
        if skills:
            skill_counts[category.replace('_', ' ').title()] = len(skills)
    
    if skill_counts:
        fig = px.bar(
            x=list(skill_counts.keys()),
            y=list(skill_counts.values()),
            title="Skills Distribution by Category",
            labels={'x': '', 'y': 'Number of Skills'},
            color=list(skill_counts.values()),
            color_continuous_scale='plasma'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title=dict(font=dict(size=20, color='white'), x=0.5),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)', zeroline=False)
        )
        
        fig.update_traces(
            marker=dict(line=dict(color='rgba(255,255,255,0.3)', width=1)),
            hovertemplate='<b>%{x}</b><br>Skills: %{y}<extra></extra>'
        )
        
        return fig
    return None

def create_modern_gauge(score, title, color_scheme="plasma"):
    """Create modern gauge chart with glassmorphism effect"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': 'white'}},
        number={'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar': {'color': "#4ecdc4", 'thickness': 0.7},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.3)",
            'steps': [
                {'range': [0, 25], 'color': "rgba(255,255,255,0.1)"},
                {'range': [25, 50], 'color': "rgba(255,255,255,0.15)"},
                {'range': [50, 75], 'color': "rgba(255,255,255,0.2)"},
                {'range': [75, 100], 'color': "rgba(255,255,255,0.25)"}
            ],
            'threshold': {
                'line': {'color': "#ff6b6b", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_radar_chart(results):
    """Create a radar chart for comprehensive analysis"""
    categories = ['Experience', 'Education', 'Job Match', 'Skills Diversity', 'Resume Quality']
    values = [
        results['experience_score'],
        results['education_score'],
        results['job_match_score'],
        min(100, sum(len(skills) for skills in results['skills'].values()) * 10),
        min(100, results['resume_length'] / 5)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Resume',
        line=dict(color='#4ecdc4', width=3),
        fillcolor='rgba(78, 205, 196, 0.3)',
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.3)',
                tickcolor='white',
                tickfont=dict(color='white')
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.3)',
                tickcolor='white',
                tickfont=dict(color='white', size=12)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        title=dict(text="Resume Analysis Radar", font=dict(size=20, color='white'), x=0.5),
        height=400
    )
    
    return fig

def display_modern_header():
    """Display modern header with glassmorphism effect"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🚀 AI Resume Analyzer</div>
        <div class="main-subtitle">
            Transform your career with AI-powered resume insights • Advanced NLP • Real-time Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_metric_cards(results):
    """Display modern metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Overall Score", results['overall_score'], "🎯"),
        ("Experience", results['experience_score'], "💼"),
        ("Education", results['education_score'], "🎓"),
        ("Job Match", results['job_match_score'], "🔍")
    ]
    
    for i, (label, value, icon) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div class="metric-value">{value:.1f}%</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def display_skills_tags(skills_data):
    """Display skills as modern tags"""
    st.markdown('<div class="section-header">🛠️ Skills Portfolio</div>', unsafe_allow_html=True)
    
    for category, skills in skills_data.items():
        if skills:
            st.markdown(f"""
            <h4 style="color: white; margin: 1.5rem 0 1rem 0; font-weight: 600;">
                {category.replace('_', ' ').title()}
            </h4>
            """, unsafe_allow_html=True)
            
            tags_html = ""
            for skill in skills:
                tags_html += f'<span class="skill-tag">{skill}</span>'
            
            st.markdown(tags_html, unsafe_allow_html=True)

def display_recommendations(results):
    """Display modern recommendation cards"""
    st.markdown('<div class="section-header">💡 AI Recommendations</div>', unsafe_allow_html=True)
    
    recommendations = []
    
    if results['experience_score'] < 60:
        recommendations.append({
            'icon': '📈',
            'title': 'Boost Experience Score',
            'desc': 'Add more specific achievements and quantifiable results to demonstrate your impact.'
        })
    
    if results['education_score'] < 50:
        recommendations.append({
            'icon': '🎓',
            'title': 'Enhance Education Section',
            'desc': 'Include certifications, courses, and relevant training to strengthen your qualifications.'
        })
    
    total_skills = sum(len(skills) for skills in results['skills'].values())
    if total_skills < 10:
        recommendations.append({
            'icon': '🛠️',
            'title': 'Expand Skill Set',
            'desc': 'Add more technical and soft skills relevant to your target role.'
        })
    
    if results['resume_length'] < 300:
        recommendations.append({
            'icon': '📝',
            'title': 'Expand Content',
            'desc': 'Provide more detailed descriptions of your projects and accomplishments.'
        })
    
    if not recommendations:
        st.markdown("""
        <div class="recommendation-card" style="background: linear-gradient(45deg, rgba(76, 175, 80, 0.3), rgba(139, 195, 74, 0.3));">
            <h3 style="color: #4CAF50; margin-bottom: 1rem;">🌟 Excellent Resume!</h3>
            <p style="color: white;">Your resume demonstrates strong alignment with industry best practices. Keep up the great work!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for rec in recommendations:
            st.markdown(f"""
            <div class="recommendation-card">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">{rec['icon']}</span>
                    <h4 style="color: white; margin: 0;">{rec['title']}</h4>
                </div>
                <p style="color: rgba(255, 255, 255, 0.9); margin: 0; line-height: 1.5;">{rec['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def display_feature_cards():
    """Display feature cards for demo"""
    st.markdown('<div class="section-header">✨ Platform Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        ('🤖', 'AI-Powered Analysis', 'Advanced NLP algorithms analyze your resume with precision and provide actionable insights.'),
        ('📊', 'Visual Analytics', 'Beautiful charts and metrics help you understand your resume\'s strengths and areas for improvement.'),
        ('🎯', 'Job Matching', 'Smart algorithms compare your resume against job descriptions for perfect alignment.')
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <h3 style="color: white; margin-bottom: 1rem; font-weight: 600;">{title}</h3>
                <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AI Resume Analyzer | Modern CV Analysis",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Display modern header
    display_modern_header()
    
    # Initialize analyzer
    analyzer = ResumeAnalyzer()
    
    # Modern sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h2 style="color: white; font-weight: 600;">📄 Upload Resume</h2>
            <p style="color: rgba(255, 255, 255, 0.8);">Drag and drop your resume file below</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose your resume",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files"
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h3 style="color: white; font-weight: 600;">💼 Job Description</h3>
            <p style="color: rgba(255, 255, 255, 0.8);">Paste job description for better matching</p>
        </div>
        """, unsafe_allow_html=True)
        
        job_description = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste the job description here to get accurate job matching scores...",
            label_visibility="collapsed"
        )
        
        # Quick tips
        with st.expander("💡 Quick Tips"):
            st.markdown("""
            <div style="color: rgba(255, 255, 255, 0.9); line-height: 1.6;">
                <strong>📈 Improve Your Score:</strong><br>
                • Use action verbs (developed, managed, led)<br>
                • Include quantifiable achievements<br>
                • Add relevant technical skills<br>
                • Match keywords from job descriptions<br>
                • Keep content detailed but concise
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file is not None:
        # Create columns for layout
        main_col, chart_col = st.columns([2, 1])
        
        with main_col:
            # Show loading animation
            with st.spinner("🤖 AI is analyzing your resume..."):
                # Simulate processing time for better UX
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    resume_text = analyzer.extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resume_text = analyzer.extract_text_from_docx(uploaded_file)
                else:  # txt file
                    resume_text = str(uploaded_file.read(), "utf-8")
                
                progress_bar.empty()
                
                if resume_text:
                    # Analyze resume
                    results = analyzer.analyze_resume(resume_text, job_description)
                    
                    # Display success message
                    st.success("✅ Analysis Complete! Here are your results:")
                    
                    # Display metric cards
                    display_metric_cards(results)
                    
                    # Additional metrics row
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    additional_metrics = [
                        ("Years Experience", results['total_years_experience'], "⏱️"),
                        ("Resume Length", f"{results['resume_length']} words", "📝"),
                        ("Skills Found", sum(len(skills) for skills in results['skills'].values()), "🔧"),
                        ("Categories", len([cat for cat, skills in results['skills'].items() if skills]), "📂")
                    ]
                    
                    for i, (label, value, icon) in enumerate(additional_metrics):
                        with [col1, col2, col3, col4][i]:
                            st.markdown(f"""
                            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px); 
                                        border-radius: 15px; padding: 1.5rem; text-align: center; 
                                        border: 1px solid rgba(255, 255, 255, 0.2); margin: 0.5rem 0;">
                                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: white; margin-bottom: 0.25rem;">{value}</div>
                                <div style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.7);">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)
        
        with chart_col:
            # Radar chart
            radar_chart = create_radar_chart(results)
            st.plotly_chart(radar_chart, use_container_width=True)
        
        # Full-width visualizations section
        st.markdown('<div class="section-header">📊 Visual Analytics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Skills chart
            skills_chart = create_modern_skills_chart(results['skills'])
            if skills_chart:
                st.plotly_chart(skills_chart, use_container_width=True)
            else:
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px); 
                            border-radius: 20px; padding: 3rem; text-align: center; 
                            border: 1px solid rgba(255, 255, 255, 0.2);">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                    <h3 style="color: white;">No Skills Detected</h3>
                    <p style="color: rgba(255, 255, 255, 0.8);">Try uploading a more detailed resume with technical skills listed.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Overall score gauge
            overall_gauge = create_modern_gauge(results['overall_score'], "Overall Score")
            st.plotly_chart(overall_gauge, use_container_width=True)
        
        # Skills portfolio section
        if any(skills for skills in results['skills'].values()):
            display_skills_tags(results['skills'])
        
        # Recommendations section
        display_recommendations(results)
        
        # Detailed analysis section
        with st.expander("🔬 Detailed Analysis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Score Breakdown")
                
                # Score breakdown chart
                scores_data = {
                    'Category': ['Experience', 'Education', 'Job Match'],
                    'Score': [results['experience_score'], results['education_score'], results['job_match_score']],
                    'Weight': ['40%', '30%', '30%']
                }
                
                breakdown_fig = px.bar(
                    x=scores_data['Category'],
                    y=scores_data['Score'],
                    title="Score Components",
                    color=scores_data['Score'],
                    color_continuous_scale='viridis'
                )
                
                breakdown_fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title=dict(font=dict(size=16, color='white'), x=0.5),
                    showlegend=False,
                    height=300
                )
                
                st.plotly_chart(breakdown_fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Resume Statistics")
                
                # Create statistics
                stats_data = {
                    'Metric': ['Total Words', 'Unique Skills', 'Experience Keywords', 'Education Keywords'],
                    'Count': [
                        results['resume_length'],
                        sum(len(skills) for skills in results['skills'].values()),
                        sum(1 for keyword in analyzer.experience_keywords if keyword in resume_text.lower()),
                        sum(1 for keyword in analyzer.education_keywords if keyword in resume_text.lower())
                    ]
                }
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(
                    stats_df,
                    use_container_width=True,
                    hide_index=True
                )
        
        # Resume text preview
        with st.expander("📄 Resume Preview", expanded=False):
            preview_text = resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; 
                        padding: 1.5rem; border: 1px solid rgba(255, 255, 255, 0.1);
                        font-family: 'Courier New', monospace; color: rgba(255, 255, 255, 0.9);
                        line-height: 1.6; max-height: 400px; overflow-y: auto;">
                {preview_text.replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        # Export results section
        st.markdown('<div class="section-header">📤 Export Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Report", use_container_width=True):
                # Create a simple report
                report_data = {
                    'Metric': ['Overall Score', 'Experience Score', 'Education Score', 'Job Match Score', 'Total Skills'],
                    'Value': [
                        f"{results['overall_score']:.1f}%",
                        f"{results['experience_score']:.1f}%",
                        f"{results['education_score']:.1f}%",
                        f"{results['job_match_score']:.1f}%",
                        sum(len(skills) for skills in results['skills'].values())
                    ]
                }
                
                report_df = pd.DataFrame(report_data)
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="resume_analysis_report.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🔄 Analyze Another", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("💡 Get Tips", use_container_width=True):
                st.info("💡 Check the sidebar for quick improvement tips!")
    
    else:
        # Landing page when no file is uploaded
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Feature showcase
        display_feature_cards()
        
        # Demo section
        st.markdown('<div class="section-header">🎮 Try the Demo</div>', unsafe_allow_html=True)
        
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        
        with demo_col1:
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px); 
                        border-radius: 20px; padding: 2rem; text-align: center; 
                        border: 1px solid rgba(255, 255, 255, 0.2); height: 200px; 
                        display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📤</div>
                <h4 style="color: white; margin-bottom: 0.5rem;">Step 1</h4>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">Upload your resume in PDF, DOCX, or TXT format</p>
            </div>
            """, unsafe_allow_html=True)
        
        with demo_col2:
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px); 
                        border-radius: 20px; padding: 2rem; text-align: center; 
                        border: 1px solid rgba(255, 255, 255, 0.2); height: 200px; 
                        display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🤖</div>
                <h4 style="color: white; margin-bottom: 0.5rem;">Step 2</h4>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">AI analyzes your resume using advanced NLP algorithms</p>
            </div>
            """, unsafe_allow_html=True)
        
        with demo_col3:
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px); 
                        border-radius: 20px; padding: 2rem; text-align: center; 
                        border: 1px solid rgba(255, 255, 255, 0.2); height: 200px; 
                        display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📊</div>
                <h4 style="color: white; margin-bottom: 0.5rem;">Step 3</h4>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">Get detailed insights and actionable recommendations</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Call to action
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255, 255, 255, 0.05); 
                    border-radius: 25px; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h2 style="color: white; margin-bottom: 1rem;">Ready to optimize your resume?</h2>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.2rem; margin-bottom: 2rem;">
                Join thousands of professionals who have improved their resumes with AI-powered insights
            </p>
            <div style="font-size: 1.5rem;">👆 Upload your resume in the sidebar to get started!</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()