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

# 🚀 MUST BE FIRST STREAMLIT COMMAND
# ✅ MUST BE FIRST Streamlit command
st.set_page_config(
page_title="AI Resume Analyzer",
page_icon="🚀",
layout="wide",
initial_sidebar_state="expanded"
)


# Download required NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    """Download NLTK data with caching"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

# Initialize NLTK data
download_nltk_data()

def load_css():
    """Load optimized CSS for Streamlit Cloud deployment"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Reset and Base Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Header Styling - Fixed for deployment */
    .main-header {
        text-align: center;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .main-title {
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: #ffffff; /* fallback */
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
    }
    
    .main-subtitle {
        font-size: clamp(1rem, 3vw, 1.3rem);
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Card Styling - Improved for deployment */
    .metric-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 20px rgba(31, 38, 135, 0.3);
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: clamp(1.5rem, 4vw, 2.5rem);
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
    }
    
    /* Sidebar Styling - Fixed for deployment */
    .css-1d391kg, .css-1y4p8pa {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .css-17eq0hr, .css-1544g2n {
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        font-size: clamp(1.5rem, 4vw, 2rem);
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        text-align: center;
        position: relative;
    }
    
    .section-header:after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border-radius: 2px;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.2s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Skills Tags */
    .skill-tag {
        display: inline-block;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 0.4rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4ecdc4;
        box-shadow: 0 4px 15px rgba(31, 38, 135, 0.3);
    }
    
    /* Buttons - Fixed for Streamlit */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        color: white !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
    }
    
    /* Text areas */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4) !important;
        border-radius: 10px !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .feature-card {
            margin: 0.5rem;
            padding: 1rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #4ecdc4 !important;
    }
    
    /* Success/Info messages */
    .stAlert > div {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)

class ResumeAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Predefined skill categories and job requirements
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'kotlin', 'swift', 'scala', 'dart'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'nextjs', 'nuxt'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib', 'seaborn', 'jupyter', 'keras', 'spark'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'oracle', 'cassandra', 'dynamodb'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'heroku', 'vercel'],
            'analytics': ['excel', 'tableau', 'powerbi', 'looker', 'google analytics', 'sql', 'r', 'stata', 'spss'],
            'project_management': ['agile', 'scrum', 'kanban', 'jira', 'trello', 'asana', 'confluence', 'slack'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem-solving', 'analytical', 'creative', 'management']
        }
        
        self.experience_keywords = ['experience', 'worked', 'developed', 'managed', 'led', 'created', 'implemented', 
                                  'designed', 'built', 'maintained', 'optimized', 'improved', 'collaborated', 'achieved']
        
        self.education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'certification', 
                                 'course', 'training', 'certified', 'diploma', 'graduate']
    
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

def create_skills_chart(skills_data):
    """Create skills visualization"""
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
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            title=dict(font=dict(size=16, color='white'), x=0.5),
            showlegend=False,
            height=300,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)', zeroline=False)
        )
        
        fig.update_traces(
            marker=dict(line=dict(color='rgba(255,255,255,0.3)', width=1)),
            hovertemplate='<b>%{x}</b><br>Skills: %{y}<extra></extra>'
        )
        
        return fig
    return None

def create_gauge_chart(score, title):
    """Create gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': 'white'}},
        number={'font': {'size': 32, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white', 'size': 10}},
            'bar': {'color': "#4ecdc4", 'thickness': 0.6},
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
                'line': {'color': "#ff6b6b", 'width': 3},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return fig

def create_radar_chart(results):
    """Create radar chart for comprehensive analysis"""
    categories = ['Experience', 'Education', 'Job Match', 'Skills Diversity', 'Content Quality']
    values = [
        results['experience_score'],
        results['education_score'],
        results['job_match_score'],
        min(100, sum(len(skills) for skills in results['skills'].values()) * 8),
        min(100, results['resume_length'] / 3)
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
                tickfont=dict(color='white', size=10)
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.3)',
                tickcolor='white',
                tickfont=dict(color='white', size=10)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        title=dict(text="Resume Analysis Overview", font=dict(size=16, color='white'), x=0.5),
        height=350,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def display_header():
    """Display header"""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🚀 AI Resume Analyzer</div>
        <div class="main-subtitle">
            Transform your career with AI-powered resume insights • Advanced NLP • Real-time Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_metrics(results):
    """Display metric cards"""
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
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <div class="metric-value">{value:.1f}%</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def display_skills(skills_data):
    """Display skills as tags"""
    st.markdown('<div class="section-header">🛠️ Skills Portfolio</div>', unsafe_allow_html=True)
    
    for category, skills in skills_data.items():
        if skills:
            st.markdown(f"""
            <h4 style="color: white; margin: 1rem 0 0.5rem 0; font-weight: 500;">
                {category.replace('_', ' ').title()}
            </h4>
            """, unsafe_allow_html=True)
            
            tags_html = ""
            for skill in skills:
                tags_html += f'<span class="skill-tag">{skill}</span>'
            
            st.markdown(tags_html, unsafe_allow_html=True)

def display_recommendations(results):
    """Display recommendations"""
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
            'desc': 'Include certifications, courses, and relevant training to strengthen your profile.'
        })
    
    total_skills = sum(len(skills) for skills in results['skills'].values())
    if total_skills < 8:
        recommendations.append({
            'icon': '🛠️',
            'title': 'Expand Skill Set',
            'desc': 'Add more technical and soft skills relevant to your target role.'
        })
    
    if results['resume_length'] < 250:
        recommendations.append({
            'icon': '📝',
            'title': 'Expand Content',
            'desc': 'Provide more detailed descriptions of your projects and accomplishments.'
        })
    
    if not recommendations:
        st.success("🌟 Excellent Resume! Your resume demonstrates strong alignment with best practices.")
    else:
        for rec in recommendations:
            st.markdown(f"""
            <div class="recommendation-card">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem; margin-right: 1rem;">{rec['icon']}</span>
                    <h4 style="color: white; margin: 0; font-size: 1.1rem;">{rec['title']}</h4>
                </div>
                <p style="color: rgba(255, 255, 255, 0.9); margin: 0; line-height: 1.4; font-size: 0.95rem;">{rec['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def display_features():
    """Display feature cards"""
    st.markdown('<div class="section-header">✨ Platform Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        ('🤖', 'AI-Powered Analysis', 'Advanced NLP algorithms analyze your resume with precision.'),
        ('📊', 'Visual Analytics', 'Beautiful charts help you understand your resume\'s strengths.'),
        ('🎯', 'Job Matching', 'Smart algorithms compare your resume against job descriptions.')
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <h3 style="color: white; margin-bottom: 1rem; font-weight: 600; font-size: 1.2rem;">{title}</h3>
                <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.5; margin: 0; font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app"""
    load_css()  # Load custom CSS for styling
    display_header()  # Display header

    # Initialize analyzer ✅ FIXED
    analyzer = ResumeAnalyzer()  # Initialize analyzer
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="color: white; font-weight: 600; font-size: 1.5rem;">📄 Upload Resume</h2>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">Upload your resume file below</p>
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
                    <h3 style="color: white; font-weight: 600; font-size: 1.2rem;">🎯 Job Description</h3>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Paste the job description here for a tailored match score.</p>
        </div>
        
        """, unsafe_allow_html=True)
        job_description = st.text_area("Paste job description", height=150, key="jobdesc")
        
        analyze_button = st.button("🔍 Analyze Resume")
    
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
                display_features()
            else:
                st.error("⚠️ Could not extract text from the uploaded file. Please try another format.")
    elif analyze_button and not uploaded_file:
        st.warning("⚠️ Please upload a resume file before analysis.")

if __name__ == "__main__":
    main()
