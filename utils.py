import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

def load_css():
    """Load optimized CSS for Streamlit Cloud deployment"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Reset and Base Styles */
    .stApp {
        background: #1a1a2e; /* Dark, professional background */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e0e0e0; /* Light grey text for contrast */
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        background: rgba(255, 255, 255, 0.05); /* More subtle background */
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 12px; /* Slightly less rounded */
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3); /* Softer shadow */
    }
    
    .main-title {
        font-size: clamp(2rem, 5vw, 3rem); /* Slightly smaller, more refined */
        font-weight: 700;
        color: #8be9fd; /* Professional blue accent */
        margin-bottom: 0.8rem;
        text-shadow: none; /* Remove text shadow */
    }
    
    .main-subtitle {
        font-size: clamp(0.9rem, 2.5vw, 1.1rem); /* More concise */
        color: rgba(255, 255, 255, 0.7);
        font-weight: 400;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.08); /* Darker, more subtle */
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.4rem 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-1px); /* More subtle hover */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: clamp(1.4rem, 3.5vw, 2.2rem);
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.4rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1y4p8pa {
        background: #2a2a4a; /* Darker sidebar */
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    
    .css-17eq0hr, .css-1544g2n {
        background: #2a2a4a;
    }
    
    /* Section Headers */
    .section-header {
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: 600;
        color: #ffffff;
        margin: 1.8rem 0 1rem 0;
        text-align: center;
        position: relative;
    }
    
    .section-header:after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 50px;
        height: 2px;
        background: #8be9fd; /* Accent color */
        border-radius: 2px;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 10px;
        padding: 1.8rem;
        margin: 0.8rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
        color: #a1c4fd; /* Muted accent for icons */
    }
    
    /* Skills Tags */
    .skill-tag {
        background: #3a3a5a; /* Darker, more professional tag color */
        color: #f0f0f0;
        padding: 0.3rem 0.7rem;
        margin: 0.15rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.1);
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-left: 3px solid #8be9fd; /* Accent color */
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%) !important; /* Professional gradient */
        border: none !important;
        border-radius: 20px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        color: white !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        width: auto !important; /* Allow buttons to size naturally */
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
        background: linear-gradient(45deg, #2575fc 0%, #6a11cb 100%) !important; /* Reverse gradient on hover */
    }

    /* File uploader styling */
    .stFileUploader > div > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }
    
    /* Text areas */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        color: #f0f0f0 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #8be9fd, #a1c4fd) !important; /* Professional progress bar */
        border-radius: 8px !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: #f0f0f0 !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header,
        .hero-section,
        .st-section {
            padding: 1.5rem;
            margin: 1.5rem auto;
        }
        
        .feature-card {
            margin: 0.5rem;
            padding: 1.2rem;
        }
        
        .metric-card {
            padding: 0.8rem;
        }
        .navbar-links {
            display: none; /* Hide for small screens, implement hamburger menu if needed */
        }
        .navbar-logo {
            width: 100%;
            text-align: center;
        }
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #8be9fd !important; /* Professional spinner color */
    }
    
    /* Success/Info messages */
    .stAlert > div {
        background: rgba(40, 44, 52, 0.8) !important; /* Darker alert background */
        backdrop-filter: blur(8px) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(139, 233, 253, 0.3) !important; /* Accent border */
        color: #f0f0f0 !important;
    }

    /* New Styles for Professional UI */

    /* Sticky Navbar */
    .sticky-navbar {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(26, 26, 46, 0.8); /* Darker, more solid background */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 0.8rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.4);
    }
    .navbar-logo {
        font-size: 1.4rem;
        font-weight: 700;
        color: #8be9fd; /* Professional accent color */
        text-shadow: none;
    }
    .navbar-links a {
        color: rgba(255, 255, 255, 0.7);
        text-decoration: none;
        margin-left: 1.8rem;
        font-weight: 500;
        transition: color 0.2s ease, transform 0.1s ease;
    }
    .navbar-links a:hover {
        color: #8be9fd; /* Accent color on hover */
        transform: translateY(-1px);
    }

    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 5rem 2rem;
        background: #1f1f3f; /* Dark, solid background */
        border-radius: 15px;
        margin: 2.5rem auto;
        max-width: 800px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 5px 25px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.8s ease-out; /* Slightly faster animation */
    }
    .hero-headline {
        font-size: clamp(2rem, 5vw, 3.8rem);
        font-weight: 800;
        color: #ffffff; /* Solid white for headline */
        margin-bottom: 1rem;
        text-shadow: none;
    }
    .hero-subtext {
        font-size: clamp(1rem, 2vw, 1.3rem);
        color: rgba(255, 255, 255, 0.6);
        font-weight: 400;
        line-height: 1.6;
        max-width: 600px;
        margin: 0 auto 2rem auto;
    }
    .hero-button .stButton > button {
        padding: 0.7rem 2rem !important;
        font-size: 1rem !important;
        border-radius: 25px !important;
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    .hero-button .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
    }

    /* General Section Styling */
    .st-section {
        padding: 3rem 2rem;
        margin: 2.5rem auto;
        max-width: 1000px;
        background: #1f1f3f; /* Consistent dark background */
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        animation: slideInUp 0.7s ease-out; /* Slightly faster animation */
    }
    .st-section-title {
        font-size: clamp(1.8rem, 3.5vw, 2.8rem);
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2.5rem;
        position: relative;
    }
    .st-section-title:after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: #8be9fd; /* Accent color */
        border-radius: 1.5px;
    }

    /* Input field and textarea styling */
    input[type="text"], input[type="email"], textarea {
        background: #2a2a4a !important; /* Darker input background */
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
        transition: all 0.2s ease;
    }
    input[type="text"]:focus, input[type="email"]:focus, textarea:focus {
        border-color: #8be9fd !important; /* Accent color on focus */
        box-shadow: 0 0 0 2px rgba(139, 233, 253, 0.2) !important;
        outline: none;
    }
    input::placeholder, textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# Predefined skill categories and job requirements
SKILL_CATEGORIES = {
    'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'kotlin', 'swift', 'scala', 'dart'],
    'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'nextjs', 'nuxt'],
    'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib', 'seaborn', 'jupyter', 'keras', 'spark'],
    'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'oracle', 'cassandra', 'dynamodb'],
    'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'heroku', 'vercel'],
    'analytics': ['excel', 'tableau', 'powerbi', 'looker', 'google analytics', 'sql', 'r', 'stata', 'spss'],
    'project_management': ['agile', 'scrum', 'kanban', 'jira', 'trello', 'asana', 'confluence', 'slack'],
    'soft_skills': ['leadership', 'communication', 'teamwork', 'problem-solving', 'analytical', 'creative', 'management']
}

EXPERIENCE_KEYWORDS = ['experience', 'worked', 'developed', 'managed', 'led', 'created', 'implemented', 
                      'designed', 'built', 'maintained', 'optimized', 'improved', 'collaborated', 'achieved']

EDUCATION_KEYWORDS = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'certification', 
                     'course', 'training', 'certified', 'diploma', 'graduate']
