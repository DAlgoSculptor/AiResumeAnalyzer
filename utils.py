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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles & Reset */
    html, body {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        scroll-behavior: smooth;
    }

    .stApp {
        background-color: #121212; /* Deep dark background */
        font-family: 'Inter', sans-serif;
        color: #e0e0e0; /* Soft white for readability */
        margin: 0;
        padding: 0;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .main .block-container {
        padding: 2rem 1rem 4rem; /* Adjusted padding for better content flow */
        max-width: 1200px; /* Consistent max width */
        margin: 0 auto; /* Center content */
    }
    
    /* Streamlit Overrides */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* --- Navbar Styling --- */
    .sticky-navbar {
        position: sticky;
        top: 0;
        z-index: 1000;
        background-color: rgba(18, 18, 18, 0.95); /* Slightly transparent dark */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 0.8rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08); /* Subtle separator */
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.5); /* Pronounced shadow */
    }
    .navbar-logo {
        font-size: 1.6rem;
        font-weight: 800;
        color: #00bcd4; /* Accent color */
        text-transform: uppercase;
        letter-spacing: 1px;
        cursor: pointer;
        transition: color 0.3s ease;
    }
    .navbar-logo:hover {
        color: #00e5ff; /* Lighter accent on hover */
    }
    .navbar-links {
        display: flex;
        gap: 2rem; /* Space out links */
    }
    .navbar-links a {
        color: rgba(255, 255, 255, 0.7);
        text-decoration: none;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.5rem 0;
        position: relative;
        transition: color 0.3s ease;
    }
    .navbar-links a:hover {
        color: #00bcd4; /* Accent color on hover */
    }
    .navbar-links a::after {
        content: '';
        position: absolute;
        left: 0;
        bottom: 0;
        width: 0%;
        height: 2px;
        background-color: #00bcd4; /* Underline accent */
        transition: width 0.3s ease;
    }
    .navbar-links a:hover::after {
        width: 100%;
    }

    /* --- Hero Section --- */
    .hero-section {
        text-align: center;
        padding: 8rem 2rem; /* More generous padding */
        background: linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%); /* Subtle gradient */
        border-radius: 20px;
        margin: 3rem auto; /* Consistent margin */
        max-width: 1000px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6); /* Deeper shadow */
        animation: fadeInScale 1s ease-out forwards; /* New animation */
        opacity: 0;
        transform: scale(0.95);
    }
    .hero-headline {
        font-size: clamp(2.8rem, 6vw, 4.8rem);
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 1.5rem;
        line-height: 1.2;
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
    }
    .hero-subtext {
        font-size: clamp(1.1rem, 2.5vw, 1.5rem);
        color: rgba(255, 255, 255, 0.7);
        font-weight: 400;
        line-height: 1.7;
        max-width: 700px;
        margin: 0 auto 3rem auto;
    }
    .hero-button .stButton > button {
        padding: 0.9rem 2.8rem !important;
        font-size: 1.15rem !important;
        border-radius: 30px !important;
        background: linear-gradient(45deg, #00bcd4 0%, #00e5ff 100%) !important; /* New vibrant gradient */
        box-shadow: 0 6px 20px rgba(0, 188, 212, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .hero-button .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(0, 188, 212, 0.6) !important;
        background: linear-gradient(45deg, #00e5ff 0%, #00bcd4 100%) !important; /* Reverse gradient on hover */
    }

    /* --- General Section Styling --- */
    .st-section {
        padding: 5rem 2.5rem; /* Increased padding for sections */
        margin: 4rem auto; /* More vertical separation */
        max-width: 1100px; /* Slightly wider sections */
        background-color: #1f1f1f; /* Solid dark background for sections */
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.4); /* Consistent shadow */
        animation: slideInUp 0.8s ease-out forwards; /* Animation for sections */
        opacity: 0;
        transform: translateY(30px);
    }
    .st-section-title {
        font-size: clamp(2.2rem, 4.5vw, 3.5rem);
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
        width: 80px; /* Wider underline */
        height: 3px;
        background-color: #00bcd4; /* Accent color */
        border-radius: 1.5px;
    }

    /* --- Metric Cards --- */
    .metric-card {
        background-color: #282828; /* Slightly lighter dark */
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.7rem 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.4);
    }
    .metric-value {
        font-size: clamp(1.8rem, 4vw, 2.8rem);
        font-weight: 700;
        color: #00e5ff; /* Bright accent */
        margin-bottom: 0.6rem;
    }
    .metric-label {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 500;
    }

    /* --- Feature Cards (from Services/Features sections) --- */
    .feature-card {
        background-color: #282828;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 3px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%; /* Ensure consistent height in flex containers */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.45);
    }
    .feature-icon {
        font-size: 3rem; /* Larger icons */
        margin-bottom: 1rem;
        color: #00bcd4; /* Accent color */
    }
    .feature-card h3 {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
    }
    .feature-card p {
        color: rgba(255, 255, 255, 0.7);
        line-height: 1.6;
        font-size: 0.95rem;
    }

    /* --- Skills Tags --- */
    .skill-tag {
        background-color: #005f6b; /* Darker accent background */
        color: #ffffff;
        padding: 0.4rem 0.9rem;
        margin: 0.3rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.2);
        transition: background-color 0.2s ease;
    }
    .skill-tag:hover {
        background-color: #00838f; /* Lighter on hover */
    }

    /* --- Recommendation Cards --- */
    .recommendation-card {
        background-color: #282828;
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1rem 0;
        border-left: 5px solid #00bcd4; /* More prominent accent border */
        box-shadow: 0 3px 15px rgba(0, 0, 0, 0.3);
    }
    .recommendation-card h4 {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
    }
    .recommendation-card p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
    }
    .recommendation-card span {
        color: #00e5ff; /* Icon accent */
    }

    /* --- Buttons --- */
    .stButton > button {
        background: linear-gradient(45deg, #00bcd4 0%, #00e5ff 100%) !important; /* Primary button gradient */
        border: none !important;
        border-radius: 25px !important;
        padding: 0.7rem 2.2rem !important;
        font-weight: 600 !important;
        color: white !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(0, 188, 212, 0.3) !important;
        cursor: pointer;
        width: auto; /* Allow natural button width */
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 6px 20px rgba(0, 188, 212, 0.5) !important;
        background: linear-gradient(45deg, #00e5ff 0%, #00bcd4 100%) !important; /* Reverse gradient on hover */
    }

    /* --- File Uploader --- */
    .stFileUploader > div > div > div {
        background-color: #282828 !important;
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 2rem;
        transition: border-color 0.2s ease;
    }
    .stFileUploader > div > div > div:hover {
        border-color: #00bcd4 !important;
    }

    /* --- Text Areas & Inputs --- */
    .stTextArea textarea,
    input[type="text"],
    input[type="email"] {
        background-color: #282828 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
        padding: 0.8rem 1rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextArea textarea:focus,
    input[type="text"]:focus,
    input[type="email"]:focus {
        border-color: #00bcd4 !important;
        box-shadow: 0 0 0 3px rgba(0, 188, 212, 0.3) !important;
        outline: none;
    }
    input::placeholder, textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }

    /* --- Progress Bar --- */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00bcd4, #00e5ff) !important; /* Accent gradient */
        border-radius: 8px !important;
    }

    /* --- Expander Styling --- */
    .streamlit-expanderHeader {
        background-color: #282828 !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
        padding: 1rem 1.2rem;
        transition: background-color 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background-color: #333333 !important;
    }

    /* --- Alert Messages --- */
    .stAlert > div {
        background-color: #282828 !important;
        border-radius: 8px !important;
        border: 1px solid rgba(0, 188, 212, 0.3) !important; /* Accent border */
        color: #e0e0e0 !important;
        padding: 1rem 1.2rem;
    }
    .stAlert > div [data-testid="stMarkdownContainer"] {
        color: #e0e0e0 !important; /* Ensure text color is readable */
    }

    /* --- Loading Spinner --- */
    .stSpinner > div {
        border-top-color: #00bcd4 !important; /* Accent color */
    }

    /* --- Animations --- */
    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }

    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(50px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- Responsive Design --- */
    @media (max-width: 768px) {
        .sticky-navbar {
            padding: 0.6rem 1rem;
        }
        .navbar-links {
            display: none; /* Mobile navigation toggle (hamburger menu) would be implemented here */
        }
        .navbar-logo {
            width: 100%;
            text-align: center;
            font-size: 1.4rem;
        }
        .hero-section {
            padding: 5rem 1rem;
            margin: 2rem auto;
        }
        .hero-headline {
            font-size: clamp(2rem, 8vw, 3.5rem);
        }
        .hero-subtext {
            font-size: clamp(0.9rem, 3vw, 1.2rem);
            margin-bottom: 2rem;
        }
        .st-section {
            padding: 3rem 1.5rem;
            margin: 3rem auto;
        }
        .st-section-title {
            font-size: clamp(1.8rem, 6vw, 2.5rem);
            margin-bottom: 2rem;
        }
        .feature-card {
            padding: 1.5rem;
            margin: 0.5rem 0;
        }
        .metric-card {
            padding: 1rem;
        }
        .stButton > button {
            padding: 0.6rem 1.8rem !important;
            font-size: 0.95rem !important;
        }
        .stFileUploader > div > div > div {
            padding: 1.5rem;
        }
        .stTextArea textarea, input[type="text"], input[type="email"] {
            padding: 0.6rem 0.8rem !important;
        }
    }

    /* Sidebar Styling */
    .css-1d391kg, .css-1y4p8pa {
        background-color: #1a1a1a; /* Darker, solid sidebar background */
        /* backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); Removed for consistency */
    }
    
    .css-17eq0hr, .css-1544g2n {
        background-color: #1a1a1a; /* Consistent dark background */
    }

    /* Specific Streamlit components within sidebar for professional look */
    .stFileUploader {
        background-color: #1a1a1a !important; /* Ensure uploader background matches sidebar */
    }
    .stTextArea {
        background-color: #1a1a1a !important;
    }

    /* Text within sidebar titles/headers */
    [data-testid="stSidebarContent"] h2, [data-testid="stSidebarContent"] h3 {
        color: #e0e0e0; /* Light text for readability */
        text-shadow: none; /* Remove any text shadows */
        text-align: center; /* Center sidebar titles */
    }

    /* General text in sidebar */
    [data-testid="stSidebarContent"] p {
        color: rgba(255, 255, 255, 0.6); /* Slightly muted text for less emphasis */
        text-align: center; /* Center sidebar descriptions */
    }

    /* File uploader input area */
    .stFileUploader > div > div > div {
        background-color: #282828 !important; /* Darker background for drop area */
        border: 2px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
        padding: 1.5rem;
    }
    .stFileUploader > div > div > div:hover {
        border-color: #00bcd4 !important;
    }
    .stFileUploader label {
        color: #e0e0e0 !important;
        text-align: center; /* Center file uploader label */
        display: block; /* Make label a block to center */
        width: 100%;
    }

    /* File uploader uploaded file name */
    .stFileUploader span[data-testid="stFileUploadDropzoneText"] {
        color: #e0e0e0 !important;
    }

    /* Individual uploaded file info (e.g., wiserResume.pdf) */
    .stFileUploader [data-testid^="stFileUploaderDropzoneUploadedFile"] div {
        background-color: #282828 !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        color: #e0e0e0 !important;
        border-radius: 8px;
        margin-top: 0.5rem;
        padding: 0.5rem 1rem; /* Adjust padding for uploaded file info */
        display: flex; /* Use flexbox to align items */
        align-items: center;
        justify-content: space-between; /* Space between file name and close button */
    }
    .stFileUploader [data-testid^="stFileUploaderDropzoneUploadedFile"] p {
        margin: 0; /* Remove default paragraph margin */
        color: #e0e0e0 !important;
        text-align: left; /* Align file name to left */
    }
    .stFileUploader [data-testid^="stFileUploaderDropzoneUploadedFile"] button {
        color: rgba(255, 255, 255, 0.6) !important; /* Close button color */
        font-size: 1.2rem; /* Larger close button */
        background: none !important; /* No background for close button */
        border: none !important;
        padding: 0; /* Remove padding */
        margin-left: 0.5rem; /* Space from file name */
    }

    /* Overall sidebar padding and structure */
    [data-testid="stSidebarContent"] {
        padding: 2rem 1.5rem; /* More generous padding for sidebar content */
    }

    /* Horizontal rule in sidebar */
    .stHorizontalRule {
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin: 1.5rem 0; /* Adjust margin for hr */
    }

    /* Sidebar expand/collapse button */
    [data-testid="stSidebarCollapseButton"] {
        color: #e0e0e0 !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 50%;
        transition: all 0.2s ease;
        top: 1rem; /* Adjust position */
        right: 1rem;
    }
    [data-testid="stSidebarCollapseButton"]:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #00bcd4 !important;
    }

    /* Analyze Resume button in sidebar */
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important; /* Full width for sidebar button */
        margin-top: 1.5rem; /* Space above button */
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-1px) !important; /* Subtle hover */
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
