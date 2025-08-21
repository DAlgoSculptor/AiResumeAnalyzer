import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import SKILL_CATEGORIES

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
        ("Overall Score", results['overall_score'], "✅"),
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
    st.markdown('<div class="section-header">⚙️ Platform Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        ('🤖', 'AI-Powered Analysis', 'Advanced NLP algorithms analyze your resume with precision.'),
        ('📊', 'Visual Analytics', 'Beautiful charts help you understand your resume\'s strengths.'),
        ('🔍', 'Job Matching', 'Smart algorithms compare your resume against job descriptions.')
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

def display_sidebar():
    """Display sidebar elements for resume upload and job description input"""
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
                    <h3 style="color: white; font-weight: 600; font-size: 1.2rem;">📋 Job Description</h3>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Paste the job description here for a tailored match score.</p>
        </div>
        
        """, unsafe_allow_html=True)
        job_description = st.text_area("Paste job description", height=150, key="jobdesc")
        
        analyze_button = st.button("🔍 Analyze Resume")
    
    return uploaded_file, job_description, analyze_button

def display_hero_section():
    """Displays the hero section with headline and CTA button.
    This section will replace the initial `display_header()` call.
    """
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-headline">Unlock Your Career Potential with AI</h1>
        <p class="hero-subtext">
            Our advanced AI Resume Analyzer provides instant feedback, smart job matching, and personalized recommendations to help you stand out.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Use columns to center the button
    col1, col2, col3 = st.columns([1, 2, 1]) # Adjust ratios if needed for better centering

    with col2:
        get_started_button = st.button("💼 Get Started Now", key="get_started_hero_button")
    return get_started_button

def display_about_section():
    """Displays the About Us section."""
    st.markdown("""
    <div class="st-section">
        <h2 class="st-section-title">About AI Resume Analyzer</h2>
        <p style="color: rgba(255, 255, 255, 0.8); text-align: center; line-height: 1.6;">
            The AI Resume Analyzer is a cutting-edge tool designed to revolutionize your job application process.
            Leveraging state-of-the-art Natural Language Processing (NLP) and machine learning, we provide a comprehensive analysis of your resume,
            identifying key strengths and areas for improvement. Our mission is to empower job seekers with intelligent insights,
            helping them craft resumes that resonate with recruiters and land their dream jobs.
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_services_section():
    """Displays the Services section with key offerings."""
    st.markdown("""
    <div class="st-section">
        <h2 class="st-section-title">Our Services</h2>
        <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;">
            <div class="feature-card" style="flex: 1; min-width: 280px;">
                <div class="feature-icon">📈</div>
                <h3 style="color: white;">Detailed Resume Scoring</h3>
                <p style="color: rgba(255, 255, 255, 0.8);">Get an overall score and breakdowns for experience, education, and skills.</p>
            </div>
            <div class="feature-card" style="flex: 1; min-width: 280px;">
                <div class="feature-icon">🔍</div>
                <h3 style="color: white;">Precise Job Matching</h3>
                <p style="color: rgba(255, 255, 255, 0.8);">Compare your resume against any job description for tailored compatibility.</p>
            </div>
            <div class="feature-card" style="flex: 1; min-width: 280px;">
                <div class="feature-icon">💡</div>
                <h3 style="color: white;">Actionable Recommendations</h3>
                <p style="color: rgba(255, 255, 255, 0.8);">Receive AI-driven suggestions to optimize your resume content.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_contact_section():
    """Displays the Contact Us section."""
    st.markdown("""
    <div class="st-section">
        <h2 class="st-section-title">Contact Us</h2>
        <p style="color: rgba(255, 255, 255, 0.8); text-align: center; line-height: 1.6; margin-bottom: 2rem;">
            Have questions or feedback? Reach out to us through the form below or connect on social media.
        </p>
        <form action="#">
            <div style="margin-bottom: 1rem;">
                <input type="text" placeholder="Your Name" style="width: 100%; padding: 0.8rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.3); background: rgba(255, 255, 255, 0.08); color: white;" />
            </div>
            <div style="margin-bottom: 1rem;">
                <input type="email" placeholder="Your Email" style="width: 100%; padding: 0.8rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.3); background: rgba(255, 255, 255, 0.08); color: white;" />
            </div>
            <div style="margin-bottom: 1.5rem;">
                <textarea placeholder="Your Message" rows="5" style="width: 100%; padding: 0.8rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.3); background: rgba(255, 255, 255, 0.08); color: white;"></textarea>
            </div>
            <button class="stButton">Send Message</button>
        </form>
    </div>
    """, unsafe_allow_html=True)

def display_navbar():
    """Displays a sticky navigation bar."""
    st.markdown("""
    <div class="sticky-navbar">
        <div class="navbar-logo">AI Resume Analyzer</div>
        <div class="navbar-links">
            <a href="#hero-section">Home</a>
            <a href="#about-section">About</a>
            <a href="#services-section">Services</a>
            <a href="#features-section">Features</a>
            <a href="#contact-section">Contact</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
