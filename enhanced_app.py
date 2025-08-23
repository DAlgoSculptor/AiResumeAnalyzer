"""
Enhanced AI Resume Analyzer - Main Application
Advanced features with comprehensive analysis and modern UI
"""

import streamlit as st
import io
import time
from datetime import datetime
import hashlib

# Import enhanced components
from enhanced_analyzer import AdvancedResumeAnalyzer, DatabaseManager
from enhanced_ui import (
    create_advanced_dashboard, create_skills_radar_chart, create_experience_timeline,
    create_sentiment_gauge, create_ats_compatibility_chart, create_job_match_breakdown,
    display_advanced_recommendations, display_analysis_history
)

# Import original UI components for compatibility
from ui_elements import (
    display_navbar, display_hero_section, display_about_section, 
    display_services_section, display_contact_section
)
from utils import load_css, download_nltk_data

# Page configuration
st.set_page_config(
    page_title="Enhanced AI Resume Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
download_nltk_data()
load_css()

@st.cache_resource
def get_analyzer():
    """Get cached analyzer instance"""
    return AdvancedResumeAnalyzer()

@st.cache_resource
def get_db_manager():
    """Get cached database manager instance"""
    return DatabaseManager()

def display_enhanced_sidebar():
    """Enhanced sidebar with additional options"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="color: white; font-weight: 600; font-size: 1.5rem;">🚀 Enhanced Analyzer</h2>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">Advanced AI-powered resume analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload section
        st.markdown("### 📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose your resume",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files"
        )
        
        # Job description section
        st.markdown("### 📋 Job Description (Optional)")
        job_description = st.text_area(
            "Paste job description for tailored analysis",
            height=150,
            help="Adding a job description will provide more accurate matching scores"
        )
        
        # Analysis options
        st.markdown("### ⚙️ Analysis Options")
        
        col1, col2 = st.columns(2)
        with col1:
            include_sentiment = st.checkbox("Sentiment Analysis", value=True)
            include_entities = st.checkbox("Entity Extraction", value=True)
        with col2:
            include_ats = st.checkbox("ATS Scoring", value=True)
            save_history = st.checkbox("Save to History", value=True)
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)
        
        # Quick actions
        st.markdown("### 🔧 Quick Actions")
        if st.button("📊 View History", use_container_width=True):
            st.session_state.show_history = True
        
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared!")
        
        # Analysis stats
        db_manager = get_db_manager()
        history_count = len(db_manager.get_analysis_history(limit=100))
        
        st.markdown("### 📈 Statistics")
        st.metric("Total Analyses", history_count)
        
        return uploaded_file, job_description, analyze_button, {
            'sentiment': include_sentiment,
            'entities': include_entities,
            'ats': include_ats,
            'save_history': save_history
        }

def display_comprehensive_results(results, options):
    """Display comprehensive analysis results"""
    
    # Main dashboard
    st.markdown("## 📊 Comprehensive Analysis Dashboard")
    create_advanced_dashboard(results)
    
    # Detailed visualizations
    st.markdown("---")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        if results.get('skills_detailed'):
            st.markdown("### 🛠️ Skills Profile")
            skills_radar = create_skills_radar_chart(results['skills_detailed'])
            st.plotly_chart(skills_radar, use_container_width=True)
    
    with col2:
        if options['sentiment'] and results.get('sentiment_detailed'):
            st.markdown("### 😊 Sentiment Analysis")
            sentiment_gauge = create_sentiment_gauge(results['sentiment_detailed'])
            st.plotly_chart(sentiment_gauge, use_container_width=True)
    
    # Second row of charts
    col3, col4 = st.columns(2)
    
    with col3:
        if options['ats']:
            st.markdown("### 🤖 ATS Compatibility")
            ats_chart = create_ats_compatibility_chart(results['ats_score'])
            st.plotly_chart(ats_chart, use_container_width=True)
    
    with col4:
        if results.get('job_match_detailed') and results['job_match_detailed'].get('matched_skills'):
            st.markdown("### 🎯 Job Match Breakdown")
            job_match_chart = create_job_match_breakdown(results['job_match_detailed'])
            st.plotly_chart(job_match_chart, use_container_width=True)
    
    # Experience timeline
    if results.get('experience_detailed'):
        st.markdown("### 💼 Career Timeline")
        timeline_chart = create_experience_timeline(results['experience_detailed'])
        st.plotly_chart(timeline_chart, use_container_width=True)
    
    # Detailed insights
    st.markdown("---")
    st.markdown("## 🔍 Detailed Insights")
    
    # Create tabs for different insights
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Scores Breakdown", "🛠️ Skills Analysis", "🎯 Job Matching", "📈 Recommendations"])
    
    with tab1:
        display_scores_breakdown(results)
    
    with tab2:
        display_skills_analysis(results)
    
    with tab3:
        display_job_matching_analysis(results)
    
    with tab4:
        display_advanced_recommendations(results.get('recommendations', []))

def display_scores_breakdown(results):
    """Display detailed scores breakdown"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Score Components")
        weights = results.get('weights_used', {})
        
        score_data = {
            'Component': ['Experience', 'Education', 'Skills', 'ATS', 'Sentiment', 'Job Match'],
            'Score': [
                results.get('experience_score', 0),
                results.get('education_score', 0),
                results.get('skills_score', 0),
                results.get('ats_score', 0),
                results.get('sentiment_score', 0),
                results.get('job_match_score', 0)
            ],
            'Weight': [
                weights.get('experience', 0) * 100,
                weights.get('education', 0) * 100,
                weights.get('skills', 0) * 100,
                weights.get('ats', 0) * 100,
                weights.get('sentiment', 0) * 100,
                weights.get('job_match', 0) * 100
            ]
        }
        
        import pandas as pd
        df = pd.DataFrame(score_data)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Performance Indicators")
        
        # Performance indicators
        indicators = [
            ("Resume Length", f"{results.get('resume_length', 0)} words", 
             "✅" if 300 <= results.get('resume_length', 0) <= 800 else "⚠️"),
            ("Skills Count", f"{sum(data['count'] for data in results.get('skills_detailed', {}).values())} skills",
             "✅" if sum(data['count'] for data in results.get('skills_detailed', {}).values()) >= 10 else "⚠️"),
            ("Experience Years", f"{results.get('experience_detailed', {}).get('total_years', 0)} years",
             "✅" if results.get('experience_detailed', {}).get('total_years', 0) >= 2 else "⚠️"),
            ("Companies", f"{len(results.get('experience_detailed', {}).get('companies', []))} companies",
             "✅" if len(results.get('experience_detailed', {}).get('companies', [])) >= 2 else "⚠️")
        ]
        
        for indicator, value, status in indicators:
            st.markdown(f"**{indicator}:** {value} {status}")

def display_skills_analysis(results):
    """Display detailed skills analysis"""
    
    skills_detailed = results.get('skills_detailed', {})
    
    if skills_detailed:
        for category, data in skills_detailed.items():
            st.markdown(f"#### {category.replace('_', ' ').title()}")
            
            skills_list = []
            for skill in data['skills']:
                confidence_bar = "🟢" * int(skill['confidence'] * 5) + "⚪" * (5 - int(skill['confidence'] * 5))
                skills_list.append(f"• **{skill['name']}** - Frequency: {skill['frequency']} - Confidence: {confidence_bar}")
            
            for skill_info in skills_list:
                st.markdown(skill_info)
            
            st.markdown("---")
    else:
        st.info("No detailed skills analysis available.")

def display_job_matching_analysis(results):
    """Display job matching analysis"""
    
    job_match_data = results.get('job_match_detailed', {})
    
    if job_match_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Matched Skills")
            matched_skills = job_match_data.get('matched_skills', [])
            if matched_skills:
                for skill in matched_skills[:10]:  # Show top 10
                    st.markdown(f"• {skill}")
            else:
                st.info("No matched skills found.")
        
        with col2:
            st.markdown("#### ❌ Missing Skills")
            missing_skills = job_match_data.get('missing_skills', [])
            if missing_skills:
                for skill in missing_skills[:10]:  # Show top 10
                    st.markdown(f"• {skill}")
            else:
                st.info("No missing skills identified.")
        
        # Similarity metrics
        st.markdown("#### 📊 Similarity Metrics")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Text Similarity", f"{job_match_data.get('similarity_score', 0):.1f}%")
        with col4:
            st.metric("Keyword Overlap", f"{job_match_data.get('keyword_overlap', 0):.1f}%")
        with col5:
            skill_match_rate = (len(matched_skills) / (len(matched_skills) + len(missing_skills)) * 100) if (matched_skills or missing_skills) else 0
            st.metric("Skill Match Rate", f"{skill_match_rate:.1f}%")
    else:
        st.info("No job description provided for matching analysis.")

def main():
    """Main application function"""
    
    # Display navigation and hero section
    display_navbar()
    st.markdown('<div id="hero-section"></div>', unsafe_allow_html=True)
    get_started_button_clicked = display_hero_section()
    
    # Initialize session state
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    
    # Get analyzer and database manager
    analyzer = get_analyzer()
    db_manager = get_db_manager()
    
    # Display sidebar
    uploaded_file, job_description, analyze_button, options = display_enhanced_sidebar()
    
    # Main content area
    if st.session_state.show_history:
        st.markdown("## 📊 Analysis History")
        display_analysis_history(db_manager)
        if st.button("🔙 Back to Analysis"):
            st.session_state.show_history = False
            st.rerun()
    
    elif analyze_button and uploaded_file:
        with st.spinner("🚀 Performing comprehensive analysis..."):
            # Extract text
            resume_text = analyzer.extract_text_advanced(uploaded_file)
            
            if resume_text:
                # Perform comprehensive analysis
                results = analyzer.analyze_resume_comprehensive(resume_text, job_description)
                
                # Save to database if requested
                if options['save_history']:
                    file_hash = analyzer.calculate_file_hash(uploaded_file.getvalue())
                    db_manager.save_analysis(file_hash, uploaded_file.name, results)
                
                # Display results
                display_comprehensive_results(results, options)
                
                # Success message
                st.success("✅ Analysis completed successfully!")
                
            else:
                st.error("⚠️ Could not extract text from the uploaded file. Please try another format.")
    
    elif analyze_button and not uploaded_file:
        st.warning("⚠️ Please upload a resume file before analysis.")
    
    else:
        # Display landing sections
        st.markdown('<div id="about-section"></div>', unsafe_allow_html=True)
        display_about_section()
        st.markdown('<div id="services-section"></div>', unsafe_allow_html=True)
        display_services_section()
        st.markdown('<div id="contact-section"></div>', unsafe_allow_html=True)
        display_contact_section()

if __name__ == "__main__":
    main()
