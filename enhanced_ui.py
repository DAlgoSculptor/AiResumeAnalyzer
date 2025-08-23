"""
Enhanced UI Components for Advanced Resume Analyzer
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

def create_advanced_dashboard(results: Dict):
    """Create comprehensive analytics dashboard"""
    
    # Main metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("Overall Score", results['overall_score'], "🎯", "#4CAF50"),
        ("Experience", results['experience_score'], "💼", "#2196F3"),
        ("Education", results['education_score'], "🎓", "#FF9800"),
        ("Skills", results['skills_score'], "🛠️", "#9C27B0"),
        ("ATS Score", results['ats_score'], "🤖", "#F44336")
    ]
    
    for i, (label, value, icon, color) in enumerate(metrics):
        with [col1, col2, col3, col4, col5][i]:
            create_metric_card(label, value, icon, color)

def create_metric_card(label: str, value: float, icon: str, color: str):
    """Create an enhanced metric card"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}20, {color}10);
        border: 1px solid {color}40;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 2rem; font-weight: bold; color: {color}; margin-bottom: 0.3rem;">
            {value:.1f}%
        </div>
        <div style="font-size: 0.9rem; color: #666; font-weight: 500;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_skills_radar_chart(skills_data: Dict) -> go.Figure:
    """Create an advanced radar chart for skills"""
    categories = []
    scores = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i, (category, data) in enumerate(skills_data.items()):
        categories.append(category.replace('_', ' ').title())
        # Calculate score based on skill count and weight
        score = min(data['count'] * data['weight'] * 20, 100)
        scores.append(score)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Skills Profile',
        line=dict(color='#4ECDC4', width=3),
        fillcolor='rgba(78, 205, 196, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.2)',
                tickfont=dict(color='white', size=10)
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.2)',
                tickfont=dict(color='white', size=12)
            )
        ),
        showlegend=True,
        title=dict(
            text="Skills Profile Analysis",
            x=0.5,
            font=dict(color='white', size=16)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_experience_timeline(experience_data: Dict) -> go.Figure:
    """Create experience timeline visualization"""
    fig = go.Figure()
    
    # Sample timeline data (in real implementation, this would be extracted from resume)
    companies = experience_data.get('companies', ['Company A', 'Company B', 'Company C'])
    years = [2, 3, 1]  # Sample years at each company
    
    if companies:
        fig.add_trace(go.Bar(
            x=companies[:5],  # Limit to 5 companies
            y=years[:5],
            marker=dict(
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(companies)],
                line=dict(color='white', width=2)
            ),
            text=[f"{y} years" for y in years[:5]],
            textposition='auto',
        ))
    
    fig.update_layout(
        title=dict(
            text="Career Timeline",
            x=0.5,
            font=dict(color='white', size=16)
        ),
        xaxis=dict(
            title="Companies",
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title="Years",
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.2)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_sentiment_gauge(sentiment_data: Dict) -> go.Figure:
    """Create sentiment analysis gauge"""
    score = sentiment_data.get('score', 0.5) * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Resume Sentiment", 'font': {'color': 'white', 'size': 16}},
        delta = {'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': 'white'},
            'bar': {'color': "#4ECDC4"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(255, 107, 107, 0.3)"},
                {'range': [50, 80], 'color': "rgba(255, 234, 167, 0.3)"},
                {'range': [80, 100], 'color': "rgba(78, 205, 196, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_ats_compatibility_chart(ats_score: float) -> go.Figure:
    """Create ATS compatibility visualization"""
    
    # Create categories for ATS analysis
    categories = ['Keywords', 'Format', 'Structure', 'Length', 'Sections']
    scores = [ats_score + np.random.uniform(-10, 10) for _ in categories]  # Sample variation
    scores = [max(0, min(100, score)) for score in scores]  # Clamp to 0-100
    
    colors = ['#FF6B6B' if score < 60 else '#FFEAA7' if score < 80 else '#4ECDC4' for score in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=scores,
            marker=dict(color=colors, line=dict(color='white', width=2)),
            text=[f"{score:.0f}%" for score in scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="ATS Compatibility Analysis",
            x=0.5,
            font=dict(color='white', size=16)
        ),
        xaxis=dict(
            title="ATS Factors",
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title="Compatibility Score (%)",
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.2)',
            range=[0, 100]
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_job_match_breakdown(job_match_data: Dict) -> go.Figure:
    """Create job match breakdown visualization"""
    
    matched_skills = job_match_data.get('matched_skills', [])
    missing_skills = job_match_data.get('missing_skills', [])
    
    labels = ['Matched Skills', 'Missing Skills']
    values = [len(matched_skills), len(missing_skills)]
    colors = ['#4ECDC4', '#FF6B6B']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textfont=dict(color='white', size=14)
    )])
    
    fig.update_layout(
        title=dict(
            text="Job Match Analysis",
            x=0.5,
            font=dict(color='white', size=16)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(font=dict(color='white'))
    )
    
    return fig

def display_advanced_recommendations(recommendations: List[str]):
    """Display enhanced recommendations with categories"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    ">
        <h3 style="color: white; text-align: center; margin-bottom: 1.5rem; font-size: 1.5rem;">
            🎯 AI-Powered Recommendations
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Group recommendations by category
    categories = {
        '💼 Experience': [r for r in recommendations if '💼' in r or '📈' in r],
        '🎓 Education': [r for r in recommendations if '🎓' in r or '📚' in r],
        '🛠️ Skills': [r for r in recommendations if '🛠️' in r or '💻' in r],
        '🤖 ATS': [r for r in recommendations if '🤖' in r or '📝' in r or '📊' in r],
        '✨ General': [r for r in recommendations if r not in sum([
            [r for r in recommendations if '💼' in r or '📈' in r],
            [r for r in recommendations if '🎓' in r or '📚' in r],
            [r for r in recommendations if '🛠️' in r or '💻' in r],
            [r for r in recommendations if '🤖' in r or '📝' in r or '📊' in r]
        ], [])]
    }
    
    for category, recs in categories.items():
        if recs:
            st.markdown(f"**{category}**")
            for rec in recs:
                st.markdown(f"• {rec}")
            st.markdown("---")

def display_analysis_history(db_manager):
    """Display analysis history with trends"""
    
    history = db_manager.get_analysis_history(limit=10)
    
    if history:
        st.markdown("### 📊 Analysis History")
        
        # Create DataFrame for visualization
        df = pd.DataFrame(history)
        df['analysis_date'] = pd.to_datetime(df['analysis_date'])
        
        # Score trends chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['analysis_date'],
            y=df['overall_score'],
            mode='lines+markers',
            name='Overall Score',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8, color='#4ECDC4')
        ))
        
        fig.update_layout(
            title="Score Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # History table
        display_df = df[['filename', 'analysis_date', 'overall_score', 'experience_score', 'education_score']].copy()
        display_df['analysis_date'] = display_df['analysis_date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = ['Filename', 'Date', 'Overall', 'Experience', 'Education']
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No analysis history available yet.")
