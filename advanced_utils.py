"""
Advanced Utilities for Enhanced Resume Analyzer
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import io
import base64
from pathlib import Path

# Import configuration
from config import (
    ENHANCED_SKILL_CATEGORIES, ATS_KEYWORDS, INDUSTRY_KEYWORDS,
    SCORING_THRESHOLDS, RECOMMENDATION_TEMPLATES, UI_CONFIG
)

class AdvancedTextProcessor:
    """Advanced text processing utilities"""
    
    @staticmethod
    def extract_contact_info(text: str) -> Dict[str, str]:
        """Extract contact information from resume text"""
        contact_info = {
            'email': '',
            'phone': '',
            'linkedin': '',
            'github': '',
            'website': ''
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Phone pattern
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, text.lower())
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group()
        
        # GitHub pattern
        github_pattern = r'github\.com/[\w-]+'
        github_match = re.search(github_pattern, text.lower())
        if github_match:
            contact_info['github'] = github_match.group()
        
        # Website pattern
        website_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        website_match = re.search(website_pattern, text)
        if website_match:
            contact_info['website'] = website_match.group()
        
        return contact_info
    
    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """Extract dates from resume text"""
        date_patterns = [
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}\s*-\s*\d{4}\b',
            r'\b\d{4}\s*to\s*\d{4}\b',
            r'\b\d{4}\s*-\s*present\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """Calculate readability score using Flesch Reading Ease"""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum([AdvancedTextProcessor._count_syllables(word) for word in text.split()])
        
        if sentences == 0 or words == 0:
            return 0
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0, min(100, score))
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)

class DataExporter:
    """Handle data export functionality"""
    
    @staticmethod
    def export_to_pdf(results: Dict, filename: str = "resume_analysis.pdf") -> bytes:
        """Export analysis results to PDF"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#4ECDC4')
            )
            story.append(Paragraph("Resume Analysis Report", title_style))
            story.append(Spacer(1, 12))
            
            # Summary scores
            summary_data = [
                ['Metric', 'Score'],
                ['Overall Score', f"{results.get('overall_score', 0):.1f}%"],
                ['Experience', f"{results.get('experience_score', 0):.1f}%"],
                ['Education', f"{results.get('education_score', 0):.1f}%"],
                ['Skills', f"{results.get('skills_score', 0):.1f}%"],
                ['ATS Compatibility', f"{results.get('ats_score', 0):.1f}%"]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4ECDC4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("Recommendations", styles['Heading2']))
            recommendations = results.get('recommendations', [])
            for rec in recommendations:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
                story.append(Spacer(1, 6))
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except ImportError:
            st.error("ReportLab not installed. Cannot generate PDF.")
            return b""
    
    @staticmethod
    def export_to_excel(results: Dict, filename: str = "resume_analysis.xlsx") -> bytes:
        """Export analysis results to Excel"""
        try:
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Metric': ['Overall Score', 'Experience', 'Education', 'Skills', 'ATS Score', 'Sentiment'],
                    'Score': [
                        results.get('overall_score', 0),
                        results.get('experience_score', 0),
                        results.get('education_score', 0),
                        results.get('skills_score', 0),
                        results.get('ats_score', 0),
                        results.get('sentiment_score', 0)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Skills sheet
                skills_data = []
                for category, data in results.get('skills_detailed', {}).items():
                    for skill in data['skills']:
                        skills_data.append({
                            'Category': category,
                            'Skill': skill['name'],
                            'Frequency': skill['frequency'],
                            'Confidence': skill['confidence']
                        })
                
                if skills_data:
                    skills_df = pd.DataFrame(skills_data)
                    skills_df.to_excel(writer, sheet_name='Skills', index=False)
                
                # Recommendations sheet
                recommendations_df = pd.DataFrame({
                    'Recommendations': results.get('recommendations', [])
                })
                recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            st.error(f"Error generating Excel file: {e}")
            return b""
    
    @staticmethod
    def create_download_link(data: bytes, filename: str, file_type: str) -> str:
        """Create download link for file"""
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/{file_type};base64,{b64}" download="{filename}">Download {filename}</a>'
        return href

class PerformanceMonitor:
    """Monitor application performance"""
    
    @staticmethod
    def measure_execution_time(func):
        """Decorator to measure function execution time"""
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            if hasattr(st, 'session_state'):
                if 'performance_metrics' not in st.session_state:
                    st.session_state.performance_metrics = {}
                st.session_state.performance_metrics[func.__name__] = execution_time
            
            return result
        return wrapper
    
    @staticmethod
    def get_performance_metrics() -> Dict[str, float]:
        """Get performance metrics from session state"""
        return getattr(st.session_state, 'performance_metrics', {})

class SecurityUtils:
    """Security utilities for file handling"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent security issues"""
        # Remove path traversal attempts
        filename = filename.replace('..', '')
        filename = filename.replace('/', '_')
        filename = filename.replace('\\', '_')
        
        # Remove special characters
        filename = re.sub(r'[<>:"|?*]', '_', filename)
        
        return filename
    
    @staticmethod
    def validate_file_type(file_type: str, allowed_types: List[str]) -> bool:
        """Validate file type against allowed types"""
        return file_type.lower() in [t.lower() for t in allowed_types]
    
    @staticmethod
    def calculate_file_hash(file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()

class CacheManager:
    """Manage application caching"""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def cache_analysis_result(file_hash: str, analysis_data: Dict) -> Dict:
        """Cache analysis results"""
        return analysis_data
    
    @staticmethod
    def clear_cache():
        """Clear all cached data"""
        st.cache_data.clear()
        st.cache_resource.clear()

class ValidationUtils:
    """Validation utilities"""
    
    @staticmethod
    def validate_resume_text(text: str) -> Tuple[bool, str]:
        """Validate resume text content"""
        if not text or len(text.strip()) < 50:
            return False, "Resume text is too short. Please provide a more detailed resume."
        
        if len(text) > 50000:
            return False, "Resume text is too long. Please provide a more concise resume."
        
        # Check for minimum required sections
        required_keywords = ['experience', 'education', 'skill']
        found_keywords = sum(1 for keyword in required_keywords if keyword in text.lower())
        
        if found_keywords < 2:
            return False, "Resume should contain at least experience, education, or skills sections."
        
        return True, "Resume text is valid."
    
    @staticmethod
    def validate_job_description(text: str) -> Tuple[bool, str]:
        """Validate job description text"""
        if not text:
            return True, "Job description is optional."
        
        if len(text.strip()) < 20:
            return False, "Job description is too short to provide meaningful analysis."
        
        if len(text) > 10000:
            return False, "Job description is too long. Please provide a more concise description."
        
        return True, "Job description is valid."

class RecommendationEngine:
    """Advanced recommendation engine"""
    
    @staticmethod
    def generate_personalized_recommendations(results: Dict, user_profile: Dict = None) -> List[str]:
        """Generate personalized recommendations based on analysis results"""
        recommendations = []
        
        # Score-based recommendations
        for metric in ['experience', 'education', 'skills']:
            score = results.get(f'{metric}_score', 0)
            
            if score < 40:
                level = 'low'
            elif score < 70:
                level = 'medium'
            else:
                level = 'high'
            
            metric_recommendations = RECOMMENDATION_TEMPLATES.get(metric, {}).get(level, [])
            recommendations.extend(metric_recommendations)
        
        # Industry-specific recommendations
        if user_profile and 'industry' in user_profile:
            industry = user_profile['industry']
            industry_keywords = INDUSTRY_KEYWORDS.get(industry, [])
            
            # Check if resume contains industry-specific keywords
            resume_text = results.get('original_text', '').lower()
            found_keywords = [kw for kw in industry_keywords if kw in resume_text]
            
            if len(found_keywords) < len(industry_keywords) * 0.3:
                recommendations.append(f"🎯 Consider adding more {industry}-specific keywords to improve relevance.")
        
        # Remove duplicates and limit
        recommendations = list(dict.fromkeys(recommendations))[:10]
        
        return recommendations
    
    @staticmethod
    def suggest_skill_improvements(skills_data: Dict, target_role: str = None) -> List[str]:
        """Suggest skill improvements based on current skills and target role"""
        suggestions = []
        
        # Analyze skill gaps
        total_skills = sum(data['count'] for data in skills_data.values())
        
        if total_skills < 10:
            suggestions.append("🛠️ Consider expanding your skill set to include more relevant technologies.")
        
        # Category-specific suggestions
        for category, data in ENHANCED_SKILL_CATEGORIES.items():
            if category not in skills_data:
                category_display = data['category_display']
                suggestions.append(f"💡 Consider learning {category_display} skills to broaden your expertise.")
        
        return suggestions[:5]

def load_enhanced_css():
    """Load enhanced CSS with advanced styling"""
    css = f"""
    <style>
    /* Enhanced CSS with modern design */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }}
    
    .metric-card {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }}
    
    .analysis-section {{
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .recommendation-card {{
        background: linear-gradient(135deg, {UI_CONFIG['primary_color']}20, {UI_CONFIG['secondary_color']}20);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid {UI_CONFIG['primary_color']};
    }}
    
    .skill-badge {{
        background: {UI_CONFIG['accent_color']};
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }}
    
    .progress-bar {{
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        overflow: hidden;
        height: 8px;
    }}
    
    .progress-fill {{
        background: linear-gradient(90deg, {UI_CONFIG['primary_color']}, {UI_CONFIG['accent_color']});
        height: 100%;
        transition: width 0.3s ease;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
