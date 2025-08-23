"""
Configuration file for Enhanced AI Resume Analyzer
"""

import os
from typing import Dict, List

# Application Configuration
APP_CONFIG = {
    'title': 'Enhanced AI Resume Analyzer',
    'version': '2.0.0',
    'description': 'Advanced AI-powered resume analysis with comprehensive insights',
    'author': 'AI Resume Analyzer Team',
    'debug': os.getenv('DEBUG', 'False').lower() == 'true'
}

# Database Configuration
DATABASE_CONFIG = {
    'path': 'resume_analyzer.db',
    'backup_enabled': True,
    'backup_interval_hours': 24,
    'max_history_records': 1000
}

# AI Model Configuration
AI_CONFIG = {
    'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'spacy_model': 'en_core_web_sm',
    'max_text_length': 10000,
    'confidence_threshold': 0.6,
    'enable_gpu': False
}

# Analysis Weights Configuration
ANALYSIS_WEIGHTS = {
    'with_job_description': {
        'experience': 0.25,
        'education': 0.15,
        'skills': 0.20,
        'ats': 0.15,
        'sentiment': 0.10,
        'job_match': 0.15
    },
    'without_job_description': {
        'experience': 0.30,
        'education': 0.20,
        'skills': 0.25,
        'ats': 0.15,
        'sentiment': 0.10,
        'job_match': 0.0
    }
}

# Enhanced Skill Categories with Industry Focus
ENHANCED_SKILL_CATEGORIES = {
    'programming_languages': {
        'skills': [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 
            'go', 'rust', 'kotlin', 'swift', 'scala', 'dart', 'r', 'matlab', 'perl',
            'shell', 'bash', 'powershell', 'sql', 'plsql', 'nosql'
        ],
        'weight': 0.25,
        'category_display': 'Programming Languages'
    },
    'web_technologies': {
        'skills': [
            'html', 'css', 'sass', 'less', 'react', 'angular', 'vue', 'svelte',
            'node.js', 'express', 'django', 'flask', 'spring', 'nextjs', 'nuxt',
            'gatsby', 'webpack', 'vite', 'babel', 'jquery', 'bootstrap', 'tailwind'
        ],
        'weight': 0.20,
        'category_display': 'Web Technologies'
    },
    'data_science_ml': {
        'skills': [
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
            'matplotlib', 'seaborn', 'plotly', 'jupyter', 'spark', 'hadoop',
            'tableau', 'powerbi', 'excel', 'statistics', 'machine learning',
            'deep learning', 'nlp', 'computer vision', 'data mining'
        ],
        'weight': 0.20,
        'category_display': 'Data Science & ML'
    },
    'cloud_devops': {
        'skills': [
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform',
            'ansible', 'heroku', 'vercel', 'gitlab-ci', 'github-actions', 'circleci',
            'travis-ci', 'vagrant', 'chef', 'puppet', 'nagios', 'prometheus', 'grafana'
        ],
        'weight': 0.15,
        'category_display': 'Cloud & DevOps'
    },
    'databases': {
        'skills': [
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite',
            'oracle', 'cassandra', 'dynamodb', 'neo4j', 'influxdb', 'couchdb',
            'mariadb', 'firestore', 'bigquery', 'snowflake', 'databricks'
        ],
        'weight': 0.10,
        'category_display': 'Databases'
    },
    'mobile_development': {
        'skills': [
            'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic',
            'cordova', 'swift', 'kotlin', 'objective-c', 'dart', 'unity'
        ],
        'weight': 0.15,
        'category_display': 'Mobile Development'
    },
    'soft_skills': {
        'skills': [
            'leadership', 'communication', 'teamwork', 'problem-solving', 'analytical',
            'creative', 'management', 'agile', 'scrum', 'kanban', 'project management',
            'time management', 'critical thinking', 'adaptability', 'collaboration'
        ],
        'weight': 0.10,
        'category_display': 'Soft Skills'
    },
    'cybersecurity': {
        'skills': [
            'cybersecurity', 'penetration testing', 'ethical hacking', 'firewall',
            'encryption', 'vulnerability assessment', 'incident response', 'siem',
            'compliance', 'risk assessment', 'security audit', 'malware analysis'
        ],
        'weight': 0.15,
        'category_display': 'Cybersecurity'
    }
}

# ATS Keywords and Patterns
ATS_KEYWORDS = {
    'action_verbs': [
        'achieved', 'administered', 'analyzed', 'built', 'collaborated', 'created',
        'delivered', 'developed', 'designed', 'enhanced', 'established', 'executed',
        'generated', 'implemented', 'improved', 'increased', 'led', 'managed',
        'optimized', 'organized', 'planned', 'produced', 'reduced', 'resolved',
        'streamlined', 'supervised', 'trained', 'transformed'
    ],
    'section_headers': [
        'experience', 'education', 'skills', 'summary', 'objective', 'projects',
        'certifications', 'achievements', 'awards', 'publications', 'languages'
    ],
    'quantifiable_terms': [
        'percent', '%', 'million', 'thousand', 'billion', 'revenue', 'budget',
        'team', 'users', 'customers', 'projects', 'efficiency', 'productivity'
    ]
}

# Industry-Specific Keywords
INDUSTRY_KEYWORDS = {
    'technology': [
        'software', 'development', 'programming', 'coding', 'algorithm', 'api',
        'framework', 'architecture', 'scalability', 'performance', 'optimization'
    ],
    'finance': [
        'financial', 'accounting', 'budget', 'revenue', 'profit', 'investment',
        'portfolio', 'risk', 'compliance', 'audit', 'forecasting'
    ],
    'healthcare': [
        'patient', 'clinical', 'medical', 'healthcare', 'treatment', 'diagnosis',
        'therapy', 'pharmaceutical', 'regulatory', 'compliance'
    ],
    'marketing': [
        'campaign', 'brand', 'digital marketing', 'seo', 'social media', 'content',
        'analytics', 'conversion', 'engagement', 'roi', 'lead generation'
    ],
    'sales': [
        'sales', 'revenue', 'quota', 'pipeline', 'prospect', 'client', 'customer',
        'relationship', 'negotiation', 'closing', 'territory'
    ]
}

# Scoring Thresholds
SCORING_THRESHOLDS = {
    'excellent': 90,
    'good': 75,
    'average': 60,
    'needs_improvement': 40,
    'poor': 0
}

# File Processing Configuration
FILE_CONFIG = {
    'max_file_size_mb': 10,
    'supported_formats': ['pdf', 'docx', 'txt', 'rtf'],
    'ocr_enabled': True,
    'max_pages': 10,
    'text_extraction_timeout': 30
}

# UI Configuration
UI_CONFIG = {
    'theme': 'dark',
    'primary_color': '#4ECDC4',
    'secondary_color': '#FF6B6B',
    'accent_color': '#45B7D1',
    'chart_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
    'animation_enabled': True,
    'responsive_design': True
}

# Security Configuration
SECURITY_CONFIG = {
    'file_encryption': True,
    'data_retention_days': 90,
    'anonymize_data': True,
    'secure_deletion': True,
    'audit_logging': True
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'cache_enabled': True,
    'cache_ttl_seconds': 3600,
    'max_concurrent_analyses': 5,
    'batch_processing_enabled': True,
    'async_processing': True
}

# Recommendation Templates
RECOMMENDATION_TEMPLATES = {
    'experience': {
        'low': [
            "💼 Add more quantifiable achievements with specific metrics and percentages",
            "📈 Include detailed descriptions of your responsibilities and impact",
            "🎯 Highlight leadership roles and team management experience",
            "📊 Use action verbs to describe your accomplishments"
        ],
        'medium': [
            "💼 Consider adding more recent work experience details",
            "📈 Quantify your achievements with specific numbers where possible"
        ],
        'high': [
            "💼 Your experience section is strong! Consider highlighting unique projects"
        ]
    },
    'education': {
        'low': [
            "🎓 Add relevant certifications or online courses to strengthen your profile",
            "📚 Include any professional development or training programs",
            "🏆 Mention academic achievements, honors, or relevant coursework"
        ],
        'medium': [
            "🎓 Consider adding recent certifications in your field",
            "📚 Include relevant continuing education or professional development"
        ],
        'high': [
            "🎓 Your educational background is impressive! Well done."
        ]
    },
    'skills': {
        'low': [
            "🛠️ Add more technical skills relevant to your target role",
            "💻 Include both hard and soft skills in your resume",
            "🔧 Consider learning trending technologies in your industry"
        ],
        'medium': [
            "🛠️ Your skills section is good, consider adding emerging technologies",
            "💻 Include proficiency levels for your key skills"
        ],
        'high': [
            "🛠️ Excellent skills profile! You have a strong technical foundation."
        ]
    }
}

# Export Configuration
EXPORT_CONFIG = {
    'pdf_enabled': True,
    'excel_enabled': True,
    'json_enabled': True,
    'include_charts': True,
    'include_recommendations': True,
    'watermark_enabled': False
}

# API Configuration (for future integrations)
API_CONFIG = {
    'job_boards': {
        'indeed': {'enabled': False, 'api_key': ''},
        'linkedin': {'enabled': False, 'api_key': ''},
        'glassdoor': {'enabled': False, 'api_key': ''}
    },
    'salary_data': {
        'payscale': {'enabled': False, 'api_key': ''},
        'glassdoor': {'enabled': False, 'api_key': ''}
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'file_path': 'logs/app.log',
    'max_file_size_mb': 10,
    'backup_count': 5,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}
