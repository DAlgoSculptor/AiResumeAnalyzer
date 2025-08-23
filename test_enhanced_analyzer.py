"""
Test Suite for Enhanced AI Resume Analyzer
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import io

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_analyzer import AdvancedResumeAnalyzer, DatabaseManager
    from advanced_utils import AdvancedTextProcessor, ValidationUtils, SecurityUtils
    from config import ENHANCED_SKILL_CATEGORIES, ATS_KEYWORDS
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and modules are available.")

class TestAdvancedResumeAnalyzer(unittest.TestCase):
    """Test cases for AdvancedResumeAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = AdvancedResumeAnalyzer()
        self.sample_resume_text = """
        John Doe
        Software Engineer
        john.doe@email.com
        (555) 123-4567
        
        EXPERIENCE
        Senior Software Engineer at Tech Corp (2020-2023)
        - Developed Python applications using Django and Flask
        - Led a team of 5 developers
        - Improved system performance by 40%
        - Managed AWS infrastructure
        
        Software Developer at StartupXYZ (2018-2020)
        - Built React applications
        - Worked with PostgreSQL databases
        - Implemented CI/CD pipelines
        
        EDUCATION
        Bachelor of Science in Computer Science
        University of Technology (2014-2018)
        
        SKILLS
        Python, JavaScript, React, Django, AWS, PostgreSQL, Docker
        """
        
        self.sample_job_description = """
        We are looking for a Senior Python Developer with experience in:
        - Python programming
        - Django framework
        - AWS cloud services
        - Team leadership
        - Database management
        - Agile development
        """
    
    def test_preprocess_text(self):
        """Test text preprocessing functionality"""
        raw_text = "Hello World! This is a TEST with NUMBERS 123."
        processed = self.analyzer.preprocess_text(raw_text)
        
        self.assertIsInstance(processed, str)
        self.assertNotIn("!", processed)
        self.assertNotIn("123", processed)
    
    def test_extract_skills_advanced(self):
        """Test advanced skill extraction"""
        skills = self.analyzer.extract_skills_advanced(self.sample_resume_text)
        
        self.assertIsInstance(skills, dict)
        
        # Check if programming skills are detected
        if 'programming' in skills:
            skill_names = [skill['name'] for skill in skills['programming']['skills']]
            self.assertIn('python', skill_names)
        
        # Check if web development skills are detected
        if 'web_development' in skills:
            skill_names = [skill['name'] for skill in skills['web_development']['skills']]
            self.assertIn('react', skill_names)
    
    def test_calculate_ats_score(self):
        """Test ATS score calculation"""
        ats_score = self.analyzer.calculate_ats_score(self.sample_resume_text)
        
        self.assertIsInstance(ats_score, float)
        self.assertGreaterEqual(ats_score, 0)
        self.assertLessEqual(ats_score, 100)
    
    def test_calculate_experience_score_advanced(self):
        """Test advanced experience scoring"""
        score, details = self.analyzer.calculate_experience_score_advanced(self.sample_resume_text)
        
        self.assertIsInstance(score, float)
        self.assertIsInstance(details, dict)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIn('total_years', details)
        self.assertIn('companies', details)
    
    def test_calculate_education_score_advanced(self):
        """Test advanced education scoring"""
        score, details = self.analyzer.calculate_education_score_advanced(self.sample_resume_text)
        
        self.assertIsInstance(score, float)
        self.assertIsInstance(details, dict)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIn('degrees', details)
        self.assertIn('institutions', details)
    
    def test_calculate_job_match_advanced(self):
        """Test advanced job matching"""
        score, details = self.analyzer.calculate_job_match_advanced(
            self.sample_resume_text, 
            self.sample_job_description
        )
        
        self.assertIsInstance(score, float)
        self.assertIsInstance(details, dict)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIn('matched_skills', details)
        self.assertIn('missing_skills', details)
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Create mock analysis results
        mock_results = {
            'experience_score': 75,
            'education_score': 60,
            'skills_score': 80,
            'ats_score': 65,
            'sentiment_score': 70,
            'resume_length': 500,
            'skills_detailed': {
                'programming': {'count': 5, 'weight': 0.25}
            }
        }
        
        recommendations = self.analyzer.generate_recommendations(mock_results)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 8)
        
        # Check that recommendations are strings
        for rec in recommendations:
            self.assertIsInstance(rec, str)
    
    def test_analyze_resume_comprehensive(self):
        """Test comprehensive resume analysis"""
        results = self.analyzer.analyze_resume_comprehensive(
            self.sample_resume_text, 
            self.sample_job_description
        )
        
        self.assertIsInstance(results, dict)
        
        # Check required keys
        required_keys = [
            'overall_score', 'experience_score', 'education_score',
            'skills_score', 'ats_score', 'sentiment_score',
            'job_match_score', 'recommendations'
        ]
        
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check score ranges
        for score_key in ['overall_score', 'experience_score', 'education_score', 'skills_score']:
            if score_key in results:
                self.assertGreaterEqual(results[score_key], 0)
                self.assertLessEqual(results[score_key], 100)

class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager"""
    
    def setUp(self):
        """Set up test fixtures with temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_manager = DatabaseManager(self.temp_db.name)
    
    def tearDown(self):
        """Clean up temporary database"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_init_database(self):
        """Test database initialization"""
        # Database should be created and tables should exist
        self.assertTrue(os.path.exists(self.temp_db.name))
    
    def test_save_and_retrieve_analysis(self):
        """Test saving and retrieving analysis results"""
        # Mock analysis results
        mock_results = {
            'overall_score': 85.5,
            'experience_score': 80.0,
            'education_score': 90.0,
            'skills_score': 85.0,
            'ats_score': 75.0,
            'sentiment_score': 80.0,
            'recommendations': ['Test recommendation 1', 'Test recommendation 2']
        }
        
        # Save analysis
        file_hash = "test_hash_123"
        filename = "test_resume.pdf"
        self.db_manager.save_analysis(file_hash, filename, mock_results)
        
        # Retrieve analysis history
        history = self.db_manager.get_analysis_history(limit=5)
        
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        
        # Check the saved data
        saved_analysis = history[0]
        self.assertEqual(saved_analysis['file_hash'], file_hash)
        self.assertEqual(saved_analysis['filename'], filename)
        self.assertEqual(saved_analysis['overall_score'], mock_results['overall_score'])

class TestAdvancedTextProcessor(unittest.TestCase):
    """Test cases for AdvancedTextProcessor"""
    
    def test_extract_contact_info(self):
        """Test contact information extraction"""
        text = """
        John Doe
        john.doe@example.com
        (555) 123-4567
        linkedin.com/in/johndoe
        github.com/johndoe
        https://johndoe.com
        """
        
        contact_info = AdvancedTextProcessor.extract_contact_info(text)
        
        self.assertIsInstance(contact_info, dict)
        self.assertEqual(contact_info['email'], 'john.doe@example.com')
        self.assertIn('555', contact_info['phone'])
        self.assertIn('linkedin.com/in/johndoe', contact_info['linkedin'])
        self.assertIn('github.com/johndoe', contact_info['github'])
    
    def test_extract_dates(self):
        """Test date extraction"""
        text = "Worked from Jan 2020 to Dec 2022. Also worked 2018-2019."
        
        dates = AdvancedTextProcessor.extract_dates(text)
        
        self.assertIsInstance(dates, list)
        self.assertGreater(len(dates), 0)
    
    def test_calculate_readability_score(self):
        """Test readability score calculation"""
        text = "This is a simple sentence. This is another sentence."
        
        score = AdvancedTextProcessor.calculate_readability_score(text)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

class TestValidationUtils(unittest.TestCase):
    """Test cases for ValidationUtils"""
    
    def test_validate_resume_text(self):
        """Test resume text validation"""
        # Valid resume text
        valid_text = "This is a resume with experience and education sections. " * 10
        is_valid, message = ValidationUtils.validate_resume_text(valid_text)
        self.assertTrue(is_valid)
        
        # Invalid resume text (too short)
        invalid_text = "Short text"
        is_valid, message = ValidationUtils.validate_resume_text(invalid_text)
        self.assertFalse(is_valid)
        self.assertIn("too short", message)
    
    def test_validate_job_description(self):
        """Test job description validation"""
        # Valid job description
        valid_desc = "We are looking for a software engineer with Python experience."
        is_valid, message = ValidationUtils.validate_job_description(valid_desc)
        self.assertTrue(is_valid)
        
        # Empty job description (should be valid as it's optional)
        empty_desc = ""
        is_valid, message = ValidationUtils.validate_job_description(empty_desc)
        self.assertTrue(is_valid)

class TestSecurityUtils(unittest.TestCase):
    """Test cases for SecurityUtils"""
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        dangerous_filename = "../../../etc/passwd"
        safe_filename = SecurityUtils.sanitize_filename(dangerous_filename)
        
        self.assertNotIn("..", safe_filename)
        self.assertNotIn("/", safe_filename)
    
    def test_validate_file_type(self):
        """Test file type validation"""
        allowed_types = ['pdf', 'docx', 'txt']
        
        self.assertTrue(SecurityUtils.validate_file_type('pdf', allowed_types))
        self.assertTrue(SecurityUtils.validate_file_type('PDF', allowed_types))
        self.assertFalse(SecurityUtils.validate_file_type('exe', allowed_types))
    
    def test_calculate_file_hash(self):
        """Test file hash calculation"""
        test_content = b"This is test content"
        hash1 = SecurityUtils.calculate_file_hash(test_content)
        hash2 = SecurityUtils.calculate_file_hash(test_content)
        
        self.assertEqual(hash1, hash2)  # Same content should produce same hash
        self.assertEqual(len(hash1), 64)  # SHA-256 produces 64-character hex string

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAdvancedResumeAnalyzer,
        TestDatabaseManager,
        TestAdvancedTextProcessor,
        TestValidationUtils,
        TestSecurityUtils
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Running Enhanced AI Resume Analyzer Tests...")
    print("=" * 60)
    
    success = run_tests()
    
    print("=" * 60)
    if success:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed. Please check the output above.")
    
    print("\n🚀 Test suite completed.")
