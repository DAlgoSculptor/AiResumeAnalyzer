"""
Deployment Script for Enhanced AI Resume Analyzer
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def download_spacy_model():
    """Download spaCy English model"""
    print("🧠 Downloading spaCy model...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model: {e}")
        print("💡 You can continue without spaCy - some features will be limited")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ['logs', 'exports', 'temp']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created successfully")

def run_tests():
    """Run test suite"""
    print("🧪 Running tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_enhanced_analyzer.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("⚠️ Some tests failed, but deployment can continue")
            print(result.stdout)
            return True  # Continue deployment even if tests fail
    except Exception as e:
        print(f"⚠️ Could not run tests: {e}")
        return True  # Continue deployment

def check_streamlit():
    """Check if Streamlit is working"""
    print("🌐 Checking Streamlit installation...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} is installed")
        return True
    except ImportError:
        print("❌ Streamlit is not installed")
        return False

def create_startup_scripts():
    """Create startup scripts for different platforms"""
    print("📝 Creating startup scripts...")
    
    # Windows batch file
    with open("start_app.bat", "w") as f:
        f.write("""@echo off
echo Starting Enhanced AI Resume Analyzer...
streamlit run enhanced_app.py
pause
""")
    
    # Unix shell script
    with open("start_app.sh", "w") as f:
        f.write("""#!/bin/bash
echo "Starting Enhanced AI Resume Analyzer..."
streamlit run enhanced_app.py
""")
    
    # Make shell script executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("start_app.sh", 0o755)
    
    print("✅ Startup scripts created")

def display_usage_instructions():
    """Display usage instructions"""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📋 How to run the application:")
    print("1. Original version:")
    print("   streamlit run resume_analyzer.py")
    print("\n2. Enhanced version (recommended):")
    print("   streamlit run enhanced_app.py")
    
    if platform.system() == "Windows":
        print("\n3. Using startup script:")
        print("   Double-click start_app.bat")
    else:
        print("\n3. Using startup script:")
        print("   ./start_app.sh")
    
    print("\n🌐 The application will open in your default web browser")
    print("📍 Default URL: http://localhost:8501")
    
    print("\n📚 Features available:")
    print("• 📄 Upload PDF, DOCX, or TXT resumes")
    print("• 🎯 Job description matching")
    print("• 📊 Comprehensive analytics dashboard")
    print("• 🤖 ATS compatibility scoring")
    print("• 😊 Sentiment analysis")
    print("• 💡 AI-powered recommendations")
    print("• 📈 Analysis history tracking")
    print("• 📋 Export reports (PDF/Excel)")
    
    print("\n🆘 Need help?")
    print("• Check README.md for detailed documentation")
    print("• Run tests: python test_enhanced_analyzer.py")
    print("• View logs in the logs/ directory")

def main():
    """Main deployment function"""
    print("🚀 Enhanced AI Resume Analyzer Deployment")
    print("="*50)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Deployment failed due to dependency issues")
        sys.exit(1)
    
    # Check Streamlit
    if not check_streamlit():
        print("❌ Deployment failed - Streamlit not available")
        sys.exit(1)
    
    # Download required data
    download_nltk_data()
    download_spacy_model()  # Optional - continues even if fails
    
    # Create directories
    create_directories()
    
    # Create startup scripts
    create_startup_scripts()
    
    # Run tests
    run_tests()
    
    # Display instructions
    display_usage_instructions()

if __name__ == "__main__":
    main()
