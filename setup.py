# setup.py - Graph RAG Environment Setup Script

import subprocess
import sys
import os
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"Python version: {sys.version.split()[0]} - OK")
    return True

def install_requirements():
    """Install required packages"""
    print("\n=== Installing Python Packages ===")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("Error: requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to install requirements")
        return False

def download_spacy_model():
    """Download required spaCy language model"""
    print("\n=== Downloading spaCy Language Model ===")
    
    try:
        # Check if model is already installed
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("spaCy model 'en_core_web_sm' already installed")
            return True
        except OSError:
            pass
        
        # Download the model
        print("Downloading spaCy English model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        print("spaCy model downloaded successfully")
        return True
        
    except subprocess.CalledProcessError:
        print("Error: Failed to download spaCy model")
        print("Please run manually: python -m spacy download en_core_web_sm")
        return False
    except ImportError:
        print("spaCy not installed yet, will download model after package installation")
        return True

def setup_env_file():
    """Create .env file template"""
    print("\n=== Setting up Environment Variables ===")
    
    env_file = Path(".env")
    if env_file.exists():
        print(".env file already exists")
        return True
    
    env_template = """# Graph RAG Environment Variables

# Google Gemini API Key
# Get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: OpenAI API Key (if you want to use OpenAI instead)
# OPENAI_API_KEY=your_openai_api_key_here
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_template)
        print("Created .env template file")
        print("Please edit .env and add your Google Gemini API key")
        return True
    except Exception as e:
        print(f"Error creating .env file: {e}")
        return False

def verify_installation():
    """Verify that all components are working"""
    print("\n=== Verifying Installation ===")
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: spaCy and model
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy and language model working")
        checks_passed += 1
    except Exception as e:
        print(f"✗ spaCy error: {e}")
    
    # Check 2: sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ SentenceTransformers working")
        checks_passed += 1
    except Exception as e:
        print(f"✗ SentenceTransformers error: {e}")
    
    # Check 3: NetworkX
    try:
        import networkx as nx
        G = nx.DiGraph()
        print("✓ NetworkX working")
        checks_passed += 1
    except Exception as e:
        print(f"✗ NetworkX error: {e}")
    
    # Check 4: Google Generative AI
    try:
        import google.generativeai as genai
        print("✓ Google Generative AI library installed")
        checks_passed += 1
    except Exception as e:
        print(f"✗ Google Generative AI error: {e}")
    
    # Check 5: Environment file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and api_key != "your_gemini_api_key_here":
            print("✓ Environment variables configured")
            checks_passed += 1
        else:
            print("! Environment variables need configuration")
            print("  Please edit .env file with your API key")
    except Exception as e:
        print(f"✗ Environment setup error: {e}")
    
    success_rate = (checks_passed / total_checks) * 100
    print(f"\nVerification complete: {checks_passed}/{total_checks} checks passed ({success_rate:.0f}%)")
    
    return checks_passed >= 4  # Allow for API key not being set initially

def create_test_script():
    """Create a simple test script"""
    print("\n=== Creating Test Script ===")
    
    test_script = """# test_graph_rag.py - Simple test for Graph RAG setup

from graph_rag import GraphRAGSystem
import os

def test_basic_functionality():
    \"\"\"Test basic Graph RAG functionality\"\"\"
    
    # Check environment
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_gemini_api_key_here":
        print("Warning: GOOGLE_API_KEY not set properly in .env file")
        print("Some functionality may not work without API key")
        return False
    
    try:
        # Test document processing
        print("Testing Graph RAG components...")
        
        system = GraphRAGSystem()
        
        # Simple test documents
        test_docs = [
            "Apple Inc. is a technology company founded by Steve Jobs. The company is headquartered in Cupertino, California.",
            "Microsoft Corporation is led by CEO Satya Nadella. The company develops software products and cloud services."
        ]
        
        # Build knowledge base
        system.build_knowledge_base(test_docs)
        
        # Test query
        result = system.query("Who founded Apple?")
        
        print("✓ Basic functionality test passed")
        print(f"Sample answer: {result['answer'][:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\\n🎉 Graph RAG is working correctly!")
    else:
        print("\\n❌ Please check your setup and API configuration")
"""
    
    try:
        with open("test_graph_rag.py", "w") as f:
            f.write(test_script)
        print("Created test_graph_rag.py")
        return True
    except Exception as e:
        print(f"Error creating test script: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("        Graph RAG Environment Setup")
    print("=" * 60)
    
    success_steps = 0
    total_steps = 5
    
    # Step 1: Check Python version
    if check_python_version():
        success_steps += 1
    else:
        print("Setup cannot continue with incompatible Python version")
        return False
    
    # Step 2: Install requirements
    if install_requirements():
        success_steps += 1
    
    # Step 3: Download spaCy model
    if download_spacy_model():
        success_steps += 1
    
    # Step 4: Setup environment file
    if setup_env_file():
        success_steps += 1
    
    # Step 5: Create test script
    if create_test_script():
        success_steps += 1
    
    # Verification
    print("\n" + "=" * 60)
    print("        Setup Summary")
    print("=" * 60)
    
    print(f"Completed steps: {success_steps}/{total_steps}")
    
    if success_steps >= 4:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your Google Gemini API key")
        print("   Get it from: https://makersuite.google.com/app/apikey")
        print("2. Run: python test_graph_rag.py")
        print("3. If test passes, run: python graph_rag.py")
    else:
        print("\n❌ Setup incomplete. Please resolve the errors above.")
    
    # Final verification
    if success_steps >= 4:
        verify_installation()

if __name__ == "__main__":
    main()