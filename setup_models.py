#!/usr/bin/env python3
"""
Setup script für spaCy-Modelle
Führt die erforderlichen spaCy-Modelle herunter
"""

import subprocess
import sys
import platform

def check_python_version():
    """Prüfe Python-Version und warne bei Problemen"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 14:
        print("⚠️  Warning: Python 3.14+ may have compatibility issues with some packages.")
        print("   Consider using Python 3.11 or 3.12 for better compatibility.")
        return False
    return True

def install_requirements():
    """Installiere Requirements mit Fallback"""
    print("Installing Python dependencies...")
    
    # Versuche zuerst die minimale Version
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-minimal.txt"
        ], check=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Minimal requirements failed, trying main requirements...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            return False

def install_spacy_models():
    """Installiere die erforderlichen spaCy-Modelle"""
    models = [
        "de_core_news_sm",  # Deutsch
        "en_core_web_sm"    # Englisch
    ]
    
    for model in models:
        print(f"Installing spaCy model: {model}")
        try:
            subprocess.run([
                sys.executable, "-m", "spacy", "download", model
            ], check=True)
            print(f"✓ {model} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {model}: {e}")
            return False
    
    return True

def main():
    """Hauptfunktion"""
    print("Setting up Tauri Sanitizer Engine...")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # Python-Version prüfen
    check_python_version()
    
    # Dependencies installieren
    if not install_requirements():
        print("✗ Failed to install dependencies")
        sys.exit(1)
    
    # spaCy-Modelle installieren
    if not install_spacy_models():
        print("✗ Failed to install spaCy models")
        sys.exit(1)
    
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: npm run engine:build")
    print("2. Run: npm run dev")

if __name__ == "__main__":
    main()
