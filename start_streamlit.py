#!/usr/bin/env python3
"""
Start the Streamlit frontend for the Bank Reconciliation System
"""

import subprocess
import sys

if __name__ == "__main__":
    print("🌟 Starting Bank Reconciliation Streamlit App...")
    print("🌐 App will be available at: http://localhost:8501")
    print("📝 Make sure the API server is running on port 8000")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped") 