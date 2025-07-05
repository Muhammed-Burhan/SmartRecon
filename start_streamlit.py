#!/usr/bin/env python3
"""
Start the Streamlit frontend for the Bank Reconciliation System
"""

import subprocess
import sys
import socket

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

if __name__ == "__main__":
    local_ip = get_local_ip()
    print("🌟 Starting Bank Reconciliation Streamlit App...")
    print("🌐 Local access: http://127.0.0.1:8501")
    print(f"🌍 Network access: http://{local_ip}:8501")
    print("📝 Make sure the API server is running on port 8000")
    print("📝 Share the network URL with others on your local network!")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped") 