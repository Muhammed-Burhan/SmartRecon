#!/usr/bin/env python3
"""
Start the FastAPI server for the Bank Reconciliation System
"""

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
    import uvicorn
    local_ip = get_local_ip()
    print("🚀 Starting Bank Reconciliation API Server...")
    print("📊 Local API Documentation: http://localhost:8000/docs")
    print(f"🌍 Network API Documentation: http://{local_ip}:8000/docs")
    print("🔄 Server will auto-reload on code changes")
    print(f"🌐 API accessible from network at: http://{local_ip}:8000")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 