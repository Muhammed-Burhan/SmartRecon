#!/usr/bin/env python3
"""
Start the FastAPI server for the Bank Reconciliation System
"""

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Bank Reconciliation API Server...")
    print("📊 API Documentation will be available at: http://localhost:8000/docs")
    print("🔄 Server will auto-reload on code changes")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 