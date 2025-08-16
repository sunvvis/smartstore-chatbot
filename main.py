"""FastAPI server execution script"""

import uvicorn

if __name__ == "__main__":
    print("🚀 SmartStore Chatbot API server starting...")
    print("📍 API docs: http://localhost:8000/docs")
    print("🔗 Streaming test: http://localhost:8000/chat")

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
