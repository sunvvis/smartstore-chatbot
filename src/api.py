from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from .rag import SmartStoreRAG
from .utils import get_api_key


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    similarity_threshold: Optional[float] = 0.1


class ChatApp:
    def __init__(self):
        self.app = FastAPI(title="SmartStore Chatbot API", version="0.1.0")
        self._setup_routes()

        # RAG 시스템 초기화
        try:
            api_key = get_api_key()
            self.rag = SmartStoreRAG(api_key)
        except Exception as e:
            print(f"RAG 시스템 초기화 실패: {e}")
            self.rag = None

    def _setup_routes(self):
        """필수 API 라우트 설정"""

        @self.app.get("/")
        async def root():
            return {"message": "스마트스토어 챗봇 API", "status": "running"}

        @self.app.get("/health")
        async def health_check():
            return {"status": "정상", "rag_available": self.rag is not None}

        @self.app.post("/chat")
        async def chat(request: ChatRequest):
            if not self.rag:
                raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")

            async def generate():
                # 초기 헤더 출력
                yield f"유저: {request.question}\n챗봇: "

                follow_up = []

                for chunk in self.rag.stream_response(
                    question=request.question, top_k=request.top_k, similarity_threshold=request.similarity_threshold
                ):
                    # 답변 청크 즉시 스트리밍
                    if chunk["type"] == "answer_chunk":
                        yield chunk["content"]
                    elif chunk["type"] == "answer":
                        yield chunk["content"]
                    elif chunk["type"] == "follow_up_questions":
                        follow_up = chunk["data"]["questions"]

                # 후속 질문이 있다면 추가
                if follow_up:
                    for q in follow_up:
                        yield f"\n챗봇:   - {q}"

            return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


chat_app = ChatApp()
app = chat_app.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
