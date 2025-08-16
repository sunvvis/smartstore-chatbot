from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict
import uuid

from .rag import SmartStoreRAG
from .utils import get_api_key


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 3
    similarity_threshold: Optional[float] = 0.1


class ChatApp:
    def __init__(self):
        self.app = FastAPI(title="SmartStore Chatbot API", version="0.1.0")
        self._setup_routes()

        # RAG 시스템 초기화
        try:
            self.api_key = get_api_key()
            self.rag = SmartStoreRAG(self.api_key)
        except Exception as e:
            print(f"RAG 시스템 초기화 실패: {e}")
            self.rag = None
            self.api_key = None

        # 세션별 RAG 인스턴스 관리
        self.sessions: Dict[str, SmartStoreRAG] = {}

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
            if not self.api_key:
                raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")

            # 세션 ID 처리
            session_id = request.session_id or str(uuid.uuid4())

            # 세션별 RAG 인스턴스 가져오기 또는 생성
            if session_id not in self.sessions:
                self.sessions[session_id] = SmartStoreRAG(self.api_key)

            session_rag = self.sessions[session_id]

            async def generate():
                import asyncio

                # 깔끔한 출력을 위해 세션 ID 표시 제거
                yield f"유저: {request.question}\n챗봇: "
                await asyncio.sleep(0.01)

                follow_up = []

                for chunk in session_rag.stream_response(
                    question=request.question, top_k=request.top_k, similarity_threshold=request.similarity_threshold
                ):
                    # 답변 청크 즉시 스트리밍
                    if chunk["type"] == "answer_chunk":
                        yield chunk["content"]
                        await asyncio.sleep(0.01)  # 즉시 플러시
                    elif chunk["type"] == "answer":
                        yield chunk["content"]
                        await asyncio.sleep(0.01)
                    elif chunk["type"] == "follow_up_questions":
                        follow_up = chunk["data"]["questions"]

                # 후속 질문이 있다면 추가
                if follow_up:
                    for q in follow_up:
                        yield f"\n챗봇:   - {q}"
                        await asyncio.sleep(0.01)

            return StreamingResponse(
                generate(),
                media_type="text/plain; charset=utf-8",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )


chat_app = ChatApp()
app = chat_app.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
