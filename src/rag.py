from typing import List, Dict, Optional, Iterator
from openai import OpenAI

from .vector_db import VectorDB
from .utils import get_api_key


class SmartStoreRAG:
    """스트리밍 기반 RAG 시스템"""

    def __init__(
        self,
        openai_api_key: str,
        vector_db: Optional[VectorDB] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        """RAG 시스템 초기화"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature

        # VectorDB 초기화
        if vector_db is None:
            self.vector_db = VectorDB(openai_api_key)
        else:
            self.vector_db = vector_db

    def _create_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """당신은 네이버 스마트스토어 전문 상담사입니다.
제공된 FAQ 정보를 바탕으로 정확하고 친절하게 답변해주세요.
사용자가 이해하기 쉽도록 명확하고 구체적으로 설명해주세요."""

    def _create_user_prompt(self, question: str, sources: List[Dict]) -> str:
        """사용자 프롬프트"""
        context = "\n\n".join([f"Q: {source['question']}\nA: {source['answer']}" for source in sources])

        return f"""관련 FAQ:
{context}

사용자 질문: {question}

위 FAQ를 참고하여 답변해주세요."""

    def stream_response(self, question: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Iterator[Dict]:
        """스트리밍 응답 생성"""

        # 1. 상태 전송
        yield {"type": "status", "message": "검색 중..."}

        # 2. 벡터 검색
        search_results = self.vector_db.search(question, top_k=top_k)
        relevant_sources = [
            result for result in search_results if result.get("similarity_score", 0) >= similarity_threshold
        ]

        # 3. 컨텍스트 없을 때
        if not relevant_sources:
            yield {"type": "answer", "content": "해당 질문에 대한 정보를 찾지 못했습니다. 다시 질문해주세요."}
            yield {"type": "sources", "data": search_results}
            return

        # 4. 스트리밍 LLM 응답
        yield {"type": "status", "message": "답변 생성 중..."}

        try:
            messages = [
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": self._create_user_prompt(question, relevant_sources)},
            ]

            stream = self.openai_client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, max_tokens=5000, stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield {"type": "answer_chunk", "content": content}

        except Exception as e:
            print(f"스트리밍 응답 오류: {e}")
            yield {"type": "answer", "content": relevant_sources[0]["answer"]}

        # 5. 검색 소스 정보
        yield {"type": "sources", "data": relevant_sources}


if __name__ == "__main__":
    try:
        api_key = get_api_key()
        rag = SmartStoreRAG(api_key)

        print("=== 스트리밍 RAG 테스트 ===")
        question = "미성년자도 판매 회원 등록이 가능한가요?"
        print(f"질문: {question}\n")

        search_completed = False

        for chunk in rag.stream_response(question):
            chunk_type = chunk["type"]

            if chunk_type == "status":
                print(f"[상태] {chunk['message']}")

                # 검색 완료 후 검색된 FAQ 목록 출력
                if chunk["message"] == "검색 중..." and not search_completed:
                    # 검색 결과 직접 조회
                    search_results = rag.vector_db.search(question, top_k=5)
                    print(f"검색된 FAQ {len(search_results)}개:")
                    for i, result in enumerate(search_results, 1):
                        similarity = result.get("similarity_score", 0)
                        print(f"  {i}. {result['question']}")
                        print(f"     유사도: {similarity:.3f}")
                    print()
                    search_completed = True

            elif chunk_type == "answer_chunk":
                print(chunk["content"], end="", flush=True)

            elif chunk_type == "answer":
                print(f"[답변] {chunk['content']}")

            elif chunk_type == "sources":
                print(f"\n\n참고한 FAQ 소스 {len(chunk['data'])}개:")
                for i, source in enumerate(chunk["data"], 1):
                    similarity = source.get("similarity_score", 0)
                    print(f"  {i}. Q: {source['question']}")
                    print(f"     A: {source['answer'][:100]}...")
                    print(f"     유사도: {similarity:.3f}")

    except Exception as e:
        print(f"테스트 오류: {e}")
