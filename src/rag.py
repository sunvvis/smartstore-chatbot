from typing import List, Dict, Optional, Iterator
from openai import OpenAI

from .vector_db import VectorDB
from .utils import get_api_key
from .memory import ConversationMemory


class SmartStoreRAG:
    """스트리밍 기반 RAG 시스템"""

    def __init__(
        self,
        openai_api_key: str,
        vector_db: Optional[VectorDB] = None,
        memory: Optional[ConversationMemory] = None,
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

        # 메모리 시스템 초기화
        if memory is None:
            self.memory = ConversationMemory()
        else:
            self.memory = memory

    def _create_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """당신은 네이버 스마트스토어 전문 상담사입니다.
제공된 FAQ 정보를 바탕으로 정확하고 친절하게 답변해주세요.
사용자가 이해하기 쉽도록 명확하고 구체적으로 설명해주세요."""

    def _create_user_prompt(self, question: str, sources: List[Dict], conversation_context: str = "") -> str:
        """사용자 프롬프트"""
        context = "\n\n".join([f"Q: {source['question']}\nA: {source['answer']}" for source in sources])

        prompt_parts = []

        if conversation_context:
            prompt_parts.append(f"이전 대화:\n{conversation_context}\n")

        prompt_parts.append(f"관련 FAQ:\n{context}")
        prompt_parts.append(f"\n사용자 질문: {question}")
        prompt_parts.append("\n위 FAQ와 이전 대화를 참고하여 답변해주세요.")

        return "\n\n".join(prompt_parts)

    def stream_response(self, question: str, top_k: int = 3, similarity_threshold: float = 0.1) -> Iterator[Dict]:
        """스트리밍 응답 생성"""

        # 1. 상태 전송
        yield {"type": "status", "message": "검색 중..."}

        # 2. 대화 맥락 가져오기
        conversation_context = self.memory.get_recent_context(num_turns=2)

        # 3. 벡터 검색
        search_results = self.vector_db.search(question, top_k=top_k)
        relevant_sources = [
            result for result in search_results if result.get("similarity_score", 0) >= similarity_threshold
        ]

        # 4. 주제 외 질문 처리 (임베딩 유사도 기반)
        if not relevant_sources:
            yield {
                "type": "answer",
                "content": "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.",
            }
            yield {"type": "sources", "data": search_results[:3]}
            return

        # 5. 검색 결과 표시
        yield {"type": "search_results", "data": relevant_sources}

        # 6. 스트리밍 LLM 응답
        yield {"type": "status", "message": "답변 생성 중..."}

        try:
            messages = [
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": self._create_user_prompt(question, relevant_sources, conversation_context)},
            ]

            stream = self.openai_client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, max_tokens=5000, stream=True
            )

            full_answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield {"type": "answer_chunk", "content": content}

        except Exception as e:
            print(f"스트리밍 응답 오류: {e}")
            yield {"type": "answer", "content": relevant_sources[0]["answer"]}
            full_answer = relevant_sources[0]["answer"]

        # 7. 대화 기록 저장
        self.memory.add_turn(question, full_answer, relevant_sources)

        # 8. 검색 소스 정보
        yield {"type": "sources", "data": relevant_sources}


if __name__ == "__main__":
    try:
        api_key = get_api_key()
        rag = SmartStoreRAG(api_key)

        print("=== 메모리 기능 포함 인터랙티브 RAG 테스트 ===")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print("대화 기록을 보려면 'history'를 입력하세요.")
        print("대화 기록을 초기화하려면 'clear'를 입력하세요.\n")

        conversation_count = 0

        while True:
            # 사용자 입력 받기
            question = input(f"\n[{conversation_count + 1}번째 질문] 질문을 입력하세요: ").strip()

            # 특수 명령어 처리
            if question.lower() in ["quit", "exit", "종료"]:
                print("테스트를 종료합니다.")
                break
            elif question.lower() == "history":
                print("\n=== 대화 기록 ===")
                history = rag.memory.get_conversation_history()
                if not history:
                    print("대화 기록이 없습니다.")
                else:
                    for i, turn in enumerate(history, 1):
                        print(f"{i}. Q: {turn['question']}")
                        print(f"   A: {turn['answer'][:100]}...")
                        print(f"   시간: {turn['timestamp']}")
                continue
            elif question.lower() == "clear":
                rag.memory.clear_memory()
                print("대화 기록을 초기화했습니다.")
                conversation_count = 0
                continue
            elif not question:
                continue

            # 이전 대화 맥락 표시
            context = rag.memory.get_recent_context(num_turns=2)
            if context:
                print("\n[이전 대화 맥락]")
                print(context)

            print("\n[응답]")

            # 스트리밍 응답 처리
            for chunk in rag.stream_response(question):
                chunk_type = chunk["type"]

                if chunk_type == "status":
                    print(f"[상태] {chunk['message']}")

                elif chunk_type == "answer_chunk":
                    print(chunk["content"], end="", flush=True)

                elif chunk_type == "answer":
                    print(f"[전체 답변] {chunk['content']}")

                elif chunk_type == "sources":
                    print(f"\n\n[참고 소스 {len(chunk['data'])}개]")
                    for i, source in enumerate(chunk["data"], 1):
                        similarity = source.get("similarity_score", 0)
                        print(f"  {i}. {source['question']} (유사도: {similarity:.3f})")

            conversation_count += 1

    except KeyboardInterrupt:
        print("\n\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"테스트 오류: {e}")
