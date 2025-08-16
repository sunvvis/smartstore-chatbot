import pytest
from unittest.mock import Mock, patch
from src.rag import SmartStoreRAG


@pytest.fixture
def mock_vector_db():
    """VectorDB 모킹"""
    mock_db = Mock()
    mock_db.search.return_value = [
        {"question": "테스트 질문 1", "answer": "테스트 답변 1", "similarity_score": 0.9},
        {"question": "테스트 질문 2", "answer": "테스트 답변 2", "similarity_score": 0.8},
    ]
    return mock_db


@pytest.fixture
def mock_openai_client():
    """OpenAI 클라이언트 모킹"""
    mock_client = Mock()

    # 스트리밍 응답 모킹
    mock_chunk1 = Mock()
    mock_chunk1.choices = [Mock()]
    mock_chunk1.choices[0].delta.content = "테스트 "

    mock_chunk2 = Mock()
    mock_chunk2.choices = [Mock()]
    mock_chunk2.choices[0].delta.content = "응답입니다."

    mock_chunk3 = Mock()
    mock_chunk3.choices = [Mock()]
    mock_chunk3.choices[0].delta.content = None

    mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]

    return mock_client


@pytest.fixture
def mock_memory():
    """ConversationMemory 모킹"""
    mock_mem = Mock()
    mock_mem.get_recent_context.return_value = ""
    mock_mem.add_turn.return_value = None
    return mock_mem


@pytest.fixture
def rag_system(mock_vector_db, mock_openai_client, mock_memory):
    """RAG 시스템 인스턴스"""
    with patch("src.rag.OpenAI", return_value=mock_openai_client):
        rag = SmartStoreRAG("test_api_key", vector_db=mock_vector_db, memory=mock_memory)
        return rag


class TestSmartStoreRAG:
    """SmartStoreRAG 테스트"""

    def test_init(self, mock_vector_db, mock_memory):
        """초기화 테스트"""
        with patch("src.rag.OpenAI") as mock_openai:
            rag = SmartStoreRAG("test_key", vector_db=mock_vector_db, memory=mock_memory)

            assert rag.model == "gpt-4o-mini"
            assert rag.temperature == 0.1
            assert rag.vector_db == mock_vector_db
            assert rag.memory == mock_memory
            mock_openai.assert_called_once_with(api_key="test_key")

    def test_create_system_prompt(self, rag_system):
        """시스템 프롬프트 생성 테스트"""
        prompt = rag_system._create_system_prompt()

        assert "네이버 스마트스토어" in prompt
        assert "FAQ" in prompt
        assert "상담사" in prompt

    def test_create_user_prompt(self, rag_system):
        """사용자 프롬프트 생성 테스트"""
        sources = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
        prompt = rag_system._create_user_prompt("테스트 질문", sources, "이전 대화 내용")

        assert "관련 FAQ:" in prompt
        assert "테스트 질문" in prompt
        assert "Q1" in prompt
        assert "A1" in prompt
        assert "이전 대화:" in prompt

    def test_stream_response_normal(self, rag_system):
        """정상 스트리밍 응답 테스트"""
        chunks = list(rag_system.stream_response("테스트 질문"))

        # 청크 타입 확인
        chunk_types = [chunk["type"] for chunk in chunks]
        assert "status" in chunk_types
        assert "answer_chunk" in chunk_types or "answer" in chunk_types
        assert "sources" in chunk_types

        # 벡터 검색 호출 확인
        rag_system.vector_db.search.assert_called_once()
        # 메모리 호출 확인
        rag_system.memory.get_recent_context.assert_called_once()
        rag_system.memory.add_turn.assert_called_once()

    def test_stream_response_no_sources(self, rag_system):
        """관련 소스 없을 때 테스트"""
        # 빈 검색 결과 설정
        rag_system.vector_db.search.return_value = []

        chunks = list(rag_system.stream_response("테스트 질문"))

        # 정보 부족 응답 확인
        answer_chunks = [c for c in chunks if c["type"] == "answer"]
        assert len(answer_chunks) > 0
        assert "스마트 스토어에 대한 질문" in answer_chunks[0]["content"]

    def test_stream_response_low_similarity(self, rag_system):
        """낮은 유사도일 때 테스트"""
        # 낮은 유사도 검색 결과 설정
        rag_system.vector_db.search.return_value = [
            {"question": "관련 없는 질문", "answer": "관련 없는 답변", "similarity_score": 0.05}
        ]

        chunks = list(rag_system.stream_response("테스트 질문", similarity_threshold=0.1))

        # 정보 부족 응답 확인
        answer_chunks = [c for c in chunks if c["type"] == "answer"]
        assert len(answer_chunks) > 0

    def test_no_relevant_sources(self, rag_system):
        """유사도 낮은 질문 처리 테스트"""
        # 낮은 유사도 검색 결과 설정
        rag_system.vector_db.search.return_value = [
            {"question": "무관한 질문", "answer": "무관한 답변", "similarity_score": 0.05}
        ]

        chunks = list(rag_system.stream_response("파스타 요리법", similarity_threshold=0.1))

        answer_chunks = [c for c in chunks if c["type"] == "answer"]
        assert len(answer_chunks) > 0
        assert "스마트 스토어에 대한 질문" in answer_chunks[0]["content"]

    def test_follow_up_questions_with_related_keywords(self, rag_system):
        """관련 키워드 기반 후속 질문 제안 테스트"""
        # 1위 질문에 related_keywords 설정
        rag_system.vector_db.search.return_value = [
            {
                "question": "상품 등록",
                "answer": "...",
                "similarity_score": 0.8,
                "related_keywords": ["상품 수정", "상품 삭제"],
            }
        ]

        # LLM 모킹 - 스트리밍과 일반 호출 구분
        def mock_llm_call(*args, **kwargs):
            if kwargs.get("stream"):
                # 스트리밍 응답
                chunk = Mock()
                chunk.choices = [Mock()]
                chunk.choices[0].delta.content = "답변"
                return [chunk]
            else:
                # 후속 질문 생성
                choice = Mock()
                choice.message.content = (
                    "상품 수정에 대해 더 자세히 안내해드릴까요?\n상품 삭제에 대해 더 자세히 안내해드릴까요?"
                )
                response = Mock()
                response.choices = [choice]
                return response

        rag_system.openai_client.chat.completions.create.side_effect = mock_llm_call

        chunks = list(rag_system.stream_response("상품 등록 방법"))

        follow_up = [c for c in chunks if c["type"] == "follow_up_questions"][0]
        assert len(follow_up["data"]["questions"]) > 0
        assert follow_up["data"]["source"] == "related_keywords"

    def test_follow_up_questions_similarity_based(self, rag_system):
        """유사도 기반 후속 질문 제안 테스트"""
        # related_keywords 없는 결과 설정
        rag_system.vector_db.search.return_value = [
            {"question": "주문 관리", "answer": "...", "similarity_score": 0.7},
            {"question": "배송 조회", "answer": "...", "similarity_score": 0.6},
        ]

        # LLM 모킹 - 스트리밍과 일반 호출 구분
        def mock_llm_call(*args, **kwargs):
            if kwargs.get("stream"):
                # 스트리밍 응답
                chunk = Mock()
                chunk.choices = [Mock()]
                chunk.choices[0].delta.content = "답변"
                return [chunk]
            else:
                # 후속 질문 생성
                choice = Mock()
                choice.message.content = (
                    "주문 관리에 대해 더 자세히 안내해드릴까요?\n배송 조회에 대해 더 자세히 안내해드릴까요?"
                )
                response = Mock()
                response.choices = [choice]
                return response

        rag_system.openai_client.chat.completions.create.side_effect = mock_llm_call

        chunks = list(rag_system.stream_response("주문 방법"))

        follow_up = [c for c in chunks if c["type"] == "follow_up_questions"][0]
        assert follow_up["data"]["source"] == "similarity"

    @patch("builtins.print")
    def test_stream_response_openai_error(self, mock_print, rag_system):
        """OpenAI API 오류 시 폴백 테스트"""
        # OpenAI API 오류 시뮬레이션
        rag_system.openai_client.chat.completions.create.side_effect = Exception("API 오류")

        chunks = list(rag_system.stream_response("테스트 질문"))

        # 폴백 응답 확인
        answer_chunks = [c for c in chunks if c["type"] == "answer"]
        assert len(answer_chunks) > 0
        assert answer_chunks[0]["content"] == "테스트 답변 1"  # 첫 번째 소스의 답변

        # 오류 로그 확인
        mock_print.assert_called_with("스트리밍 응답 오류: API 오류")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
