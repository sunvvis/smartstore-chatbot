import pytest
from datetime import datetime
from unittest.mock import patch
from src.memory import ConversationMemory


class TestConversationMemory:
    """ConversationMemory 테스트"""

    def test_init(self):
        """초기화 테스트"""
        memory = ConversationMemory(max_turns=3)
        assert memory.max_turns == 3
        assert memory.get_turn_count() == 0

        # 기본값 테스트
        default_memory = ConversationMemory()
        assert default_memory.max_turns == 5

    def test_add_turn(self):
        """대화 턴 추가 테스트"""
        memory = ConversationMemory()
        sources = [{"question": "FAQ1", "answer": "답변1"}]

        # 기본 추가
        memory.add_turn("질문1", "답변1")
        assert memory.get_turn_count() == 1

        # 소스 포함 추가
        memory.add_turn("질문2", "답변2", sources)
        turns = memory.get_conversation_history()
        assert turns[1]["sources"] == sources
        assert "timestamp" in turns[0]

    def test_max_turns_limit(self):
        """최대 턴 수 제한 테스트"""
        memory = ConversationMemory(max_turns=2)

        # 3개 턴 추가 (제한은 2개)
        memory.add_turn("질문1", "답변1")
        memory.add_turn("질문2", "답변2")
        memory.add_turn("질문3", "답변3")

        assert memory.get_turn_count() == 2
        turns = memory.get_conversation_history()

        # 가장 오래된 것이 제거되고 최신 2개만 남음
        assert turns[0]["question"] == "질문2"
        assert turns[1]["question"] == "질문3"

    def test_get_recent_context(self):
        """컨텍스트 가져오기 테스트"""
        memory = ConversationMemory()

        # 빈 메모리
        assert memory.get_recent_context() == ""

        # 단일 턴
        memory.add_turn("질문1", "답변1")
        context = memory.get_recent_context(num_turns=1)
        assert "질문1" in context and "답변1" in context

        # 다중 턴
        memory.add_turn("질문2", "답변2")
        memory.add_turn("질문3", "답변3")
        context = memory.get_recent_context(num_turns=2)
        assert "질문2" in context and "질문3" in context
        assert "질문1" not in context

    def test_context_long_answer_truncation(self):
        """긴 답변 자르기 테스트"""
        memory = ConversationMemory()
        long_answer = "긴 답변 " * 20  # 100자 이상
        memory.add_turn("질문1", long_answer)

        context = memory.get_recent_context()
        assert "..." in context

    def test_clear_memory(self):
        """메모리 초기화 테스트"""
        memory = ConversationMemory()
        memory.add_turn("질문1", "답변1")

        memory.clear_memory()
        assert memory.get_turn_count() == 0
        assert memory.get_recent_context() == ""

    def test_history_copy(self):
        """대화 기록 복사본 반환 테스트"""
        memory = ConversationMemory()
        memory.add_turn("질문1", "답변1")

        history1 = memory.get_conversation_history()
        history2 = memory.get_conversation_history()
        assert history1 is not history2

    @patch("src.memory.datetime")
    def test_timestamp_format(self, mock_datetime):
        """타임스탬프 형식 테스트"""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        memory = ConversationMemory()
        memory.add_turn("질문1", "답변1")

        turns = memory.get_conversation_history()
        assert turns[0]["timestamp"] == "2024-01-01T12:00:00"

    def test_edge_cases(self):
        """엣지 케이스 테스트"""
        memory = ConversationMemory()
        memory.add_turn("질문1", "답변1")

        # 요청 턴 수가 사용 가능한 턴보다 많은 경우
        context = memory.get_recent_context(num_turns=3)
        assert "질문1" in context

        # 0개 턴 요청
        assert memory.get_recent_context(num_turns=0) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
