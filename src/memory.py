from typing import List, Dict
from datetime import datetime


class ConversationMemory:
    """간단한 대화 메모리 시스템 - 단일 사용자용"""

    def __init__(self, max_turns: int = 5):
        """메모리 시스템 초기화

        Args:
            max_turns: 최대 저장할 대화 턴 수
        """
        self.turns: List[Dict] = []
        self.max_turns = max_turns

    def add_turn(self, question: str, answer: str, sources: List[Dict] = None) -> None:
        """대화 턴 추가"""
        turn = {
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat(),
        }

        self.turns.append(turn)

        # 최대 턴 수 초과 시 오래된 것부터 제거
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_conversation_history(self) -> List[Dict]:
        """전체 대화 기록 반환"""
        return self.turns.copy()

    def get_recent_context(self, num_turns: int = 3) -> str:
        """최근 대화를 요약한 컨텍스트 문자열 반환"""
        if not self.turns or num_turns <= 0:
            return ""

        recent_turns = self.turns[-num_turns:]
        context_parts = []

        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(f"대화{i} - 질문: {turn['question']}")
            context_parts.append(f"대화{i} - 답변: {turn['answer'][:100]}...")

        return "\n".join(context_parts)

    def clear_memory(self) -> None:
        """대화 기록 초기화"""
        self.turns.clear()

    def get_turn_count(self) -> int:
        """현재 저장된 대화 턴 수 반환"""
        return len(self.turns)


if __name__ == "__main__":
    # 테스트 코드
    memory = ConversationMemory(max_turns=3)

    # 테스트 대화 추가
    memory.add_turn("회원가입 방법이 궁금합니다", "회원가입은 다음과 같이 진행됩니다...")
    memory.add_turn("등록 서류는 무엇인가요?", "등록에 필요한 서류는...")
    memory.add_turn("수수료는 얼마인가요?", "수수료는 다음과 같습니다...")

    # 컨텍스트 확인
    print("=== 대화 기록 ===")
    history = memory.get_conversation_history()
    for i, turn in enumerate(history, 1):
        print(f"{i}. Q: {turn['question']}")
        print(f"   A: {turn['answer']}")

    print("\n=== 최근 컨텍스트 ===")
    context = memory.get_recent_context()
    print(context)

    print("\n=== 메모리 정보 ===")
    print(f"저장된 대화 수: {memory.get_turn_count()}")

    # 4번째 대화 추가 (가장 오래된 것 제거됨)
    memory.add_turn("배송은 언제 가능한가요?", "배송은...")
    print(f"\n4번째 대화 추가 후 대화 수: {memory.get_turn_count()}")
    print("남은 대화:")
    for i, turn in enumerate(memory.get_conversation_history(), 1):
        print(f"  {i}. {turn['question']}")
