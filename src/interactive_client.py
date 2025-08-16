import requests
import uuid
from typing import Iterator


class InteractiveAPIClient:
    """FastAPI 서버와 통신하는 대화형 클라이언트"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.chat_url = f"{base_url}/chat"
        self.session_id = str(uuid.uuid4())  # 세션 ID 생성
        self.conversation_history = []  # 로컬 대화 기록

    def check_server_health(self) -> bool:
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def stream_chat(self, question: str, top_k: int = 3, similarity_threshold: float = 0.1) -> Iterator[str]:
        """스트리밍 채팅 요청 (실시간 스트리밍)"""
        payload = {
            "question": question,
            "session_id": self.session_id,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }

        try:
            response = requests.post(
                self.chat_url, json=payload, headers={"Content-Type": "application/json"}, stream=True, timeout=30
            )
            response.raise_for_status()

            # 실시간 스트리밍 출력
            full_response = ""
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    full_response += chunk
                    yield chunk

            # 로컬 메모리에 저장하기 위해 답변 부분 추출
            self._save_to_history(question, full_response)

        except requests.RequestException as e:
            yield f"\n❌ API 요청 오류: {e}"

    def _save_to_history(self, question: str, response_text: str):
        """응답을 로컬 히스토리에 저장"""
        lines = response_text.strip().split("\n")

        # 답변 부분만 추출 (챗봇: 이후의 내용, 후속 질문 제외)
        answer_lines = []
        for line in lines:
            if line.startswith("챗봇:"):
                content = line.replace("챗봇: ", "", 1)
                # 후속 질문(- 시작)이 아닌 실제 답변만
                if not content.strip().startswith("-") and not line.strip().endswith("?"):
                    answer_lines.append(content)

        full_answer = "\n".join(answer_lines).strip()
        if full_answer:
            self.conversation_history.append({"question": question, "answer": full_answer, "timestamp": "방금 전"})

    def get_conversation_history(self) -> list:
        """대화 기록 반환"""
        return self.conversation_history

    def clear_memory(self):
        """메모리 초기화"""
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())  # 새 세션 ID 생성

    def get_recent_context(self, num_turns: int = 2) -> str:
        """최근 대화 맥락 반환"""
        if not self.conversation_history:
            return ""

        recent_turns = self.conversation_history[-num_turns:]
        context_lines = []
        for turn in recent_turns:
            context_lines.append(f"Q: {turn['question']}")
            context_lines.append(f"A: {turn['answer'][:100]}...")

        return "\n".join(context_lines)


def main():
    """대화형 클라이언트 실행"""
    client = InteractiveAPIClient()

    print("=== 메모리 기능 포함 인터랙티브 API 클라이언트 ===")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("대화 기록을 보려면 'history'를 입력하세요.")
    print("대화 기록을 초기화하려면 'clear'를 입력하세요.\n")

    # 서버 상태 확인
    if not client.check_server_health():
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        print("서버 시작: python main.py")
        return

    print("✅ 서버 연결 성공!")
    print(f"🔑 세션 ID: {client.session_id[:8]}...")

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
            history = client.get_conversation_history()
            if not history:
                print("대화 기록이 없습니다.")
            else:
                for i, turn in enumerate(history, 1):
                    print(f"{i}. Q: {turn['question']}")
                    print(f"   A: {turn['answer'][:100]}...")
                    print(f"   시간: {turn['timestamp']}")
            continue
        elif question.lower() == "clear":
            client.clear_memory()
            print("대화 기록을 초기화했습니다.")
            print(f"🔄 새 세션: {client.session_id[:8]}...")
            conversation_count = 0
            continue
        elif not question:
            continue

        # [이전 대화 맥락]과 [응답] 라벨 제거 - 깔끔한 출력을 위해

        # 스트리밍 응답 출력
        try:
            for chunk in client.stream_chat(question):
                print(chunk, end="", flush=True)
            print()  # 마지막 줄바꿈

        except KeyboardInterrupt:
            print("\n\n⏹️  응답이 중단되었습니다.")
            continue
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            continue

        conversation_count += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 클라이언트가 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
