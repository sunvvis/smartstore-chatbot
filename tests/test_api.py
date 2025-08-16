"""스마트스토어 챗봇 API 필수 테스트"""

import pytest
import requests


@pytest.fixture
def api_base_url() -> str:
    """API 기본 URL"""
    return "http://localhost:8000"


class TestAPI:
    """API 엔드포인트 테스트 - api.py 함수명과 매칭"""

    def test_root(self, api_base_url):
        """루트 엔드포인트 - root() 함수 테스트"""
        response = requests.get(f"{api_base_url}/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "스마트스토어 챗봇 API"
        assert data["status"] == "running"

    def test_health_check(self, api_base_url):
        """헬스 체크 - health_check() 함수 테스트"""
        response = requests.get(f"{api_base_url}/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "정상"
        assert "rag_available" in data

    def test_chat(self, api_base_url):
        """채팅 엔드포인트 - chat() 함수 테스트"""
        data = {"question": "미성년자도 판매 회원 등록이 가능한가요?"}

        with requests.post(f"{api_base_url}/chat", json=data, stream=True) as response:
            assert response.status_code == 200
            assert response.headers.get("content-type") == "text/plain; charset=utf-8"

            # 스트리밍 응답 수집
            content = ""
            for chunk in response.iter_content(decode_unicode=True):
                if chunk:
                    content += chunk
                    print(chunk, end="", flush=True)

            # 기본 형식 검증 (task.md 요구사항)
            assert "유저:" in content
            assert "챗봇:" in content
            assert data["question"] in content


if __name__ == "__main__":
    """직접 실행용"""
    print("🧪 API 핵심 기능 테스트")
    print("사용법: pytest tests/test_api.py -v")

    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("❌ pytest 설치 필요: pip install pytest")
