import os


def get_api_key() -> str:
    """OpenAI API 키 조회"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수 설정 필요")
    return api_key
