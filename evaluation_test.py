"""RAG 파이프라인 실제 평가 테스트"""

import sys

sys.path.append("src")

from src.rag_evaluator import RAGEvaluator
from src.rag import SmartStoreRAG
from src.utils import get_api_key


def evaluate_rag_pipeline():
    """실제 RAG 파이프라인 평가"""
    try:
        # RAG 시스템 초기화
        api_key = get_api_key()
        rag = SmartStoreRAG(api_key)
        evaluator = RAGEvaluator()

        # 질문 입력받기
        question = input("평가할 질문을 입력하세요: ").strip()
        if not question:
            question = "미성년자도 판매회원 등록이 가능한가요?"  # 기본 질문
            print(f"기본 질문 사용: {question}")

        print(f"\n🔍 질문: {question}")

        # RAG 검색 실행
        search_results = rag.vector_db.search(question, top_k=5)

        # 검색 성능 평가
        metrics = evaluator.evaluate_search_performance(search_results, similarity_threshold=0.1, top_k=5)

        print("\n📊 검색 성능 평가:")
        print(f"   사용률: {metrics['usage_ratio']:.2%}")
        print(f"   평균 유사도: {metrics['avg_similarity']:.3f}")
        print(f"   사용 문서: {metrics['used_docs']}/{metrics['total_docs']}")

        # 검색된 문서 표시
        print("\n📋 검색된 문서:")
        for i, result in enumerate(search_results, 1):
            score = result.get("similarity_score", 0)
            question_text = result.get("question", "N/A")[:50]
            print(f"   {i}. {question_text}... (유사도: {score:.3f})")

    except Exception as e:
        print(f"❌ 오류: {e}")


if __name__ == "__main__":
    evaluate_rag_pipeline()
