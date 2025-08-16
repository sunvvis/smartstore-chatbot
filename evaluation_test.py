"""RAG 파이프라인 실제 평가 테스트"""

import sys

sys.path.append("src")

from src.rag_evaluator import RAGEvaluator
from src.rag import SmartStoreRAG
from src.utils import get_api_key
from openai import OpenAI


def evaluate_rag_pipeline():
    """실제 RAG 파이프라인 평가"""
    try:
        # RAG 시스템 초기화
        api_key = get_api_key()
        rag = SmartStoreRAG(api_key)
        client = OpenAI(api_key=api_key)
        evaluator = RAGEvaluator(openai_client=client)

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

        # 생성 성능 평가
        print("\n🤖 답변 생성 중...")

        # RAG 스트리밍 응답 수집
        full_answer = ""
        for chunk in rag.stream_response(question):
            if chunk["type"] == "answer_chunk":
                full_answer += chunk["content"]
            elif chunk["type"] == "answer":
                full_answer = chunk["content"]

        if full_answer:
            # LLM 자기 평가
            quality_scores = evaluator.evaluate_answer_quality(question, full_answer)

            print("\n📝 생성된 답변:")
            print(f"   {full_answer}...")

            print("\n⭐ 답변 품질 평가:")
            reasons = quality_scores.get("reasons", {})
            print(f"   관련성: {quality_scores['relevance']}/5 - {reasons.get('relevance', 'N/A')}")
            print(f"   완성도: {quality_scores['completeness']}/5 - {reasons.get('completeness', 'N/A')}")
            print(f"   정확성: {quality_scores['accuracy']}/5 - {reasons.get('accuracy', 'N/A')}")
            print(f"   전체: {quality_scores['overall']}/5")
        else:
            print("\n❌ 답변 생성 실패")

    except Exception as e:
        print(f"❌ 오류: {e}")


if __name__ == "__main__":
    evaluate_rag_pipeline()
