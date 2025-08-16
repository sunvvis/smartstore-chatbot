from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI


class RAGEvaluator:
    """RAG 검색 성능 평가"""

    def __init__(self, openai_client: Optional[OpenAI] = None):
        """평가자 초기화"""
        self.openai_client = openai_client

    def evaluate_search_performance(
        self, search_results: List[Dict], similarity_threshold: float = 0.1, top_k: int = None
    ) -> Dict[str, float]:
        """
        검색 성능 평가

        Args:
            search_results: VectorDB.search() 결과
            similarity_threshold: 문서 사용 기준 임계값
            top_k: 평가할 상위 k개 (None이면 전체)

        Returns:
            {
                'usage_ratio': threshold 이상 문서 비율,
                'avg_similarity': 검색된 문서 평균 유사도,
                'used_docs': 실제 사용된 문서 수,
                'total_docs': 전체 검색된 문서 수
            }
        """
        if not search_results:
            return {"usage_ratio": 0.0, "avg_similarity": 0.0, "used_docs": 0, "total_docs": 0}

        # top_k 적용
        results = search_results[:top_k] if top_k else search_results

        # 유사도 점수 추출
        similarities = [result.get("similarity_score", 0.0) for result in results]

        # threshold 이상 문서 수
        used_docs = sum(1 for sim in similarities if sim >= similarity_threshold)

        return {
            "usage_ratio": used_docs / len(results) if results else 0.0,
            "avg_similarity": np.mean(similarities) if similarities else 0.0,
            "used_docs": used_docs,
            "total_docs": len(results),
        }

    def evaluate_answer_quality(self, question: str, answer: str, model: str = "gpt-4o-mini") -> Dict[str, float]:
        """
        LLM 자기 평가로 답변 품질 측정

        Args:
            question: 사용자 질문
            answer: 생성된 답변
            model: 평가용 모델

        Returns:
            {
                'relevance': 관련성 점수 (1-5),
                'completeness': 완성도 점수 (1-5),
                'accuracy': 정확성 점수 (1-5),
                'overall': 전체 점수 (1-5)
            }
        """
        if not self.openai_client:
            return {
                "relevance": 3,
                "completeness": 3,
                "accuracy": 3,
                "overall": 3,
                "reasons": {"relevance": "N/A", "completeness": "N/A", "accuracy": "N/A"},
            }

        prompt = f"""네이버 스마트스토어 FAQ 챗봇의 답변을 평가해주세요.

질문: {question}
답변: {answer}

평가 기준 (1-5점):
1. 관련성 (Relevance):
   - 5점: 질문의 핵심을 완벽히 이해하고 스마트스토어 맞춤 답변
   - 4점: 질문 의도를 잘 파악했으나 일부 스마트스토어 특화 정보 부족
   - 3점: 일반적인 답변이지만 질문과 어느 정도 관련성 있음
   - 2점: 질문과 부분적 관련성만 있고 스마트스토어 맥락 부족
   - 1점: 질문과 전혀 관련없거나 완전히 다른 주제

2. 완성도 (Completeness):
   - 5점: 질문자가 알고 싶어하는 모든 정보를 상세히 제공
   - 4점: 핵심 정보는 제공했으나 일부 세부사항 누락
   - 3점: 기본적인 답변은 했으나 추가 설명이 필요
   - 2점: 질문에 부분적으로만 답하고 중요 정보 누락
   - 1점: 질문에 거의 답하지 못하거나 매우 불완전

3. 정확성 (Accuracy):
   - 5점: 모든 정보가 정확하고 FAQ 기반으로 신뢰할 만함
   - 4점: 대부분 정확하나 일부 표현이 애매하거나 불명확
   - 3점: 기본적으로 맞지만 세부사항에서 부정확할 가능성
   - 2점: 일부 잘못된 정보가 포함되거나 오해 소지 있음
   - 1점: 명백히 틀린 정보이거나 FAQ와 상충하는 내용

다음 형식으로 응답하세요 (점수는 1, 2, 3, 4, 5 중 하나):
관련성: [점수]/5 - [이유]
완성도: [점수]/5 - [이유]
정확성: [점수]/5 - [이유]
전체: [평균점수]/5"""

        try:
            response = self.openai_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=200
            )

            result = response.choices[0].message.content.strip()

            # 텍스트에서 점수와 이유 추출
            import re

            scores = {"relevance": 3, "completeness": 3, "accuracy": 3, "overall": 3}
            reasons = {"relevance": "N/A", "completeness": "N/A", "accuracy": "N/A"}

            # 각 항목별 패턴 매칭
            patterns = {
                "relevance": r"관련성:\s*(\d+(?:\.\d+)?)/5\s*-\s*(.+?)(?=\n|완성도:|$)",
                "completeness": r"완성도:\s*(\d+(?:\.\d+)?)/5\s*-\s*(.+?)(?=\n|정확성:|$)",
                "accuracy": r"정확성:\s*(\d+(?:\.\d+)?)/5\s*-\s*(.+?)(?=\n|전체:|$)",
                "overall": r"전체:\s*(\d+(?:\.\d+)?)/5",
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, result, re.MULTILINE | re.DOTALL)
                if match:
                    # 1-5 정수 범위로 제한
                    score = max(1, min(5, int(float(match.group(1)))))
                    scores[key] = score
                    if key != "overall" and len(match.groups()) > 1:
                        reasons[key] = match.group(2).strip()

            return {
                "relevance": scores["relevance"],
                "completeness": scores["completeness"],
                "accuracy": scores["accuracy"],
                "overall": scores["overall"],
                "reasons": reasons,
            }

        except Exception:
            return {
                "relevance": 3,
                "completeness": 3,
                "accuracy": 3,
                "overall": 3,
                "reasons": {"relevance": "N/A", "completeness": "N/A", "accuracy": "N/A"},
            }
