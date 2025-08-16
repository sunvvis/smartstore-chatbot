from typing import List, Dict
import numpy as np


class RAGEvaluator:
    """RAG 검색 성능 평가"""

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

    def batch_evaluate(
        self, questions: List[str], vector_db, similarity_threshold: float = 0.1, top_k: int = 5
    ) -> Dict[str, float]:
        """
        여러 질문에 대한 일괄 평가

        Args:
            questions: 평가할 질문 목록
            vector_db: VectorDB 인스턴스
            similarity_threshold: 문서 사용 기준 임계값
            top_k: 검색할 상위 k개

        Returns:
            전체 평균 성능 지표
        """
        all_metrics = []

        for question in questions:
            search_results = vector_db.search(question, top_k=top_k)
            metrics = self.evaluate_search_performance(search_results, similarity_threshold, top_k)
            all_metrics.append(metrics)

        # 전체 평균 계산
        if not all_metrics:
            return {"avg_usage_ratio": 0.0, "avg_similarity": 0.0, "total_used_docs": 0, "total_docs": 0}

        return {
            "avg_usage_ratio": np.mean([m["usage_ratio"] for m in all_metrics]),
            "avg_similarity": np.mean([m["avg_similarity"] for m in all_metrics]),
            "total_used_docs": sum(m["used_docs"] for m in all_metrics),
            "total_docs": sum(m["total_docs"] for m in all_metrics),
        }
