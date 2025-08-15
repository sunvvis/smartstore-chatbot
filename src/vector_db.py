import pandas as pd
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import time

from .utils import get_api_key


class VectorDB:
    """벡터 DB 관리"""

    def __init__(
        self,
        openai_api_key: str,
        collection_name: str = "smartstore_faq",
        persist_directory: str = "./chroma_db",
        distance_metric: str = "cosine",
    ):
        """VectorDB 초기화

        Args:
            openai_api_key: OpenAI API 키
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: ChromaDB 데이터 저장 경로
            distance_metric: 거리 메트릭 (cosine, l2, ip)
        """
        # OpenAI 클라이언트 초기화
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 1536  # text-embedding-3-small 차원수

        # ChromaDB 클라이언트 초기화
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # 텔레메트리 비활성화
                allow_reset=True,  # 리셋 허용
                is_persistent=True,  # 영구 저장
            ),
        )

        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.collection = None

        # 배치 처리 설정
        self.batch_size = 1000  # 배치 크기
        self.max_tokens_per_minute = 150000  # API 제한 고려

    def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """OpenAI API로 임베딩 생성"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model, input=texts, encoding_format="float"
        )
        return [emb.embedding for emb in response.data]

    def build(self, faq_data_path: str, reset_collection: bool = True) -> None:
        """FAQ 데이터로 벡터 DB 구축"""
        # 데이터 로드
        df = pd.read_pickle(faq_data_path)

        # 컬렉션 관리
        if reset_collection:
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except ValueError:
                pass

        # 컬렉션 생성
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={
                "description": "SmartStore FAQ embeddings",
                "embedding_model": self.embedding_model,
                "embedding_dimensions": self.embedding_dimensions,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            embedding_function=None,  # 수동 임베딩 사용
        )

        # 배치 단위로 임베딩 생성
        questions = df["question"].tolist()
        all_embeddings = []

        for i in range(0, len(questions), self.batch_size):
            batch_questions = questions[i : i + self.batch_size]
            batch_embeddings = self._create_embeddings_batch(batch_questions)
            all_embeddings.extend(batch_embeddings)
            print(f"배치 {i // self.batch_size + 1} 완료")
            time.sleep(0.1)  # API 레이트 리밋 고려

        # 메타데이터 준비
        metadatas = []
        documents = []
        ids = []

        for idx, row in df.iterrows():
            # ChromaDB 문서로 질문 저장
            documents.append(row["question"])

            # 메타데이터 최적화 (JSON 직렬화 가능한 형태)
            metadata = {
                "answer": str(row["answer"]),
                "category": ",".join(row["category"]) if row["category"] else "",
                "related_keywords": ",".join(row["related_keywords"]) if row["related_keywords"] else "",
                "idx": int(idx),  # 원본 인덱스 보존
            }
            metadatas.append(metadata)
            ids.append(f"faq_{idx}")

        # ChromaDB에 저장
        self.collection.add(documents=documents, embeddings=all_embeddings, metadatas=metadatas, ids=ids)
        print(f"벡터 DB 구축 완료: {len(df)}개 FAQ")

    def search(self, query: str, top_k: int = 3, include_distances: bool = True) -> List[Dict]:
        """유사한 FAQ 검색"""
        # 컬렉션 로드
        if not self.collection:
            self.collection = self.chroma_client.get_collection(self.collection_name)

        # 쿼리 임베딩 생성
        query_embeddings = self._create_embeddings_batch([query])
        query_embedding = query_embeddings[0]

        # ChromaDB 검색
        include_list = ["documents", "metadatas"]
        if include_distances:
            include_list.append("distances")

        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, include=include_list)

        # 결과 포맷팅
        search_results = []

        for i in range(len(results["documents"][0])):
            result_item = {
                "question": results["documents"][0][i],
                "answer": results["metadatas"][0][i]["answer"],
                "category": results["metadatas"][0][i]["category"].split(",")
                if results["metadatas"][0][i]["category"]
                else [],
                "related_keywords": results["metadatas"][0][i]["related_keywords"].split(",")
                if results["metadatas"][0][i]["related_keywords"]
                else [],
            }

            if include_distances and "distances" in results:
                distance = results["distances"][0][i]
                result_item["distance"] = distance
                result_item["similarity_score"] = 1 - distance

            search_results.append(result_item)

        return search_results

    def get_collection_info(self) -> Dict:
        """컬렉션 정보 반환"""
        if not self.collection:
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
            except ValueError:
                return {"error": "컬렉션을 찾을 수 없습니다"}

        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata,
            "distance_metric": self.distance_metric,
        }

    def delete_collection(self) -> bool:
        """컬렉션 삭제"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = None
            return True
        except Exception:
            return False


if __name__ == "__main__":
    try:
        api_key = get_api_key()
        db = VectorDB(api_key)

        # 벡터 DB 구축 (첫 실행시 주석 해제)
        # db.build("./data/cleaned_result.pkl")

        # 검색 테스트
        results = db.search("미성년자도 판매 회원 등록이 가능한가요?", top_k=5)
        print(f"검색 결과: {len(results)}개")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['question']}")
            print(f"   유사도: {result.get('similarity_score', 'N/A'):.3f}")

    except Exception as e:
        print(f"오류: {e}")
