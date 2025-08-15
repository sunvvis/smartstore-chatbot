import pytest
import os
import pandas as pd
import tempfile
from unittest.mock import Mock, patch
from src.vector_db import VectorDB, get_api_key


@pytest.fixture
def sample_data():
    """테스트용 데이터"""
    return pd.DataFrame(
        {
            "question": ["회원가입 방법", "결제 오류"],
            "answer": ["네이버 아이디로...", "고객센터 문의..."],
            "category": [["회원관리"], ["결제"]],
            "related_keywords": [["회원가입"], ["결제"]],
        }
    )


@pytest.fixture
def temp_file(sample_data):
    """임시 파일"""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        sample_data.to_pickle(f.name)
        yield f.name
    os.unlink(f.name)


def test_get_api_key():
    """API 키 테스트"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        assert get_api_key() == "test-key"

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            get_api_key()


def test_vector_db_init():
    """VectorDB 초기화"""
    with patch("src.vector_db.chromadb.PersistentClient"), patch("src.vector_db.OpenAI"):
        db = VectorDB("test-key")
        assert db.collection_name == "smartstore_faq"
        assert db.batch_size == 1000


def test_build(temp_file):
    """벡터 DB 구축"""
    with patch("src.vector_db.chromadb.PersistentClient") as mock_chroma, patch("src.vector_db.OpenAI") as mock_openai:
        # Mock 설정
        mock_collection = Mock()
        mock_chroma.return_value.create_collection.return_value = mock_collection

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # 테스트
        db = VectorDB("test-key")
        db.build(temp_file)

        # 검증
        mock_client.embeddings.create.assert_called_once()
        mock_collection.add.assert_called_once()


def test_search():
    """검색 테스트"""
    with patch("src.vector_db.chromadb.PersistentClient"), patch("src.vector_db.OpenAI") as mock_openai:
        # Mock 설정
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["회원가입 방법"]],
            "metadatas": [
                [{"answer": "네이버 아이디로...", "category": "회원관리", "related_keywords": "회원가입", "idx": 0}]
            ],
            "distances": [[0.1]],
        }

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # 테스트
        db = VectorDB("test-key")
        db.collection = mock_collection
        results = db.search("회원가입")

        # 검증
        assert len(results) == 1
        assert results[0]["question"] == "회원가입 방법"
        assert results[0]["similarity_score"] == 0.9


def test_collection_info():
    """컬렉션 정보"""
    with patch("src.vector_db.chromadb.PersistentClient"), patch("src.vector_db.OpenAI"):
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_collection.metadata = {"description": "FAQ"}

        db = VectorDB("test-key")
        db.collection = mock_collection

        info = db.get_collection_info()
        assert info["count"] == 100


def test_delete_collection():
    """컬렉션 삭제"""
    with patch("src.vector_db.chromadb.PersistentClient") as mock_chroma, patch("src.vector_db.OpenAI"):
        db = VectorDB("test-key")
        result = db.delete_collection()

        assert result is True
        mock_chroma.return_value.delete_collection.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
