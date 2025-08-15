import pytest
from src.preprocess import load_faq_data, extract_category
import pickle
from tempfile import NamedTemporaryFile


@pytest.fixture
def mock_pickle_file() -> str:
    """임시 pickle 파일 생성"""
    data = {
        "[가입절차] 스마트스토어센터 회원가입은 어떻게 하나요? (ID만들기)": "네이버 커머스 ID 하나로 스마트스토어센터와 같은 네이버의 다양한 커머스 서비스를 편리하게 이용하실 수 있습니다.",
        "[가입서류] 스마트스토어 판매자 유형별 필요한 서류가 어떻게 되나요?": "스마트스토어 판매 회원 분류는 아래와 같으며 반드시 모든 서류를 제출해 주셔야 가입이 가능합니다.",
    }
    with NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(data, tmp)
    return tmp.name


def test_load_faq_data(mock_pickle_file: str):
    """정상 파일 로드 테스트"""
    result = load_faq_data(mock_pickle_file)
    assert len(result) == 2, "FAQ 개수 불일치"
    assert result[0]["question"] == "[가입절차] 스마트스토어센터 회원가입은 어떻게 하나요? (ID만들기)", "질문 키 불일치"
    assert (
        result[0]["answer"]
        == "네이버 커머스 ID 하나로 스마트스토어센터와 같은 네이버의 다양한 커머스 서비스를 편리하게 이용하실 수 있습니다."
    ), "답변 값 불일치"


def test_extract_category():
    """extract_category: 카테고리와 정제된 질문 추출"""
    sample_input = "[가입절차][쇼핑윈도/패션타운] 네이버 쇼핑윈도 노출 절차는 어떻게 되나요?"
    expected_output = (["가입절차", "쇼핑윈도/패션타운"], "네이버 쇼핑윈도 노출 절차는 어떻게 되나요?")
    assert extract_category(sample_input) == expected_output
    assert extract_category("[카테고리] 질문") == (["카테고리"], "질문")
    assert extract_category("질문만") == ([], "질문만")
