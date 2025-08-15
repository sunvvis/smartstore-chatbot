import pytest
from src.preprocess import load_faq_data, extract_category, extract_related_keywords, clean_answer, preprocess_faq
import pickle
from tempfile import NamedTemporaryFile
import pandas as pd


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


def test_extract_related_keywords():
    """extract_related_keywords: 관련 질문 추출"""
    sample_input = "\n관련 도움말/키워드\n\n스마트스토어 로그인ID(매니저)를 추가하거나 변경할 수 없나요?\n네이버 커머스 ID 전환 이후, 이전 아이디로 로그인이 불가한가요?\n\n\n\n도움말 닫기"
    expected_output = [
        "스마트스토어 로그인ID(매니저)를 추가하거나 변경할 수 없나요?",
        "네이버 커머스 ID 전환 이후, 이전 아이디로 로그인이 불가한가요?",
    ]
    assert extract_related_keywords(sample_input) == expected_output
    assert extract_related_keywords("답변 본문...") == []
    assert extract_related_keywords("답변\n관련 도움말/키워드\n\n도움말 닫기") == []


def test_clean_answer():
    """clean_answer: 불필요 정보 및 특수 문자 제거"""
    sample_input = "가입 서류는 가입 단계에서 업로드 가능하며,\xa0가입 신청 시 서류 준비가 되지 않은 경우\xa0가입 완료 후 [판매자정보\xa0> 심사내역 조회] 메뉴에서 업로드 가능합니다.\n위 도움말이 도움이 되었나요?\n\n\n별점1점\n\n별점2점\n\n별점3점\n\n별점4점\n\n별점5점\n\n\n\n소중한 의견을 남겨주시면 보완하도록 노력하겠습니다.\n\n보내기\n\n\n\n도움말 닫기"
    expected_output = "가입 서류는 가입 단계에서 업로드 가능하며, 가입 신청 시 서류 준비가 되지 않은 경우 가입 완료 후 [판매자정보 > 심사내역 조회] 메뉴에서 업로드 가능합니다."
    assert clean_answer(sample_input) == expected_output
    assert clean_answer("") == ""
    assert clean_answer("\xa0\u200b") == ""
    assert clean_answer("테스트\n\n위 도움말이 도움이 되었나요?") == "테스트"


def test_preprocess_faq(mock_pickle_file, tmp_path):
    """preprocess_faq: 통합 전처리 및 저장 테스트."""
    output_file = tmp_path / "cleaned_faqs.pkl"
    preprocess_faq(mock_pickle_file, str(output_file))

    # 출력 파일 확인
    assert output_file.exists(), "출력 파일 생성 안 됨"
    df = pd.read_pickle(str(output_file))
    assert len(df) == 2, "FAQ 개수 불일치"
    assert df["question"].iloc[0] == "스마트스토어센터 회원가입은 어떻게 하나요? (ID만들기)", "질문 정제 실패"
    assert (
        df["answer"].iloc[0]
        == "네이버 커머스 ID 하나로 스마트스토어센터와 같은 네이버의 다양한 커머스 서비스를 편리하게 이용하실 수 있습니다."
    ), "답변 정제 실패"
    assert df["category"].iloc[0] == ["가입절차"], "카테고리 추출 실패"
    assert df["related_keywords"].iloc[0] == [], "키워드 추출 실패"
