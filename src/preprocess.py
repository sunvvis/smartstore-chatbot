import pickle
from typing import List, Dict
import re
import pandas as pd


def load_faq_data(file_path: str) -> List[Dict[str, str]]:
    """pickle 데이터 로드"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return [{"question": k, "answer": v} for k, v in data.items()]


def extract_category(question: str) -> tuple[list[str], str]:
    """질문에서 카테고리 추출"""
    categories = []
    while True:
        match = re.match(r"\[(.*?)\]", question)
        if not match:
            break
        categories.append(match.group(1))
        question = re.sub(r"^\[.*?\]", "", question).strip()
    return categories, question


def extract_related_keywords(answer: str) -> list:
    """답변에서 관련 질문 추출"""
    match = re.search(r"관련 도움말/키워드([\s\S]*?)도움말 닫기", answer)
    if match:
        return [line.strip() for line in match.group(1).split("\n") if line.strip()]
    return []


def clean_answer(answer: str) -> str:
    """답변에서 불필요 정보, 특수 문자 제거"""
    clean_text = re.split(r"위 도움말이 도움이 되었나요\?", answer)[0].strip()
    clean_text = re.sub(r"\xa0|\u200b|\ufeff", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return clean_text


def preprocess_faq(input_path: str, output_path: str) -> None:
    """통합 전처리 및 저장"""
    data = load_faq_data(input_path)
    df = pd.DataFrame(data)

    # 전처리 적용
    df[["category", "question"]] = df["question"].apply(lambda x: pd.Series(extract_category(x)))
    df["related_keywords"] = df["answer"].apply(extract_related_keywords)
    df["answer"] = df["answer"].apply(clean_answer)

    # 저장
    df.to_pickle(output_path)
    print(f"전처리 완료, 저장: {output_path}")
    print(f"정제된 FAQ 수: {len(df)}")
    print(f"샘플 데이터: {df.iloc[0].to_dict()}")
