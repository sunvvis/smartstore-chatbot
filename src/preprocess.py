import pickle
from typing import List, Dict
import re


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
