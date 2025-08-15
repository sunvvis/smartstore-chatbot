import pickle
from typing import List, Dict


def load_faq_data(file_path: str) -> List[Dict[str, str]]:
    """pickle 데이터 로드"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return [{"question": k, "answer": v} for k, v in data.items()]
