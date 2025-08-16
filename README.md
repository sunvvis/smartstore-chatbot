# 📦 SmartStore Chatbot

네이버 스마트스토어의 FAQ 데이터를 기반으로 질의응답하는 RAG 기반 챗봇 시스템입니다.

## 🚀 주요 기능

- **스트리밍 API**: FastAPI 기반 실시간 응답 스트리밍
- **RAG 파이프라인**: 2,717개 FAQ 데이터 기반 검색증강생성
- **대화 메모리**: 이전 대화 맥락을 고려한 답변 생성
- **후속 질문 제안**: 답변 후 관련 질문 자동 추천
- **주제 외 질문 필터링**: 스마트스토어 관련 질문만 처리
- **성능 평가**: 검색 및 생성 품질 평가 도구 내장

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   RAG System    │    │   Vector DB     │
│   (Streaming)   │───▶│   (OpenAI)      │───▶│   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client        │    │   Memory        │    │   FAQ Data      │
│   (HTTP/curl)   │    │   System        │    │   (2,717개)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📂 프로젝트 구조

```
smartstore-chatbot/
├── src/                          # 핵심 모듈
│   ├── api.py                   # FastAPI 서버
│   ├── rag.py                   # RAG 파이프라인
│   ├── vector_db.py             # 벡터 데이터베이스
│   ├── memory.py                # 대화 메모리
│   ├── preprocess.py            # 데이터 전처리
│   ├── rag_evaluator.py         # 성능 평가
│   ├── evaluation_test.py       # 평가 테스트 실행
│   └── utils.py                 # 유틸리티
├── tests/                       # 단위 테스트
├── data/                        # 데이터 파일
│   ├── final_result.pkl         # 원본 FAQ 데이터
│   └── cleaned_result.pkl       # 전처리된 데이터
├── chroma_db/                   # ChromaDB 저장소
├── notebooks/                   # 분석 노트북
├── main.py                      # 서버 실행 파일
└── requirements.txt             # 의존성
```

## 🔧 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/sunvvis/smartstore-chatbot.git
cd smartstore-chatbot

# 가상환경 생성 및 활성화
python -m venv env
source env/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 데이터 전처리 및 벡터 DB 구축

```bash
# 1. FAQ 데이터 전처리
python -m src.preprocess

# 2. 벡터 DB 생성
python -m src.vector_db
```

### 4. 서버 실행

```bash
# FastAPI 서버 시작
python main.py
```

서버가 시작되면:
- API 문서: http://localhost:8000/docs
- 채팅 엔드포인트: http://localhost:8000/chat

## 🔍 API 사용법

### POST /chat

스트리밍 방식으로 질의응답을 처리합니다.

**요청 예시:**
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "미성년자도 판매회원 등록이 가능한가요?",
       "top_k": 3,
       "similarity_threshold": 0.1
     }'
```

**응답 예시:**
```
유저: 미성년자도 판매회원 등록이 가능한가요?
챗봇: 네이버 스마트스토어는 만 14세 미만의 개인(개인 사업자 포함) 또는 법인사업자는 입점이 불가함을 양해 부탁 드립니다.
챗봇:   - 등록에 필요한 서류 안내해드릴까요?
챗봇:   - 등록 절차는 얼마나 오래 걸리는지 안내가 필요하신가요?
```

## 🧪 테스트

### 단위 테스트 실행
```bash
# 전체 테스트 실행
python -m pytest tests/ -v

# 특정 모듈 테스트
python -m pytest tests/test_rag.py -v
```

### 성능 평가
```bash
# RAG 파이프라인 성능 평가
python -m src.evaluation_test
```

## 📊 기술 스택

### Backend
- **FastAPI**: 비동기 웹 프레임워크, 스트리밍 API
- **OpenAI API**: GPT-4o-mini 모델 사용
- **ChromaDB**: 로컬 벡터 데이터베이스
- **Python 3.12**: 메인 프로그래밍 언어

### 주요 라이브러리
- `chromadb==1.0.16`: 벡터 데이터베이스
- `fastapi==0.116.1`: 웹 프레임워크
- `openai==1.99.9`: OpenAI API 클라이언트

## 🎯 주요 특징

### 1. 스트리밍 응답
- 실시간 응답 생성으로 사용자 경험 향상
- 토큰 단위 스트리밍으로 빠른 반응성

### 2. 대화 메모리
- 최근 3턴 대화 내역 유지
- 맥락을 고려한 개인화된 답변

### 3. 후속 질문 제안
- 답변 후 관련 질문 1-3개 자동 생성
- 사용자 경험 향상 및 대화 연속성

### 4. 주제 필터링
- 스마트스토어 관련 질문만 처리
- 부적절한 질문에 대한 안내 메시지

### 5. 성능 평가
- 검색 정확도 평가 (Precision, Recall)
- 생성 품질 평가 (Relevance, Accuracy)

## TODO

- [x] repo 초기화
- [x] 가상 환경 설정, requirements.txt 생성
- [x] 데이터 로드 및 분석
- [x] 벡터 DB 구축
- [x] RAG 로직 구현 및 평가
    - [x] 구현
    - [x] 평가
- [x] 부가 시스템 구현
    - [x] 메모리 시스템 구현
    - [x] 주제 외 질문 처리
    - [x] 후속 질문 추가
- [x] API 구현: FastAPI 엔드포인트 생성 및 스트리밍 지원
- [x] 전체 테스트
- [x] 문서화

## 🔗 참고 자료

- [네이버 스마트스토어 도움말](https://help.sell.smartstore.naver.com/index.help)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [ChromaDB 공식 문서](https://docs.trychroma.com/)
- [OpenAI API 문서](https://platform.openai.com/docs)

## 👨‍💻 개발자

**Sunwoo Yu** ([@sunvvis](https://github.com/sunvvis))
