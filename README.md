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
│   ├── interactive_client.py    # 대화형 API 클라이언트
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

다음 명령어로 OpenAI API 키를 설정:

```bash
export OPENAI_API_KEY="{your_key}"
source ~/.bashrc
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

### 5. 챗봇 사용하기

```bash
# 대화형 클라이언트 실행
python -m src.interactive_client
```

**사용법:**
- 질문 입력 후 Enter로 답변 받기
- `history`: 대화 기록 조회
- `clear`: 메모리 초기화 (새 세션 시작)
- `quit` 또는 `exit`: 종료

**사용 예시:**
```
[1번째 질문] 질문을 입력하세요: 대학생도 스마트스토어 개설이 가능한가요?

유저: 대학생도 스마트스토어 개설이 가능한가요?
챗봇: 대학생도 네이버 스마트스토어를 개설할 수 있습니다. 스마트스토어 가입 후 솔루션을 이용할 수 있습니다.
챗봇:   - 내 스마트스토어를 양도하는 방법에 대해 안내해드릴까요?
챗봇:   - 여러 개의 스마트스토어를 운영할 때 정산대금 입금계좌를 다르게 설정하는 방법이 궁금하신가요?

[2번째 질문] 질문을 입력하세요: 준비해야 할 서류가 뭐가 있나요?

유저: 준비해야 할 서류가 뭐가 있나요?
챗봇: 대학생이 스마트스토어를 개설할 때 필요한 서류는 없습니다. 그러나 의무보험 가입 대상이라면 해당 보험에 따라 필요한 서류가 있을 수 있습니다. 가입 신청 후 30일 이내에 필수 서류를 제출해야 하며, 서류는 가입 신청 시 파일 업로드를 통해 제출할 수 있습니다.
챗봇:   - 가입 시 필요한 서류를 어디로 보내야 하는지 안내해드릴까요?
챗봇:   - 가입 시 서류 제출 마감일에 대해 궁금하신가요?
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

### 1. 사용하기 쉬운 대화형 인터페이스
- `python -m src.interactive_client` 한 줄로 실행
- 실시간 스트리밍 응답으로 빠른 반응성
- 직관적인 명령어 (`history`, `clear`, `quit`)

### 2. 스마트한 대화 관리
- 세션 기반 대화 메모리 유지
- 이전 대화 맥락을 고려한 개인화된 답변
- 대화 기록 조회 및 초기화 기능

### 3. 똑똑한 후속 질문 제안
- AI가 생성하는 자연스러운 후속 질문
- "~해드릴까요?", "~가 필요하신가요?" 형태의 친근한 제안
- 대화 흐름 유지로 사용자 경험 향상

### 4. 정확한 주제 필터링
- 스마트스토어 관련 질문만 처리
- 무관한 질문에 대한 정중한 안내
- 2,717개 FAQ 데이터 기반 정확한 답변

### 5. 개발자 친화적 설계
- RESTful API 제공
- 성능 평가 도구 내장
- 모듈화된 코드 구조

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
