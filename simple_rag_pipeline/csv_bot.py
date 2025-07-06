import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json
from datetime import datetime

# LangChain 라이브러리 임포트
from langchain_community.document_loaders import CSVLoader # CSV 파일 로드용
from langchain.text_splitter import CharacterTextSplitter # 텍스트 청킹용 (현재 미사용)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # OpenAI API 인터페이스
from langchain_community.vectorstores import FAISS # 로컬 벡터 데이터베이스
from langchain.chains import RetrievalQA # 검색+답변 생성 체인
from langchain.prompts import PromptTemplate # LLM 프롬프트 템플릿
from langchain.schema import Document # 문서 객체 클래스

# 유틸리티 라이브러리들
import tiktoken # OpenAI 토큰 계산용
from tenacity import retry, stop_after_attempt, wait_exponential # 재시도 로직용 (현재 미사용)
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # 불필요한 경고 메시지 숨김


"""
    RAG 시스템 설정을 관리하는 데이터 클래스

    모든 시스템 파라미터를 중앙 집중식으로 관리
"""
@dataclass
class Config:
    # 문서 처리 관련 설정
    chunk_size: int = 1000 # 텍스트 청킹 크기 (현재 CSV는 행별 처리로 미사용)
    chunk_overlap: int = 100 # 청크 간 겹침 크기 (현재 미사용)
    
    # LLM 관련 설정
    max_tokens: int = 150 # LLM 응답 최대 토큰 수 (비용 제어)
    model_name: str = "gpt-4o" # 사용할 LLM 모델명
    temperature: float = 0.4 # 창의성 vs 일관성 균형 (0=일관성, 1=창의성)
    
    # 검색 관련 설정
    retrieval_k: int = 4 # 검색할 상위 문서 개수
    embedding_model: str = "text-embedding-ada-002" # 임베딩 모델명
    
    # API 관련 설정
    max_retries: int = 3 # API 호출 재시도 횟수
    request_timeout: int = 30 # API 요청 타임아웃 (초)

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # 딕셔너리를 Config 객체로 변환
            return cls(**config_dict)
        except FileNotFoundError:
            # 파일이 없으면 기본 설정 사용
            logging.warning(f"설정 파일 {config_path}을 찾을 수 없습니다. 기본 설정을 사용합니다.")
            return cls()
        except json.JSONDecodeError as e:
            # JSON 형식이 잘못되면 기본 설정 사용
            logging.error(f"설정 파일의 JSON 형식이 잘못되었습니다: {e}")
            return cls()




"""
    CSV 데이터를 기반으로 한 RAG (Retrieval Augmented Generation) 파이프라인

    주요 기능:
    1. CSV 데이터를 Document 객체로 변환
    2. OpenAI 임베딩으로 벡터화
    3. FAISS 벡터스토어에 저장
    4. 자연어 질문에 대한 답변 생성
    5. 대화 기록 및 비용 추적
"""
class CSVRAGPipeline:

    """
        RAG 파이프라인 초기화
    """   
    def __init__(self, config: Config):
        # 설정 저장
        self.config = config

        # 핵심 구성요소들 초기화 (나중에 설정됨)
        self.vectorstore: Optional[FAISS] = None # FAISS 벡터 데이터베이스
        self.qa_chain: Optional[RetrievalQA] = None # 질의응답 체인
        self.embeddings: Optional[OpenAIEmbeddings] = None # 임베딩 모델
        self.documents: List[Document] = [] # 로드된 문서들

        # 비용 추적용 토큰 카운터
        self.encoding = tiktoken.encoding_for_model(self.config.model_name) # 토큰 인코더
        self.total_tokens_used = 0 # 누적 토큰 사용량

        logging.info("CSV RAG 파이프라인이 성공적으로 초기화되었습니다")


    """
        CSV 파일을 읽어서 LangChain Document 객체 리스트로 변환
        각 CSV 행을 하나의 Document로 변환하여 구조화된 텍스트 형태로 만들기 (벡터 검색과 LLM 이해도 향상)
    """
    def load_csv_data(self, csv_path: str) -> List[Document]:

        # pandas로 CSV 파일 읽기
        df = pd.read_csv(csv_path)

        documents = []

        # 각 행을 Document 객체로 변환
        for _, row in df.iterrows():
            # 구조화된 텍스트 형태로 변환 (이렇게 하면 LLM이 정보를 더 잘 이해)
            content = f"""Model: {row['model']}
                            Color: {row['color']}
                            Fuel Type: {row['fuel_type']}
                            Transmission: {row['transmission']}
                            Price: ${row['price']:,.2f}
                            Manufacture Date: {row['manufacture_date']}
                            Sale Date: {row['sale_date']}
                            State: {row['state']}
                            Mileage: {row['mileage']:,.1f} miles"""
            
            # 메타데이터 설정 (필터링이나 후처리에 활용)
            metadata = {
                "source": "tesla_motors_data",
                "model": row['model'],
                "color": row['color'],
                "state": row['state'],
                "price": row['price'],
                "mileage": row['mileage']
            }

            # Document 객체 생성
            documents.append(Document(page_content=content, metadata=metadata))

        # 문서 저장 (시스템 통계용)
        self.documents = documents
        logging.info(f"CSV 문서 {len(documents)}개 로드 완료")
        return documents
    

    """
        문서들을 임베딩 벡터로 변환
        
        주의: 이 메서드는 현재 사용되지 않음
        create_vectorstore()에서 FAISS.from_documents()를 사용하여 
        문서 로드와 벡터화를 한 번에 처리함
    """
    def create_embeddings(self, documents: List[Document]) -> Tuple[List[str], List[List[float]]]:
        # 임베딩 모델이 없으면 초기화
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)

        # 문서에서 텍스트만 추출
        texts = [doc.page_content for doc in documents]

        # OpenAI API를 통해 임베딩 생성 (이 과정에서 API 비용 발생)
        embeddings = self.embeddings.embed_documents(texts)
        
        logging.info(f"임베딩 {len(embeddings)}개 생성 완료")
        return texts, embeddings
    

    """
        문서들로부터 FAISS 벡터스토어를 생성
        
        다음 과정 수행:
        1. 문서 텍스트 추출
        2. OpenAI API로 임베딩 생성
        3. FAISS 인덱스 생성 및 저장
        4. 메타데이터 연결
    """
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        # 임베딩 모델이 없으면 초기화
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        
        # FAISS 벡터스토어 생성
        # 이 한 줄에서 다음이 모두 일어난다:
        # 1. 각 문서의 텍스트를 OpenAI API로 벡터화
        # 2. FAISS 인덱스 생성
        # 3. 벡터들을 인덱스에 추가
        # 4. 메타데이터 연결
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        logging.info("벡터스토어 생성 완료")
        return vectorstore
    
    """
        질의응답 체인 설정
        
        검색기(Retriever) + LLM + 프롬프트를 연결하여
        질문에 대한 자동 답변 시스템을 구축
    """
    def setup_qa_chain(self, vectorstore: FAISS) -> None:

        # 벡터스토어 저장
        self.vectorstore = vectorstore

        # 검색기 설정
        # search_kwargs={"k": 4}: 상위 4개 문서만 검색
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.config.retrieval_k})
        
        # 커스텀 프롬프트 템플릿 생성 (이 템플릿이 LLM의 답변 품질을 좌우)
        prompt_template = """
                다음 컨텍스트를 사용하여 질문에 답변하세요. 
                답변은 한국어로 작성하고, 정확하고 간결하게 답변해주세요.

                컨텍스트: {context}

                질문: {question}

                답변:
            """
        
        # PromptTemplate 객체 생성
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"] # 템플릿에서 사용할 변수들
        )
        
        # QA 체인 생성 (모든 구성요소를 연결)
        self.qa_chain = RetrievalQA.from_chain_type(
            # LLM 설정
            llm=ChatOpenAI(
                model=self.config.model_name, # 사용할 모델 (예: gpt-4o)
                temperature=self.config.temperature, # 창의성 vs 일관성
                max_tokens=self.config.max_tokens # 답변 길이 제한
            ),
            # 체인 유형 설정
            chain_type="stuff", # 모든 검색된 문서를 한 번에 LLM에 전달
            # 검색기 연결
            retriever=retriever,
            # 프롬프트 템플릿 연결
            chain_type_kwargs={"prompt": PROMPT},
            # 소스 문서도 함께 반환하도록 설정
            return_source_documents=True
        )
        logging.info("QA 체인 설정 완료")

    """
        사용자 질문을 처리하여 답변 생성
        
        다음 과정 수행:
        1. 질문을 벡터화
        2. 관련 문서 검색
        3. 컨텍스트와 함께 LLM에 전달
        4. 답변 생성
        5. 메타데이터 수집
    """
    def process_query(self, query: str) -> Tuple[str, List[Document], Dict[str, Any]]:
        # QA 체인이 설정되었는지 확인
        if self.qa_chain is None:
            raise RuntimeError("QA 체인이 설정되지 않았습니다.")
        
        # 처리 시간 측정 시작
        start_time = time.time()
        
        # QA 체인 실행
        # 내부적으로 다음 과정이 일어난다:
        # 1. 질문 벡터화
        # 2. 벡터스토어에서 유사 문서 검색
        # 3. 프롬프트 템플릿에 컨텍스트와 질문 삽입
        # 4. LLM API 호출
        # 5. 답변 반환
        result = self.qa_chain({"query": query})

        # 처리 시간 측정 완료
        end_time = time.time()

        # 결과 파싱
        answer = result["result"] # LLM이 생성한 답변
        source_docs = result.get("source_documents", []) # 참조한 문서들

        # 토큰 사용량 계산 (비용 추적용)
        total_tokens = len(self.encoding.encode(query + answer)) 
        self.total_tokens_used += total_tokens 

        # 메타데이터 생성
        metadata = {
            "processing_time": end_time - start_time, # 처리 시간 (초)
            "total_tokens": total_tokens, # 이번 질문의 토큰 수
            "sources_count": len(source_docs) # 참조 문서 개수
        }

        return answer, source_docs, metadata


    """
        대화 기록을 JSON 파일에 저장
        
        감사 추적(Audit Trail)과 시스템 분석을 위해
        모든 질문-답변 기록을 영구 저장
    """
    def save_conversation_history(self, query: str, answer: str, sources: List[Document], metadata: Dict[str, Any]) -> None:
        # 저장할 대화 로그 생성
        conversation_log = {
            "timestamp": datetime.now().isoformat(), # 타임스탬프
            "query": query, # 질문
            "answer": answer, # 답변
            "metadata": metadata, # 성능 메타데이터
             # 참조 문서 정보 (디버깅용)
            "sources": [
                {
                    "content": doc.page_content, # 문서 내용
                    "metadata": doc.metadata, # 문서 메타데이터
                } 
                for doc in sources
            ]
        }
        
        # 로그 파일 경로
        log_file = "conversation_history.json"

        # 기존 히스토리 로드 (파일이 있다면)
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError):
                # 파일이 손상된 경우 새로 시작
                history = []
        
        # 새 로그 추가
        history.append(conversation_log)
        
        # 파일에 저장
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)



    """
        시스템 통계 정보 반환
        
        현재 시스템 상태와 사용 통계 제공
        모니터링과 비용 추적에 활용
    """
    def get_system_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.documents), # 로드된 총 문서 수
            "total_tokens_used": self.total_tokens_used, # 누적 토큰 사용량
            "estimated_cost": self.total_tokens_used * 0.00001,  # 추정 비용 (예시 단가)
            "vectorstore_loaded": self.vectorstore is not None, # 벡터스토어 로드 상태
            "qa_chain_ready": self.qa_chain is not None # QA 체인 준비 상태
        }


    """
        저장된 벡터스토어를 로드
        
        이미 생성된 벡터스토어를 다시 사용하여
        임베딩 재생성 비용과 시간 절약
    """
    def load_saved_vectorstore(self, vectorstore_path: str = "vectorstore") -> bool:
        try:
            # 벡터스토어 파일이 존재하는지 확인
            if os.path.exists(vectorstore_path):
                # 임베딩 모델 초기화 (벡터스토어 로드에 필요)
                self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
                
                # FAISS 벡터스토어 로드
                # 이 과정에서 인덱스 파일과 메타데이터를 모두 로드
                self.vectorstore = FAISS.load_local(
                    vectorstore_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # 보안 경고 무시
                )
                logging.info("저장된 벡터스토어를 성공적으로 로드했습니다.")
                return True
            else:
                logging.info("저장된 벡터스토어를 찾을 수 없습니다.")
                return False
        except Exception as e:
            logging.error(f"벡터스토어 로드 중 오류 발생: {e}")
            return False

    """
        현재 벡터스토어를 로컬 디렉토리에 저장
        
        나중에 재사용할 수 있도록 벡터스토어를 영구 저장
        대용량 데이터셋의 경우 임베딩 재생성 비용을 크게 절약
    """
    def save_vectorstore(self, vectorstore_path: str = "vectorstore") -> bool:
        # 벡터스토어가 존재하는지 확인
        try:
            if self.vectorstore is not None:
                # FAISS 벡터스토어를 로컬에 저장 (인덱스와 메타데이터가 별도 파일로 저장된다)
                self.vectorstore.save_local(vectorstore_path)
                logging.info(f"벡터스토어를 {vectorstore_path}에 저장했습니다.")
                return True
            else:
                logging.warning("저장할 벡터스토어가 없습니다.")
                return False
        except Exception as e:
            logging.error(f"벡터스토어 저장 중 오류 발생: {e}")
            return False


    """
        샘플 설정 파일을 생성

        기본 설정값들을 포함한 config.json 파일 생성
        사용자가 이 파일을 수정하여 시스템을 커스터마이즈할 수 있다
    """
    def create_sample_config() -> None:
        config = {
            "chunk_size": 1000, # 청킹 크기
            "chunk_overlap": 100, # 청크 겹침
            "max_tokens": 150, # LLM 최대 토큰
            "model_name": "gpt-4o", # LLM 모델명
            "temperature": 0.4, # LLM 온도 설정
            "retrieval_k": 4, # 검색할 문서 수
            "embedding_model": "text-embedding-ada-002", # 임베딩 모델
            "max_retries": 3, # API 재시도 횟수
            "request_timeout": 30 # API 타임아웃
        }

        # JSON 파일로 저장
        with open("config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("샘플 설정 파일 'config.json'이 생성되었습니다.")



    """
        메인 함수 - 대화형 RAG 시스템 실행

        사용자와의 상호작용을 통해 RAG 시스템을 초기화하고
        질의응답을 수행하는 메인 루프를 실행

        주요 단계:
        1. 로깅 설정
        2. 설정 로드
        3. 파이프라인 초기화
        4. 벡터스토어 로드/생성
        5. QA 체인 설정
        6. 대화 루프 실행
    """
    def main():
        # 로깅 시스템 설정
        logging.basicConfig(
            level=logging.INFO,  # 로그 레벨
            format='%(asctime)s - %(levelname)s - %(message)s', # 로그 형식
            handlers=[
                logging.FileHandler('rag_pipeline.log', encoding='utf-8'), # 파일 로그
                logging.StreamHandler(), # 콘솔 로그
            ]
        )

        print("=== CSV RAG 파이프라인에 오신 것을 환영합니다! ===\n")

        try:
            # 1. 설정 로드
            config_path = input("설정 파일 경로를 입력하세요 (엔터: 기본 설정 사용): ").strip()
            if config_path and Path(config_path).exists():
                config = Config.from_file(config_path) # 사용자 지정 설정 파일 로드
            else:
                config = Config() # 기본 설정 사용
                if not config_path:
                    print("기본 설정을 사용합니다.")

            # 2. RAG 파이프라인 초기화
            pipeline = CSVRAGPipeline(config)

            # 3. 벡터스토어 설정 (기존 것 사용 vs 새로 생성)
            use_saved = input("저장된 벡터스토어를 사용하시겠습니까? (y/n): ").lower() == 'y'
            
            if use_saved and pipeline.load_saved_vectorstore():
                # 기존 벡터스토어 사용
                print("저장된 벡터스토어를 사용합니다.")
                pipeline.setup_qa_chain(pipeline.vectorstore)
            else:
                csv_path = "tesla_motors_data.csv"
                print("현재 작업 디렉토리:", os.getcwd())
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

                print("\nCSV 데이터 로드 및 처리 중...")
                documents = pipeline.load_csv_data(csv_path)

                print("벡터스토어 생성 중...")
                vectorstore = pipeline.create_vectorstore(documents)

                print("QA 체인 설정 중...")
                pipeline.setup_qa_chain(vectorstore)

                # 벡터스토어 저장 옵션
                save_vectorstore = input("벡터스토어를 저장하시겠습니까? (y/n): ").lower() == 'y'
                if save_vectorstore:
                    pipeline.save_vectorstore()

            print("\n✅ RAG 파이프라인 초기화 완료!")
            print("이제 CSV 데이터에 대해 질문할 수 있습니다.")
            print("종료하려면 'quit' 또는 '종료'를 입력하세요.")
            print("시스템 통계를 보려면 'stats'를 입력하세요.\n")

            # 대화 루프
            while True:
                try:
                    query = input("\n질문을 입력하세요: ").strip()

                    if query.lower() in ['quit', '종료', 'exit']:
                        print("프로그램을 종료합니다. 안녕히 가세요!")
                        break

                    if query.lower() == 'stats':
                        stats = pipeline.get_system_stats()
                        print("\n=== 시스템 통계 ===")
                        print(f"총 문서 수: {stats['total_documents']}")
                        print(f"사용된 총 토큰: {stats['total_tokens_used']}")
                        print(f"추정 비용: ${stats['estimated_cost']:.4f}")
                        print(f"벡터스토어 상태: {'✅ 로드됨' if stats['vectorstore_loaded'] else '❌ 미로드'}")
                        print(f"QA 체인 상태: {'✅ 준비됨' if stats['qa_chain_ready'] else '❌ 미준비'}")
                        continue

                    if not query:
                        print("질문을 입력해주세요.")
                        continue

                    answer, sources, metadata = pipeline.process_query(query)

                    print(f"\n💬 답변: {answer}")
                    print(f"\n📊 메타데이터:")
                    print(f"  - 처리 시간: {metadata['processing_time']:.2f}초")
                    print(f"  - 사용 토큰: {metadata['total_tokens']}")
                    print(f"  - 참조 소스: {metadata['sources_count']}개")

                    if sources:
                        print(f"\n📚 참조 소스:")
                        for i, source in enumerate(sources[:3], 1):
                            content_preview = source.page_content[:150].replace('\n', ' ')
                            print(f"  {i}. {content_preview}...")

                    pipeline.save_conversation_history(query, answer, sources, metadata)

                except KeyboardInterrupt:
                    print("\n\n프로그램이 중단되었습니다.")
                    break
                except Exception as e:
                    logging.error(f"쿼리 처리 중 오류 발생: {e}")
                    print(f"❌ 오류가 발생했습니다: {e}")
                    continue

        except Exception as e:
            logging.error(f"메인 함수 실행 중 오류: {e}")
            print(f"❌ 오류: {e}")
        finally:
            try:
                if 'pipeline' in locals():
                    stats = pipeline.get_system_stats()
                    print(f"\n=== 최종 통계 ===")
                    print(f"총 사용 토큰: {stats['total_tokens_used']}")
                    print(f"총 추정 비용: ${stats['estimated_cost']:.4f}")
            except:
                pass


    if __name__ == "__main__":
        if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
            create_sample_config()
        else:
            main()