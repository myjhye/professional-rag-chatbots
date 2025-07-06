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

    def load_csv_data(self, csv_path: str) -> List[Document]:
        df = pd.read_csv(csv_path)

        documents = []
        for _, row in df.iterrows():
            # 모든 컬럼을 포함하여 content 생성
            content = f"""Model: {row['model']}
                            Color: {row['color']}
                            Fuel Type: {row['fuel_type']}
                            Transmission: {row['transmission']}
                            Price: ${row['price']:,.2f}
                            Manufacture Date: {row['manufacture_date']}
                            Sale Date: {row['sale_date']}
                            State: {row['state']}
                            Mileage: {row['mileage']:,.1f} miles"""
            
            metadata = {
                "source": "tesla_motors_data",
                "model": row['model'],
                "color": row['color'],
                "state": row['state'],
                "price": row['price'],
                "mileage": row['mileage']
            }
            documents.append(Document(page_content=content, metadata=metadata))

        self.documents = documents  # 문서 저장
        logging.info(f"CSV 문서 {len(documents)}개 로드 완료")
        return documents
    
    def create_embeddings(self, documents: List[Document]) -> Tuple[List[str], List[List[float]]]:
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)

        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        logging.info(f"임베딩 {len(embeddings)}개 생성 완료")
        return texts, embeddings
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """문서로부터 직접 벡터스토어 생성"""
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        
        # FAISS.from_documents를 사용하여 직접 생성
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        logging.info("벡터스토어 생성 완료")
        return vectorstore
    
    def setup_qa_chain(self, vectorstore: FAISS) -> None:
        self.vectorstore = vectorstore
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.config.retrieval_k})
        
        # 커스텀 프롬프트 템플릿 추가
        prompt_template = """다음 컨텍스트를 사용하여 질문에 답변하세요. 
        답변은 한국어로 작성하고, 정확하고 간결하게 답변해주세요.

        컨텍스트:
        {context}

        질문: {question}

        답변:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                model=self.config.model_name, 
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            ),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        logging.info("QA 체인 설정 완료")

    def process_query(self, query: str) -> Tuple[str, List[Document], Dict[str, Any]]:
        if self.qa_chain is None:
            raise RuntimeError("QA 체인이 설정되지 않았습니다.")
        
        start_time = time.time()
        result = self.qa_chain({"query": query})
        end_time = time.time()

        answer = result["result"]
        source_docs = result.get("source_documents", [])
        total_tokens = len(self.encoding.encode(query + answer))
        self.total_tokens_used += total_tokens

        metadata = {
            "processing_time": end_time - start_time,
            "total_tokens": total_tokens,
            "sources_count": len(source_docs)
        }

        return answer, source_docs, metadata

    def save_conversation_history(self, query: str, answer: str, sources: List[Document], metadata: Dict[str, Any]) -> None:
        """대화 기록을 JSON 파일에 저장"""
        conversation_log = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "metadata": metadata,
            "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in sources]
        }
        
        # 로그 파일에 추가
        log_file = "conversation_history.json"
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(conversation_log)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def get_system_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.documents),
            "total_tokens_used": self.total_tokens_used,
            "estimated_cost": self.total_tokens_used * 0.00001,  # 예시: 토큰당 가상의 비용
            "vectorstore_loaded": self.vectorstore is not None,
            "qa_chain_ready": self.qa_chain is not None
        }

    def load_saved_vectorstore(self, vectorstore_path: str = "vectorstore") -> bool:
        """저장된 벡터스토어 로드"""
        try:
            if os.path.exists(vectorstore_path):
                self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
                self.vectorstore = FAISS.load_local(vectorstore_path, self.embeddings)
                logging.info("저장된 벡터스토어를 성공적으로 로드했습니다.")
                return True
            else:
                logging.info("저장된 벡터스토어를 찾을 수 없습니다.")
                return False
        except Exception as e:
            logging.error(f"벡터스토어 로드 중 오류 발생: {e}")
            return False

    def save_vectorstore(self, vectorstore_path: str = "vectorstore") -> bool:
        """벡터스토어 저장"""
        try:
            if self.vectorstore is not None:
                self.vectorstore.save_local(vectorstore_path)
                logging.info(f"벡터스토어를 {vectorstore_path}에 저장했습니다.")
                return True
            else:
                logging.warning("저장할 벡터스토어가 없습니다.")
                return False
        except Exception as e:
            logging.error(f"벡터스토어 저장 중 오류 발생: {e}")
            return False


def create_sample_config() -> None:
    config = {
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "max_tokens": 150,
        "model_name": "gpt-4o",
        "temperature": 0.4,
        "retrieval_k": 4,
        "embedding_model": "text-embedding-ada-002",
        "max_retries": 3,
        "request_timeout": 30
    }
    with open("config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("샘플 설정 파일 'config.json'이 생성되었습니다.")


def main():
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_pipeline.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    print("=== CSV RAG 파이프라인에 오신 것을 환영합니다! ===\n")

    try:
        config_path = input("설정 파일 경로를 입력하세요 (엔터: 기본 설정 사용): ").strip()
        if config_path and Path(config_path).exists():
            config = Config.from_file(config_path)
        else:
            config = Config()
            if not config_path:
                print("기본 설정을 사용합니다.")

        pipeline = CSVRAGPipeline(config)

        use_saved = input("저장된 벡터스토어를 사용하시겠습니까? (y/n): ").lower() == 'y'
        if use_saved and pipeline.load_saved_vectorstore():
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