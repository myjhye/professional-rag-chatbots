from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import time
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
import re
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """문서 전처리 및 청킹을 담당하는 클래스"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 여러 공백을 하나로 줄임
        text = re.sub(r'\s+', ' ', text)
        # 특수문자 정리 (필요에 따라 조정)
        text = re.sub(r'[^\w\s가-힣.,!?;:]', ' ', text)
        return text.strip()
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def process_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """문서들을 처리하여 청크로 분할"""
        processed_chunks = []
        
        for doc in documents:
            content = self.clean_text(doc['content'])
            chunks = self.split_text_into_chunks(content)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc.get('source', 'unknown')}_{i}"
                processed_chunks.append({
                    'id': chunk_id,
                    'content': chunk,
                    'source': doc.get('source', 'unknown'),
                    'metadata': doc.get('metadata', {}),
                    'chunk_index': i,
                    'timestamp': datetime.now().isoformat()
                })
        
        return processed_chunks

class VectorSearchEngine:
    """벡터 검색 엔진 클래스"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.pc = None
        self.index = None
        self.index_name = None
        logger.info(f"✅ SentenceTransformer 모델 로드 완료: {model_name}")
    
    def init_pinecone(self) -> None:
        """Pinecone 초기화"""
        load_dotenv()
        api_key = os.environ.get('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")
        
        self.pc = Pinecone(api_key=api_key)
        logger.info("✅ Pinecone 초기화 성공")
    
    def create_index(self, index_name: str, dimension: int, cloud: str = "aws", region: str = "us-east-1") -> None:
        """인덱스 생성 또는 연결"""
        if not self.pc:
            self.init_pinecone()
        
        existing_indexes = [idx['name'] for idx in self.pc.list_indexes()]
        
        if index_name in existing_indexes:
            logger.info(f"✅ 기존 인덱스에 연결: {index_name}")
        else:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            logger.info(f"✅ 새 인덱스 생성 완료: {index_name}")
            
            # 인덱스가 준비될 때까지 대기
            self._wait_for_index_ready(index_name)
        
        self.index = self.pc.Index(index_name)
        self.index_name = index_name
        logger.info(f"✅ 인덱스 연결 완료: {index_name}")
    
    def _wait_for_index_ready(self, index_name: str, max_wait: int = 300) -> None:
        """인덱스 준비 대기"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                index = self.pc.Index(index_name)
                # 인덱스 상태 확인
                stats = index.describe_index_stats()
                logger.info(f"⏳ 인덱스 준비 중... (경과 시간: {int(time.time() - start_time)}초)")
                logger.info(f"📊 인덱스 상태: {stats}")
                time.sleep(5)
                return  # 인덱스가 준비됨
            except Exception as e:
                logger.info(f"⏳ 인덱스 준비 대기 중... 에러: {e}")
                time.sleep(5)
        else:
            raise TimeoutError(f"인덱스 {index_name}이 {max_wait}초 내에 준비되지 않았습니다.")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """텍스트를 임베딩으로 변환"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"✅ {len(texts)}개의 텍스트를 임베딩으로 변환 완료")
        return embeddings.tolist()
    
    def insert_documents(self, documents: List[Dict[str, str]], batch_size: int = 100) -> None:
        """문서들을 벡터 DB에 삽입"""
        if not self.index:
            raise ValueError("인덱스가 초기화되지 않았습니다. create_index()를 먼저 호출하세요.")
        
        logger.info(f"🔄 {len(documents)}개 문서의 임베딩 생성 시작")
        texts = [doc['content'] for doc in documents]
        embeddings = self.embed_texts(texts)
        
        logger.info(f"📐 생성된 임베딩 정보: 개수={len(embeddings)}, 차원={len(embeddings[0]) if embeddings else 0}")
        
        # 배치로 처리
        total_inserted = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            vectors = []
            for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                vector_id = doc.get('id', f"doc_{i+j}")
                metadata = {
                    'content': doc['content'][:1000],  # Pinecone 메타데이터 크기 제한
                    'source': doc.get('source', 'unknown'),
                    'chunk_index': doc.get('chunk_index', 0),
                    'timestamp': doc.get('timestamp', datetime.now().isoformat())
                }
                # 추가 메타데이터가 있으면 포함
                if 'metadata' in doc:
                    for key, value in doc['metadata'].items():
                        if isinstance(value, (str, int, float, bool)):  # Pinecone이 지원하는 타입만
                            metadata[key] = value
                
                vectors.append((vector_id, embedding, metadata))
            
            logger.info(f"💾 배치 {i//batch_size + 1}: {len(vectors)}개 벡터 삽입 중...")
            
            try:
                upsert_response = self.index.upsert(vectors=vectors)
                logger.info(f"✅ 배치 {i//batch_size + 1} 삽입 응답: {upsert_response}")
                total_inserted += len(vectors)
            except Exception as e:
                logger.error(f"❌ 배치 {i//batch_size + 1} 삽입 실패: {e}")
                raise
            
            # 삽입 후 잠시 대기 (Pinecone 인덱싱 시간 확보)
            time.sleep(2)
        
        # 인덱싱 완료까지 추가 대기
        logger.info("⏳ 벡터 인덱싱 완료 대기 중...")
        for wait_time in [5, 10, 15]:  # 점진적으로 대기
            time.sleep(wait_time)
            stats = self.index.describe_index_stats()
            current_count = stats.get('total_vector_count', 0)
            logger.info(f"📊 대기 중... 현재 벡터 수: {current_count}")
            if current_count > 0:
                break
        
        # 최종 확인
        final_stats = self.index.describe_index_stats()
        actual_count = final_stats.get('total_vector_count', 0)
        logger.info(f"✅ 총 {len(documents)}개의 문서 삽입 요청 완료")
        logger.info(f"📊 실제 인덱스에 저장된 벡터 수: {actual_count}")
        
        if actual_count == 0:
            logger.warning("⚠️ 벡터가 인덱스에 저장되지 않았습니다. Pinecone 설정을 확인해주세요.")
            # 인덱스 정보 출력
            logger.info(f"인덱스 통계: {final_stats}")
        
        return actual_count
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict]:
        """쿼리를 통한 유사 문서 검색"""
        if not self.index:
            raise ValueError("인덱스가 초기화되지 않았습니다.")
        
        query_embedding = self.model.encode([query])[0].tolist()
        
        result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = []
        logger.info(f"🔎 원시 검색 결과: {len(result.get('matches', []))}개")
        
        for i, match in enumerate(result.get('matches', [])):
            score = match['score']
            logger.info(f"  - 매치 {i+1}: ID={match['id']}, 점수={score:.4f}")
            
            if score >= score_threshold:
                matches.append({
                    'id': match['id'],
                    'score': score,
                    'content': match['metadata'].get('content', ''),
                    'source': match['metadata'].get('source', ''),
                    'metadata': match['metadata']
                })
        
        logger.info(f"🔎 쿼리 '{query}'에 대해 {len(matches)}개의 관련 문서 발견 (임계값: {score_threshold})")
        return matches
    
    def delete_index(self) -> None:
        """인덱스 삭제"""
        if self.pc and self.index_name:
            self.pc.delete_index(self.index_name)
            logger.info(f"🗑️ 인덱스 {self.index_name} 삭제 완료")
    
    def get_index_stats(self) -> Dict:
        """인덱스 통계 정보 조회"""
        if not self.index:
            raise ValueError("인덱스가 초기화되지 않았습니다.")
        
        stats = self.index.describe_index_stats()
        return stats

class RAGSystem:
    """RAG 시스템 통합 클래스"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 500):
        self.document_processor = DocumentProcessor(chunk_size=chunk_size)
        self.search_engine = VectorSearchEngine(model_name=model_name)
        self.search_engine.init_pinecone()
    
    def load_documents_from_file(self, file_path: str) -> List[Dict[str, str]]:
        """파일에서 문서 로드"""
        documents = []
        file_path = Path(file_path)
        
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'content': content,
                    'source': str(file_path),
                    'metadata': {'file_type': 'txt'}
                })
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
        
        return documents
    
    def setup_knowledge_base(self, documents: List[Dict[str, str]], index_name: str = None) -> str:
        """지식 베이스 설정"""
        if not index_name:
            timestamp = int(time.time())
            index_name = f"rag-knowledge-base-{timestamp}"
        
        logger.info(f"📄 원본 문서 수: {len(documents)}")
        
        # 문서 전처리
        processed_docs = self.document_processor.process_documents(documents)
        logger.info(f"📄 {len(documents)}개 문서를 {len(processed_docs)}개 청크로 분할")
        
        # 처리된 문서 내용 확인
        for i, doc in enumerate(processed_docs[:3]):  # 처음 3개만 확인
            logger.info(f"청크 {i}: ID={doc['id']}, 길이={len(doc['content'])}, 내용={doc['content'][:100]}...")
        
        # 임베딩 차원 계산
        sample_embedding = self.search_engine.model.encode(["sample text"])
        dimension = sample_embedding.shape[1]
        logger.info(f"📐 임베딩 차원: {dimension}")
        
        # 인덱스 생성 및 문서 삽입
        self.search_engine.create_index(index_name, dimension)
        
        # 삽입 전 인덱스 상태 확인
        initial_stats = self.search_engine.get_index_stats()
        logger.info(f"📊 삽입 전 벡터 수: {initial_stats.get('total_vector_count', 0)}")
        
        self.search_engine.insert_documents(processed_docs)
        
        # 삽입 후 인덱스 상태 확인
        final_stats = self.search_engine.get_index_stats()
        logger.info(f"📊 삽입 후 벡터 수: {final_stats.get('total_vector_count', 0)}")
        
        return index_name
    
    def query(self, question: str, top_k: int = 3, score_threshold: float = 0.0) -> Dict:
        """질문에 대한 답변 생성을 위한 관련 문서 검색"""
        matches = self.search_engine.search(question, top_k, score_threshold)
        
        context = "\n\n".join([match['content'] for match in matches])
        
        return {
            'question': question,
            'context': context,
            'sources': matches,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict:
        """시스템 통계 정보"""
        return self.search_engine.get_index_stats()

class AdvancedRAGSystem(RAGSystem):
    """고급 RAG 시스템 - 답변 생성 및 추가 기능 포함"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 500):
        super().__init__(model_name, chunk_size)
        self.conversation_history = []
    
    def generate_answer(self, question: str, top_k: int = 3, score_threshold: float = 0.2) -> Dict:
        """질문에 대한 답변 생성 (컨텍스트 기반)"""
        search_result = self.query(question, top_k, score_threshold)
        
        if not search_result['sources']:
            return {
                'question': question,
                'answer': "죄송합니다. 관련된 정보를 찾을 수 없습니다.",
                'confidence': 0.0,
                'sources': [],
                'timestamp': datetime.now().isoformat()
            }
        
        # 가장 관련성 높은 소스 기반으로 답변 생성
        best_source = search_result['sources'][0]
        confidence = best_source['score']
        
        # 간단한 답변 생성 로직 (실제로는 LLM을 사용할 수 있음)
        answer = self._generate_contextual_answer(question, search_result['context'])
        
        # 대화 기록에 추가
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'sources': search_result['sources'],
            'context': search_result['context'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_contextual_answer(self, question: str, context: str) -> str:
        """컨텍스트 기반 답변 생성 (간단한 추출 방식)"""
        context_sentences = context.split('.')
        
        # 질문과 가장 관련성 높은 문장들 찾기
        question_keywords = set(question.lower().replace('?', '').split())
        
        relevant_sentences = []
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_keywords & sentence_words)
            if overlap > 0:
                relevant_sentences.append((sentence.strip(), overlap))
        
        if relevant_sentences:
            # 가장 관련성 높은 문장들 정렬
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            answer = '. '.join([s[0] for s in relevant_sentences[:2]])
            return answer + '.' if answer else "관련 정보를 찾았지만 구체적인 답변을 생성할 수 없습니다."
        else:
            return "검색된 컨텍스트에서 직접적인 답변을 찾을 수 없습니다."
    
    def ask_followup(self, question: str) -> Dict:
        """이전 대화 컨텍스트를 고려한 후속 질문 처리"""
        if self.conversation_history:
            # 이전 질문의 컨텍스트를 현재 질문에 통합
            recent_context = self.conversation_history[-1]['answer']
            enhanced_question = f"{question} (이전 컨텍스트: {recent_context[:100]}...)"
            return self.generate_answer(enhanced_question)
        else:
            return self.generate_answer(question)
    
    def get_conversation_summary(self) -> Dict:
        """대화 요약 정보"""
        if not self.conversation_history:
            return {"message": "대화 기록이 없습니다."}
        
        return {
            'total_questions': len(self.conversation_history),
            'average_confidence': sum(h['confidence'] for h in self.conversation_history) / len(self.conversation_history),
            'recent_questions': [h['question'] for h in self.conversation_history[-3:]],
            'start_time': self.conversation_history[0]['timestamp'],
            'last_time': self.conversation_history[-1]['timestamp']
        }
    
    def suggest_related_questions(self, current_question: str) -> List[str]:
        """현재 질문과 관련된 추천 질문 생성"""
        search_result = self.query(current_question, top_k=5, score_threshold=0.1)
        
        suggestions = []
        if search_result['sources']:
            for source in search_result['sources'][:3]:
                content = source['content']
                source_name = source['source']
                
                # 소스별 관련 질문 제안
                if 'pinecone' in source_name.lower():
                    suggestions.extend([
                        "Pinecone의 장점은 무엇인가요?",
                        "벡터 데이터베이스는 어떻게 작동하나요?"
                    ])
                elif 'rag' in source_name.lower():
                    suggestions.extend([
                        "RAG 시스템의 구성요소는 무엇인가요?",
                        "RAG의 한계점은 무엇인가요?"
                    ])
                elif 'embedding' in content.lower() or 'ml_fundamentals' in source_name:
                    suggestions.extend([
                        "임베딩 모델은 어떻게 선택하나요?",
                        "BERT와 RoBERTa의 차이점은 무엇인가요?"
                    ])
                elif 'chunk' in content.lower() or 'preprocessing' in source_name:
                    suggestions.extend([
                        "최적의 청크 크기는 얼마인가요?",
                        "텍스트 전처리 방법에는 어떤 것들이 있나요?"
                    ])
        
        # 중복 제거 및 현재 질문과 다른 것만 반환
        unique_suggestions = list(set(suggestions))
        return [s for s in unique_suggestions if s.lower() != current_question.lower()][:5]
    
    def load_documents_from_file(self, file_path: str) -> List[Dict[str, str]]:
        """파일에서 문서 로드"""
        documents = []
        file_path = Path(file_path)
        
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'content': content,
                    'source': str(file_path),
                    'metadata': {'file_type': 'txt'}
                })
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
        
        return documents
    
    def setup_knowledge_base(self, documents: List[Dict[str, str]], index_name: str = None) -> str:
        """지식 베이스 설정"""
        if not index_name:
            timestamp = int(time.time())
            index_name = f"rag-knowledge-base-{timestamp}"
        
        logger.info(f"📄 원본 문서 수: {len(documents)}")
        
        # 문서 전처리
        processed_docs = self.document_processor.process_documents(documents)
        logger.info(f"📄 {len(documents)}개 문서를 {len(processed_docs)}개 청크로 분할")
        
        # 처리된 문서 내용 확인
        for i, doc in enumerate(processed_docs[:3]):  # 처음 3개만 확인
            logger.info(f"청크 {i}: ID={doc['id']}, 길이={len(doc['content'])}, 내용={doc['content'][:100]}...")
        
        # 임베딩 차원 계산
        sample_embedding = self.search_engine.model.encode(["sample text"])
        dimension = sample_embedding.shape[1]
        logger.info(f"📐 임베딩 차원: {dimension}")
        
        # 인덱스 생성 및 문서 삽입
        self.search_engine.create_index(index_name, dimension)
        
        # 삽입 전 인덱스 상태 확인
        initial_stats = self.search_engine.get_index_stats()
        logger.info(f"📊 삽입 전 벡터 수: {initial_stats.get('total_vector_count', 0)}")
        
        self.search_engine.insert_documents(processed_docs)
        
        # 삽입 후 인덱스 상태 확인
        final_stats = self.search_engine.get_index_stats()
        logger.info(f"📊 삽입 후 벡터 수: {final_stats.get('total_vector_count', 0)}")
        
        return index_name
    
    def query(self, question: str, top_k: int = 3, score_threshold: float = 0.0) -> Dict:
        """질문에 대한 답변 생성을 위한 관련 문서 검색"""
        matches = self.search_engine.search(question, top_k, score_threshold)
        
        context = "\n\n".join([match['content'] for match in matches])
        
        return {
            'question': question,
            'context': context,
            'sources': matches,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict:
        """시스템 통계 정보"""
        return self.search_engine.get_index_stats()

def main():
    """메인 실행 함수"""
    try:
        # RAG 시스템 초기화
        rag = RAGSystem(chunk_size=300)
        
        # 샘플 문서 데이터 (실제로는 파일에서 로드하거나 API에서 가져올 수 있음)
        sample_documents = [
            {
                'content': """
                Pinecone은 머신러닝 애플리케이션을 위한 벡터 데이터베이스입니다. 
                고차원 벡터 데이터를 효율적으로 저장하고 검색할 수 있으며, 
                실시간 유사도 검색과 추천 시스템 구축에 최적화되어 있습니다.
                Serverless 아키텍처를 지원하여 확장성이 뛰어납니다.
                """,
                'source': 'pinecone_docs',
                'metadata': {'category': 'database', 'language': 'korean'}
            },
            {
                'content': """
                벡터 임베딩은 텍스트, 이미지, 오디오 등의 데이터를 고차원 숫자 벡터로 변환하는 기술입니다.
                Sentence Transformers는 문장 수준의 임베딩을 생성하는 라이브러리로,
                BERT, RoBERTa 등의 사전 훈련된 모델을 기반으로 합니다.
                이를 통해 의미적으로 유사한 문장들을 벡터 공간에서 가까운 위치에 배치할 수 있습니다.
                """,
                'source': 'ml_fundamentals',
                'metadata': {'category': 'machine_learning', 'language': 'korean'}
            },
            {
                'content': """
                RAG(Retrieval-Augmented Generation)는 외부 지식 베이스에서 관련 정보를 검색하여
                언어 모델의 생성 능력을 향상시키는 기법입니다.
                이 방법은 사전 훈련된 언어 모델의 한계를 극복하고,
                최신 정보나 도메인 특화 지식을 활용할 수 있게 해줍니다.
                검색된 문서를 컨텍스트로 활용하여 더 정확하고 신뢰할 수 있는 답변을 생성합니다.
                """,
                'source': 'rag_guide',
                'metadata': {'category': 'ai_techniques', 'language': 'korean'}
            },
            {
                'content': """
                자연어 처리에서 청킹(chunking)은 긴 텍스트를 작은 단위로 분할하는 과정입니다.
                효과적인 청킹은 문맥을 보존하면서도 검색 성능을 최적화하는 것이 중요합니다.
                일반적으로 문장 경계, 단락, 또는 의미적 단위를 기준으로 텍스트를 분할합니다.
                청크 크기와 오버랩을 적절히 조정하여 정보 손실을 최소화해야 합니다.
                """,
                'source': 'nlp_preprocessing',
                'metadata': {'category': 'preprocessing', 'language': 'korean'}
            }
        ]
        
        # 지식 베이스 설정
        logger.info("🚀 지식 베이스 설정 시작")
        index_name = rag.setup_knowledge_base(sample_documents)
        
        # 설정 완료 후 상태 확인
        final_stats = rag.get_stats()
        logger.info(f"📊 설정 완료 후 벡터 수: {final_stats.get('total_vector_count', 0)}")
        
        if final_stats.get('total_vector_count', 0) == 0:
            logger.error("❌ 벡터가 저장되지 않았습니다. 프로그램을 종료합니다.")
            return
        
        # 쿼리 테스트
        test_queries = [
            "Pinecone이 무엇인가요?",
            "RAG 기법에 대해 설명해주세요",
            "벡터 임베딩의 원리는 무엇인가요?",
            "청킹은 왜 중요한가요?"
        ]
        
        print("\n" + "="*80)
        print("🔍 RAG 시스템 테스트 결과")
        print("="*80)
        
        for query in test_queries:
            print(f"\n질문: {query}")
            print("-" * 50)
            
            result = rag.query(query, top_k=3, score_threshold=0.0)  # 임계값을 0으로 설정
            
            if result['sources']:
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. [점수: {source['score']:.3f}] {source['source']}")
                    print(f"   내용: {source['content'][:150]}...")
                    print()
            else:
                print("❌ 관련된 문서를 찾을 수 없습니다.")
                # 디버그를 위해 인덱스 상태 확인
                stats = rag.get_stats()
                print(f"   현재 인덱스의 벡터 수: {stats.get('total_vector_count', 0)}")
                
                # 더 자세한 디버그 정보
                if stats.get('total_vector_count', 0) > 0:
                    print("   벡터는 존재하지만 검색되지 않음 - 임베딩 또는 쿼리 문제일 수 있음")
        
        # 통계 정보 출력
        print("\n" + "="*80)
        print("📊 시스템 통계")
        print("="*80)
        stats = rag.get_stats()
        print(f"총 벡터 수: {stats.get('total_vector_count', 0)}")
        print(f"인덱스 이름: {index_name}")
        
        # 정리 옵션 (운영 환경에서는 주석 처리)
        cleanup = input("\n인덱스를 삭제하시겠습니까? (y/N): ").lower()
        if cleanup == 'y':
            rag.search_engine.delete_index()
        
    except Exception as e:
        logger.error(f"🚨 에러 발생: {e}")
        raise

if __name__ == "__main__":
    main()