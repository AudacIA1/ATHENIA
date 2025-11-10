import os
import re
import shutil
import logging
from typing import List, Tuple, Set

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
import google.generativeai as genai
import num2words

from app.models.models import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        # === LEER API KEY ===
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Configura GEMINI_API_KEY en .env")

        self.vector_db_path = "./storage/athenia_data/chroma_db"
        self.chunk_size = 500
        self.chunk_overlap = 100

        # === CONFIGURAR GEMINI ===
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
            logger.info("Gemini configurado con: models/gemini-2.5-flash")
        except Exception as e:
            logger.error(f"Error en Gemini: {e}")
            raise

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self._init_vectorstore()

    def _init_vectorstore(self) -> None:
        os.makedirs(self.vector_db_path, exist_ok=True)
        if os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path):
            self.vectorstore = Chroma(persist_directory=self.vector_db_path, embedding_function=self.embeddings)
            logger.info("VectorDB cargada")
        else:
            self.vectorstore = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine", "hnsw:M": 16}
            )
            logger.info("VectorDB creada")

    def index_document(self, document: Document) -> int:
        if not document.text or not document.text.strip():
            raise ValueError(f"Documento {document.id} sin texto")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(document.text)

        docs = [
            LangchainDocument(
                page_content=chunk,
                metadata={
                    "document_id": document.id,
                    "filename": document.filename,
                    "chunk_index": i
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        self.vectorstore.add_documents(docs)
        logger.info(f"Indexado {len(chunks)} chunks de {document.filename}")
        return len(chunks)

    def query(self, question: str, documents: List[Document], top_k: int = None) -> Tuple[str, float, List[int]]:
        if not documents:
            return "No tengo documentos disponibles.", 0.0, []

        doc_ids = {doc.id for doc in documents}
        target_docs = min(len(documents), 4)
        chunks_per_doc = 3
        total_k = target_docs * chunks_per_doc

        logger.info(f"Query: '{question}' | Docs: {len(documents)}")

        try:
            retrieved = self.vectorstore.similarity_search(
                query=question,
                k=total_k * 2,
                filter={"document_id": {"$in": list(doc_ids)}}
            )
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return "Error en búsqueda.", 0.0, []

        seen = set()
        selected = []
        for doc in retrieved:
            doc_id = doc.metadata["document_id"]
            if doc_id not in seen and len(selected) < total_k:
                selected.append(doc)
                seen.add(doc_id)
            if len(seen) >= target_docs:
                break

        if not selected:
            return "No encontré información relevante.", 0.0, []

        context_parts = [f"Del archivo *{doc.metadata['filename']}*: {doc.page_content.strip()}" for doc in selected]
        contexto_texto = "\n\n".join(context_parts)
        source_ids = list(seen)

        answer, confidence = self._generate_natural_answer(question, contexto_texto, documents)
        return answer, confidence, source_ids

    def _generate_natural_answer(self, question: str, context: str, documents: List[Document]) -> Tuple[str, float]:
        prompt = f"""Eres ATHENIA, una asistente experta, cálida y precisa.

REGLAS:
- Responde en español, natural y humano
- Menciona archivos así: "En X dice...", "En A y B se menciona..."
- Combina o compara información
- Usa TODO el contexto relevante
- Máximo 4-5 oraciones
- Si no sabes: "No lo veo en los documentos"

CONTEXTO:
{context}

PREGUNTA:
{question}

Respuesta:"""

        try:
            logger.info("Llamando a gemini-2.5-flash...")
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_output_tokens": 512
                }
            )
            raw_answer = response.text.strip()
            clean_answer = self._limpiar_markdown(raw_answer)
            clean_answer = self._convertir_numeros_a_palabras(clean_answer)
            confidence = self._calcular_confianza(clean_answer, context)
            logger.info("Respuesta generada")
            return clean_answer, confidence
        except Exception as e:
            logger.error(f"Error en Gemini: {e}")
            if "429" in str(e):
                return "Límite de uso alcanzado. Intenta mañana.", 0.0
            return "Error al generar respuesta.", 0.0

    def _limpiar_markdown(self, texto: str) -> str:
        """Eliminar formato Markdown del texto"""
        texto = re.sub(r'\*\*(.+?)\*\*', r'\1', texto)
        texto = re.sub(r'__(.+?)__', r'\1', texto)
        texto = re.sub(r'\*(.+?)\*', r'\1', texto)
        texto = re.sub(r'_(.+?)_', r'\1', texto)
        texto = re.sub(r'^#+\s+', '', texto, flags=re.MULTILINE)
        texto = re.sub(r'``````', '', texto, flags=re.DOTALL)
        texto = re.sub(r'`(.+?)`', r'\1', texto)
        texto = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', texto)
        texto = re.sub(r'^[\*\-]\s+', '', texto, flags=re.MULTILINE)
        return texto.strip()
    
    def _convertir_numeros_a_palabras(self, texto: str) -> str:
        """Convierte números a palabras en español, manejando monedas y formatos grandes"""
        
        texto = re.sub(r'([$€£¥₡])\s*', '', texto)
        
        def reemplazar_numero(match):
            numero = match.group(0)
            try:
                numero_limpio = numero.replace(',', '')
                
                if '.' in numero_limpio:
                    partes = numero_limpio.split('.')
                    if len(partes[1]) <= 2: 
                        numero_float = float(numero_limpio)
                        return num2words(numero_float, lang='es')
                    else:
                        return num2words(int(partes[0]), lang='es')
                else:
                    numero_int = int(numero_limpio)
                    resultado = num2words(numero_int, lang='es')
                    
                    resultado = resultado.replace('billones', 'mil millones')
                    
                    return resultado
            except:
                return numero
        
        texto = re.sub(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', reemplazar_numero, texto)
        texto = re.sub(r'\bmillones\s+millones\b', 'millones', texto, flags=re.IGNORECASE)
        
        return texto

    def _calcular_confianza(self, answer: str, context: str) -> float:
        if len(answer) < 30:
            return 0.4
        if any(p in answer.lower() for p in ["no lo veo", "no aparece", "no dice"]):
            return 0.5
        overlap = len(set(answer.lower().split()) & set(context.lower().split()))
        return round(min(0.6 + overlap / 50, 1.0), 2)

    def safe_clear_index(self):
        """Limpia el índice de forma segura"""
        try:
            if hasattr(self, 'vectorstore'):
                self.vectorstore = None
            if os.path.exists(self.vector_db_path):
                shutil.rmtree(self.vector_db_path)
                logger.info("Carpeta de índice eliminada")
            self._init_vectorstore()
            logger.info
        except Exception as e:
            logger.error(f"Error en safe_clear_index: {e}")
            raise

    def delete_document_chunks(self, document_id: int):
        """Borra solo los chunks de un documento"""
        try:
            results = self.vectorstore._collection.get(
                where={"document_id": document_id},
                include=["ids"]
            )
            if results['ids']:
                self.vectorstore._collection.delete(ids=results['ids'])
                logger.info(f"Eliminados {len(results['ids'])} chunks del doc {document_id}")
            else:
                logger.info(f"No hay chunks para document_id {document_id}")
        except Exception as e:
            logger.error(f"Error borrando chunks: {e}")

    def get_db_size(self) -> float:
        try:
            total = sum(os.path.getsize(os.path.join(d, f))
                        for d, _, fs in os.walk(self.vector_db_path) for f in fs)
            return round(total / (1024 * 1024), 2)
        except:
            return 0.0