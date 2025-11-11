import os
import re
import shutil
import logging
from typing import List, Tuple, Set
import random
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
import google.generativeai as genai
import num2words
from google.generativeai.types import HarmCategory, HarmBlockThreshold
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
        # === DEFINIR RESPUESTAS DE SMALL TALK ===
        self.small_talk_responses = {
            "greeting": {
                "patterns": ["hola", "buenos días", "buenas tardes", "hey", "qué tal"],
                "responses": [
                    "¡Hola! Soy Athenia, tu asistente de documentos. ¿En qué puedo ayudarte?",
                    "¡Hola! ¿Qué necesitas saber hoy?",
                    "¡Hola! Pregúntame sobre tus documentos."
                ]
            },
            "identity": {
                "patterns": ["quién eres", "qué eres", "tu nombre", "como te llamas"],
                "responses": [
                    "Soy Athenia, tu asistente inteligente para consultar documentos.",
                    "Me llamo Athenia y estoy aquí para ayudarte con tus archivos."
                ]
            },
            "capabilities": {
                "patterns": ["qué puedes hacer", "qué haces", "para qué sirves", "ayuda", "ayúdame"],
                "responses": [
                    "Puedo responder preguntas sobre tus documentos. Solo pregunta lo que necesitas saber.",
                    "Busco información en tus archivos y te doy respuestas rápidas. ¿Qué quieres saber?"
                ]
            },
            "thanks": {
                "patterns": ["gracias", "muchas gracias", "perfecto", "excelente"],
                "responses": [
                    "¡De nada! ¿Algo más?",
                    "Con gusto. ¿Necesitas otra cosa?",
                    "¡Para eso estoy!"
                ]
            },
            "goodbye": {
                "patterns": ["adiós", "chao", "hasta luego", "nos vemos"],
                "responses": [
                    "¡Hasta pronto!",
                    "¡Nos vemos!",
                    "¡Chao! Vuelve cuando quieras."
                ]
            }
        }

        # === INICIALIZAR EMBEDDINGS ===
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self._init_vectorstore()

    

    def _detect_small_talk(self, question: str) -> Tuple[bool, str]:
        """
        Detecta si la pregunta es small talk
        Retorna: (es_small_talk, respuesta)
        """
        question_lower = question.lower().strip()
        
        # Revisar cada categoría
        for category, data in self.small_talk_responses.items():
            for pattern in data["patterns"]:
                if pattern in question_lower:
                    response = random.choice(data["responses"])
                    return True, response
        
        return False, ""


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
        # 1. PRIMERO: Detectar small talk
        is_small_talk, small_talk_response = self._detect_small_talk(question)
        if is_small_talk:
            logger.info(f"Small talk detectado: {question}")
            return small_talk_response, 1.0, []
        
        # 2. Si no hay documentos
        if not documents:
            return "Todavía no tengo documentos. Sube algunos para ayudarte.", 0.0, []

        doc_ids = {doc.id for doc in documents}
        target_docs = min(len(documents), 4)
        chunks_per_doc = 3
        total_k = target_docs * chunks_per_doc

        logger.info(f"Consulta RAG: '{question}' | Docs: {len(documents)}")

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

        context_parts = [doc.page_content.strip() for doc in selected]
        contexto_texto = "\n\n".join(context_parts)
        source_ids = list(seen)

        answer, confidence = self._generate_natural_answer(question, contexto_texto, documents)
        return answer, confidence, source_ids

    def _generate_natural_answer(self, question: str, context: str, documents: List[Document]) -> Tuple[str, float]:
        prompt = f"""Eres ATHENIA, asistente amigable y directa.

REGLAS ESTRICTAS:
- Responde en máximo 2 oraciones
- Habla como si la información fuera tuya
- NUNCA digas "según", "de acuerdo a", "en el documento"
- Si no sabes: "No encuentro eso en los docs"
- SIN markdown, SIN asteriscos, SIN formato
- Responde directo, sin mencionar fuentes ni archivos

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
                    "top_p": 0.9,
                    "max_output_tokens": 500
                },
                safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            if not response.candidates:
                logger.warning("Sin candidatos en respuesta de Gemini")
                return "No pude generar una respuesta. Intenta reformular la pregunta.", 0.3
            candidate = response.candidates[0]

            # Verificar finish_reason
            if candidate.finish_reason != 1:  # 1 = STOP (normal)
                logger.warning(f"finish_reason anormal: {candidate.finish_reason}")
                
                # finish_reason: 2 = MAX_TOKENS, 3 = SAFETY, 4 = RECITATION
                if candidate.finish_reason == 2:
                    return "La respuesta fue muy larga. Sé más específico en tu pregunta.", 0.3
                elif candidate.finish_reason == 3:
                    return "No puedo responder eso por razones de seguridad.", 0.3
                elif candidate.finish_reason == 4:
                    return "Encontré contenido protegido. Reformula tu pregunta.", 0.3

            if not candidate.content or not candidate.content.parts:
                logger.warning("Respuesta sin contenido")
                return "No pude generar una respuesta. Intenta de nuevo.", 0.3

            raw_answer = response.text.strip()
            clean_answer = self._limpiar_markdown(raw_answer)
            clean_answer = self._limpiar_referencias(clean_answer)
            clean_answer = self._convertir_numeros_a_palabras(clean_answer)

            sentences = clean_answer.split('. ')
            if len(sentences) > 2:
                clean_answer = '. '.join(sentences[:2]) + '.'
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
    
    def _limpiar_referencias(self, texto: str) -> str:
        """Eliminar referencias a fuentes como 'Según X', 'De acuerdo a', etc."""
        
        patrones = [
            r'^Según\s+\w+,\s*',              
            r'^De acuerdo a\s+\w+,\s*',       
            r'^En el documento\s+[\w\s]+,\s*',
            r'^El documento dice que\s+',     
            r'^Basado en\s+\w+,\s*',          
            r'^\w+\s+dice que\s+',            
            r'^\w+\s+menciona que\s+',        
            r'^\w+\s+indica que\s+',          
            r'^Conforme a\s+\w+,\s*',         
            r'^Como indica\s+\w+,\s*',        
        ]
        
        for patron in patrones:
            texto = re.sub(patron, '', texto, flags=re.IGNORECASE)
        
        if texto:
            texto = texto[0].upper() + texto[1:]
        
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