"""
SAP Document processing module for handling, analyzing, and searching through DOCX documents.

This module provides functionality for processing SAP documents, including:
- Document analysis and extraction
- Document upload and storage in S3
- Vector database integration for semantic search
- Text and image summarization
- Multi-modal RAG (Retrieval Augmented Generation) functionality
"""

import io
from datetime import date
from be_module.src.models.sap_documents_model import SapDocument
from docx import Document
from dotenv import load_dotenv
from fastapi import UploadFile
from fastapi.exceptions import HTTPException
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from loguru import logger
from sqlalchemy import select, desc

from shared_access.utils.s3_handler import S3Handler

from .analyzer import Analyzer
from .config import sap_documents_config
from .document_processor import DocumentProcessor
from .document_search import DocumentSearch
from .summarizer import Summarizer
from .utils import img_prompt_func, split_image_text_types
import os
import smtplib
from datetime import date, datetime, timezone
from email.message import EmailMessage

class InternalQueryExpander:
	"""Internal query expander for retrieval.

	Goal:
	- Generate a BM25-friendly keyword/phrase query (2-6 tokens/phrases).
	- Keep the original user question for vector retrieval (semantic).
	"""
	def __init__(self):
		self.model = ChatBedrock(
			model_id=sap_documents_config.LLM_MODEL_ID,
			model_kwargs={'temperature': 0},
			region_name=sap_documents_config.REGION_NAME,
		)
		self.parser = StrOutputParser()
		self.prompt = (
			"Strictly extract 2-6 core technical keywords or short phrases for search.\n"
			"Remove conversational filler words (e.g., 'actually', 'anymore', 'anyhow').\n"
			"Fix obvious typos if any.\n"
			"Return ONLY the keywords/phrases separated by spaces.\n"
			"Question: {question}\n"
			"Keywords:"
		)
		# Document expiry notification recipients.
		# Configure via DOC_EXPIRY_NOTIFY_EMAILS="a@b.com,c@d.com".
		self.doc_expiry_notify_emails = [e.strip() for e in os.getenv('DOC_EXPIRY_NOTIFY_EMAILS', 'replace_by_real_email_address').split(',') if e.strip()]
		self.notify_from_email = os.getenv('NOTIFY_FROM_EMAIL', 'no-reply@solver.local')


	def expand(self, question: str) -> str:
		q = (question or "").strip()
		if not q:
			return q
		try:
			res = (self.model | self.parser).invoke(self.prompt.format(question=q))
			res = (res or "").replace('"', '').strip()
			return res or q
		except Exception as e:
			logger.error(f"Query expansion failed: {e}")
			return q


load_dotenv()


class SAPDocuments:
	"""
	Main class for SAP document processing and retrieval operations.

	Handles document uploads, processing, analysis, and multi-modal RAG queries.
	"""

	def __init__(self, document_search: DocumentSearch):
		self.document_search = document_search
		self.s3_image_handler = S3Handler(sap_documents_config.S3_IMAGE_CACHE)
		self.s3_doc_handler = S3Handler(sap_documents_config.S3_SAP_DOCS)
		self.analyzer = Analyzer()
		self.doc_processor = DocumentProcessor()
		self.summarizer = Summarizer()
		self.expander = InternalQueryExpander()

	async def analyze_document(self, file: UploadFile):
		"""
		Analyze a DOCX document and extract its contents.

		Args:
		    file: UploadFile object containing the document to analyze

		Returns:
		    Analysis results of the document

		Raises:
		    HTTPException: If the file is not a DOCX document
		"""
		if not file.filename.lower().endswith('.docx'):
			raise HTTPException(status_code=400, detail='Only .docx files are supported')

		content = await file.read()
		return self.analyzer.analyze_document(file=content)

	async def check_document_uploadable(self, file_name: str) -> bool:
		if not file_name.lower().endswith('.docx'):
			raise HTTPException(status_code=400, detail='Only .docx files are supported')

		existing_doc_name = self.document_search.get_document_id_by_name(file_name)
		if existing_doc_name is not None:
			raise HTTPException(status_code=400, detail=f"Document '{file_name}' already exists")

		return True

	async def upload_document(self, 
						   s3_key: str,
						   *,
						   version:str = 'v1.0',
						   status: str = 'active',
						   validity_end: date | None = None,
						   validity_year: int = 1,
						   domain: str | None = None,
						   language: str | None = None,
						   author: str | None = None,
						   keywords: list[str] | None = None,
						   ) -> str:
		"""
		Upload a DOCX document to S3 and ingest it into the vector database.

		Args:
		    file: UploadFile object containing the document to upload

		Returns:
		    str: Document name/identifier

		Raises:
		    HTTPException: If the file is not a DOCX document or already exists
		"""
		try:
			content = self.s3_doc_handler.get_file(s3_key)
			self.ingest_document_to_vector_db(file=content, 
									 file_name=s3_key,
									 version=version,
									 status=status,
									 validity_end=validity_end,
									 validity_year=validity_year,
									 domain=domain,
									 language=language,
									 author=author,
									 keywords=keywords,
									 )
		except Exception as e:
			raise HTTPException(status_code=400, detail=str(e)) from e
		return s3_key



	def list_documents(self, limit: int = 200, offset: int = 0) -> list[dict]:
		"""List documents for the UI.

		This is a thin wrapper around DocumentSearch.list_documents(), which owns DB access.
		"""
		self.enforce_document_validity()
		documents = self.document_search.list_documents(limit=limit, offset=offset)

		# Keep API response schema stable for the frontend/UI.
		for doc in documents:
			doc.setdefault("presigned_url", None)
		return documents
	
	def _send_email(self, *, to_emails: list[str], subject: str, body: str) -> bool:
		"""Send a plain-text email via SMTP.

		Environment variables:
		- SMTP_HOST (required)
		- SMTP_PORT (optional, default 587)
		- SMTP_USER / SMTP_PASSWORD (optional)
		- SMTP_USE_TLS (optional, default true)

		If SMTP_HOST is not configured, the function logs a warning and returns False.
		"""
		smtp_host = os.getenv('SMTP_HOST')
		if not smtp_host:
			logger.warning('SMTP_HOST is not configured; skip sending document expiry emails.')
			return False

		smtp_port = int(os.getenv('SMTP_PORT', '587'))
		smtp_user = os.getenv('SMTP_USER')
		smtp_password = os.getenv('SMTP_PASSWORD')
		use_tls = os.getenv('SMTP_USE_TLS', 'true').strip().lower() in {'1', 'true', 'yes', 'y'}

		msg = EmailMessage()
		msg['From'] = self.notify_from_email
		msg['To'] = ', '.join(to_emails)
		msg['Subject'] = subject
		msg.set_content(body)
		try:
			with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
				if use_tls:
					server.starttls()
				if smtp_user and smtp_password:
					server.login(smtp_user, smtp_password)
				server.send_message(msg)
			return True
		except Exception as e:
			logger.error(f'Failed to send document expiry email: {e}')
			return False


	def enforce_document_validity(self) -> list[dict]:
		"""Enforce document validity and notify when documents become deprecated."""
		try:
			updated = self.document_search.expire_documents(today=date.today())
			if not updated:
				return []

			timestamp = datetime.now(timezone.utc).isoformat()
			subject = f'Solver: {len(updated)} SAP document(s) automatically deprecated (expired)'
			lines = [
				f'Timestamp (UTC): {timestamp}',
				'',
				'The following documents were automatically marked as "deprecated" because they are expired:',
				'',
			]
			for d in updated:
				lines.append(
					f"- id={d.get('id')} name={d.get('document_name')} rule={d.get('expiry_rule')} "
					f"expiry_date={d.get('expiry_date')} upload_date={d.get('upload_date')} validity_year={d.get('validity_year')}"
				)
			body = '\n'.join(lines)

			# Recipients: configured list + author (if it looks like an email).
			recipients = set(self.doc_expiry_notify_emails)
			for d in updated:
				author = (d.get('author') or '').strip()
				if '@' in author and ' ' not in author:
					recipients.add(author)
			self._send_email(to_emails=sorted(recipients), subject=subject, body=body)
			return updated

		except Exception as e:
			logger.error(f'Failed to enforce document validity: {e}')
			return []

	def update_document_metadata(self, doc_id: int, updates: dict) -> dict:
		"""Update document metadata (status/domain/language/keywords/etc.) without re-uploading."""
		updated = self.document_search.update_document_metadata(doc_id=doc_id, updates=updates)
		if not updated:
			raise HTTPException(status_code=404, detail='Document not found')
		return updated
	

	def delete_document(self, doc_id: int) -> bool:
		return self.document_search.delete_document(doc_id)


	def ingest_document_to_vector_db(
			self, 
			file: bytes, 
			file_name: str,
			*,
			version:str = 'v1.0',
			status: str = 'active',
			validity_end: date | None = None,
			validity_year: int = 1,
			domain: str | None = None,
			language: str | None = None,
			author: str | None = None,
			keywords: list[str] | None = None,
			):
		"""
		Process a document and ingest its content into the vector database.

		Extracts text, tables, and images from the document, generates summaries,
		and stores everything in the vector database.

		Args:
		    file: Bytes content of the document
		    file_name: Name of the document file
		"""
		doc = Document(io.BytesIO(file))
		extracted_tables, extracted_images, extracted_text = self.doc_processor.process_document(
			doc=doc, output_file=None, file_name=file_name, is_save=False, s3_image_uploader=self.s3_image_handler
		)

		doc_id = self.document_search.create_document(
			document_name=file_name, data_type='docx',
			version=version,
			status=status,
			validity_end=validity_end,
			validity_year=validity_year,
			domain=domain,
			language=language,
			author=author,
			keywords=keywords,
			)

		if extracted_text:
			text_summaries = self.summarizer.generate_text_table_summaries(raw_data=extracted_text)
			self.document_search.add_patching_points(content_summaries=text_summaries, raw_data=extracted_text, data_type='text', document_id=doc_id)
		if extracted_tables:
			table_summaries = self.summarizer.generate_text_table_summaries(raw_data=extracted_tables)
			self.document_search.add_patching_points(
				content_summaries=table_summaries, raw_data=extracted_tables, data_type='table', document_id=doc_id
			)
			self.document_search.add_patching_points(raw_data=extracted_tables, data_type='table', document_id=doc_id)
		if extracted_images['encoded_images']:
			image_summaries = self.summarizer.generate_img_summaries(images=extracted_images['encoded_images'])
			self.document_search.add_patching_points(
				content_summaries=image_summaries, raw_data=extracted_images['file_name_images'], data_type='image', document_id=doc_id
			)

	def multi_modal_rag_chain(self):
		"""
		Create a multi-modal RAG chain for processing queries.

		Returns:
		    A runnable chain that can process multi-modal queries (text and images)
		"""
		model = ChatBedrock(
			model_id=sap_documents_config.LLM_MODEL_ID,
			model_kwargs={'temperature': sap_documents_config.LLM_TEMPERATURE},
			region_name=sap_documents_config.REGION_NAME,
		)
		return RunnablePassthrough() | RunnableLambda(img_prompt_func) | model | StrOutputParser()

	def multi_modal_rag_stream_chain(self):
		"""
		Create a streaming multi-modal RAG chain for processing queries.

		Returns:
		    A streaming-capable runnable chain for multi-modal queries
		"""
		model = ChatBedrock(
			model_id=sap_documents_config.LLM_MODEL_ID,
			model_kwargs={'temperature': sap_documents_config.LLM_TEMPERATURE},
			region_name=sap_documents_config.REGION_NAME,
		)
		return RunnablePassthrough() | RunnableLambda(img_prompt_func) | model | StrOutputParser()


	def execute(self, question: str, filters: dict | None = None):
		"""
		Execute a query against the document collection.

		Args:
		    question: String query to execute

		Returns:
		    tuple: (answer string, search results)
		"""
		self.enforce_document_validity()
		refined_query = self.expander.expand(question)
		logger.info(f"Original: {question} -> Refined: {refined_query}")
		
		chain_multimodal_rag = self.multi_modal_rag_chain()

		search_results = self.document_search.search(question=question, bm25_query=refined_query, top_k=5, filters=filters)

		if not search_results:
			return "No documentation found for the optimized query.", []
		
		context = split_image_text_types(search_results=search_results, s3_handler=self.s3_image_handler)
		
		answer = chain_multimodal_rag.invoke({'context': context, 'question': question})
		return answer, search_results

	async def execute_stream(self, question: str, filters:dict | None = None):
		"""
		Execute a query with streaming response.

		Args:
		    question: String query to execute

		Yields:
		    Dict containing streaming response chunks and references
		"""
		self.enforce_document_validity()
		refined_query = self.expander.expand(question)
		search_results = self.document_search.search(question=question, bm25_query=refined_query, top_k=5, filters=filters)
	
		if not search_results:
			yield {'type': 'message', 'deltaContent': "No documentation found."}
			return
		
		context = split_image_text_types(search_results=search_results, s3_handler=self.s3_image_handler)
		chain = self.multi_modal_rag_stream_chain()
		
		async for stream in chain.astream({'context': context, 'question': question}):
			yield {'type': 'message', 'deltaContent': stream}
