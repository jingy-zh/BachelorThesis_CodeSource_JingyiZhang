"""
Document search and embedding module for SAP documents.

This module provides functionality for managing document embeddings in a
vector database, creating and deleting documents, adding document content
with embeddings, and performing semantic search on document content.
"""
from typing import Optional

from langchain_openai import AzureOpenAIEmbeddings
from loguru import logger
from sqlalchemy import select, func, desc
from sqlalchemy.orm import aliased
from tqdm import tqdm
from datetime import date
from typing import Optional, Dict, Any

from ...database import DatabaseManager
from .config import SapDocsEmbedding, SapDocument, sap_documents_config


class DocumentSearch:
	"""
	Manages document storage, embedding generation, and semantic search operations.

	This class handles the creation, deletion, and searching of documents using
	vector embeddings. It supports document management, adding content with
	embeddings, and performing semantic similarity searches.
	"""
	def __init__(self, db_manager: DatabaseManager, creds: dict, is_reset: bool = False):
		self.embeddings_model = AzureOpenAIEmbeddings(
			model=sap_documents_config.EMBEDDING_MODEL,
			azure_deployment=sap_documents_config.DEPLOYMENT_NAME,
			azure_endpoint=creds['azure_endpoint'],
			api_key=creds['api_key'],
		)
		self.vector_size = sap_documents_config.VECTOR_SIZE
		self.batch_size = sap_documents_config.BATCH_SIZE
		self.documents_table = sap_documents_config.TABLE_DOCUMENTS
		self.embeddings_table = sap_documents_config.TABLE_EMBEDDINGS
		self.db_manager = db_manager

		if is_reset:
			self._drop_tables()
			logger.info(f'Dropped tables {self.documents_table} and {self.embeddings_table}')

	def _drop_tables(self):
		try:
			self.db_manager.wipe_table_data(self.embeddings_table)
			self.db_manager.wipe_table_data(self.documents_table)
		except Exception as e:
			logger.error(f'Error dropping tables: {str(e)}')

	def create_document(self, 
					 document_name: str, 
					 data_type: str,
					 *,
					 version:str = 'v1.0',
					 status: str = 'active',
					 validity_end: date | None = None,
					 validity_year: int = 1,
					 domain: str | None = None,
					 language: str | None = None,
					 author: str | None = None,
					 keywords: list[str] | None = None,
					 ) -> Optional[int]:
		"""
		Create a new document entry in the database.

		Creates a new document with the given name and type if it doesn't already exist.

		Args:
		    document_name: Name of the document to create
		    data_type: Type of the document

		Returns:
		    Optional[int]: ID of the created document, or None if creation failed or document exists
		"""
		with self.db_manager.create_session() as session:
			try:
				existing_document = session.query(SapDocument).filter_by(document_name=document_name, data_type=data_type).first()
				if existing_document:
					return None  # Document already exists
				 
				# add keywords
				keywords_text = ','.join(keywords) if keywords else None

				# Create a new document
				new_document = SapDocument(
					document_name=document_name, 
					data_type=data_type,
					version = version,
					status = status,
					validity_end = validity_end,
					validity_year = validity_year,
					domain = domain,
					language = language,
					author = author,
					keywords = keywords_text
					)
				session.add(new_document)
				session.commit()
				return new_document.id
			except Exception as e:
				logger.error(f'Error creating document: {str(e)}')
				session.rollback()
				return None

	def delete_document(self, document_id: int) -> bool:
		"""
		Delete a document from the database by its ID.

		Args:
		    document_id: ID of the document to delete

		Returns:
		    bool: True if the document was successfully deleted, False otherwise
		"""
		with self.db_manager.create_session() as session:
			try:
				result = session.query(SapDocument).filter_by(id=document_id).delete()
				session.commit()
				return result > 0  # Returns True if a row was deleted
			except Exception as e:
				logger.error(f'Error deleting document: {str(e)}')
				session.rollback()
				return False

	def add_patching_points(self, raw_data: list[str], data_type: str, document_id: int, content_summaries: Optional[list[str]] = None):
		"""
		Add document content with embeddings to the database.

		Processes raw data in batches, generates embeddings, and stores both the data
		and embeddings in the database. Can optionally include content summaries.

		Args:
		    raw_data: List of text strings to embed and store
		    data_type: Type of the data being stored (e.g., 'text', 'table', 'image')
		    document_id: ID of the document to associate with this content
		    content_summaries: Optional list of summaries corresponding to the raw data
		"""
		num_features = len(raw_data)
		num_batches = (num_features + self.batch_size - 1) // self.batch_size

		for i in tqdm(range(num_batches), desc=f'Processing batches for {data_type}', unit='batch'):
			start_idx = i * self.batch_size
			end_idx = min((i + 1) * self.batch_size, num_features)
			with self.db_manager.create_session() as session:
				try:
					if content_summaries is None:
						vectors = self.embeddings_model.embed_documents(raw_data[start_idx:end_idx])
						data_list = [
							{'embedding': vector, 'raw_data': raw_data[j], 'summary': '', 'data_type': data_type, 'document_id': document_id}
							for j, vector in enumerate(vectors, start=start_idx)
						]
					else:
						vectors = self.embeddings_model.embed_documents(content_summaries[start_idx:end_idx])
						data_list = [
							{
								'embedding': vector,
								'raw_data': raw_data[j],
								'summary': content_summaries[j],
								'data_type': data_type,
								'document_id': document_id,
							}
							for j, vector in enumerate(vectors, start=start_idx)
						]

					session.bulk_insert_mappings(SapDocsEmbedding, data_list)

					session.commit()
				except Exception as e:
					logger.error(f'Error adding patching points: {str(e)}')
					session.rollback()

	def search(
		self, 
		question: str, 
		top_k: int = 3,
		filters: Optional[Dict[str, Any]] = None,
		bm25_query: Optional[str] = None,
		) -> list[dict]:
		"""
		Hybrid retrieval (Vector + BM25/keyword gate) to reduce hallucinations.
	
		Strategy:
		- Always compute vector candidates (pgvector cosine_distance; lower is better).
		- Compute BM25 candidates using PostgreSQL Full-Text Search (FTS).
		- Prefer doc-level intersection: keep vector chunks whose document_id appears in BM25 hits.
		- If BM25 returns no hits, return [] by default (so caller can refuse to answer safely).
	
		"""
		q = (question or "").strip()
		if not q:
			return []

		bm25_text = (bm25_query or q).strip()

		vector = self.embeddings_model.embed_query(q)
		
	
		# Hybrid configuration (safe defaults)
		use_hybrid = bool(getattr(sap_documents_config, "HYBRID_RETRIEVAL", True))
		allow_vector_fallback = bool(getattr(sap_documents_config, "ALLOW_VECTOR_FALLBACK_WHEN_BM25_EMPTY", False))
		fts_lang = getattr(sap_documents_config, "FTS_LANGUAGE", "english")

		soft_fallback = bool(getattr(sap_documents_config, "HYBRID_SOFT_FALLBACK", True))
		fallback_max_dist = float(getattr(sap_documents_config, "HYBRID_FALLBACK_MAX_DISTANCE", 0.28))
		fts_query_mode = str(getattr(sap_documents_config, "FTS_QUERY_MODE", "websearch")).lower()

		vec_mult = int(getattr(sap_documents_config, "HYBRID_VEC_MULT", 10))
		bm25_mult = int(getattr(sap_documents_config, "HYBRID_BM25_MULT", 10))
		vec_min = int(getattr(sap_documents_config, "HYBRID_VEC_MIN", 30))
		bm25_min = int(getattr(sap_documents_config, "HYBRID_BM25_MIN", 30))

		vec_k = max(vec_min, top_k * vec_mult)
		bm25_k = max(bm25_min, top_k * bm25_mult)

		def _as_list(v):
			if v is None:
				return []
			if isinstance(v, (list, tuple, set)):
				return list(v)
			return [v]
	
		def _apply_filters(query, d_alias):
			if not filters:
				return query

			status_vals = _as_list(filters.get('status'))
			if status_vals:
				query = query.where(d_alias.status.in_(status_vals))

			data_type_vals = _as_list(filters.get('data_type'))
			if data_type_vals:
				query = query.where(d_alias.data_type.in_(data_type_vals))

			validity_year_vals = _as_list(filters.get('validity_year'))
			if validity_year_vals:
				query = query.where(d_alias.validity_year.in_(validity_year_vals))

			version_vals = _as_list(filters.get('version'))
			if version_vals:
				query = query.where(d_alias.version.in_(version_vals))

			domain_vals = _as_list(filters.get('domain'))
			if domain_vals:
				query = query.where(d_alias.domain.in_(domain_vals))

			author_vals = _as_list(filters.get('author'))
			if author_vals:
				query = query.where(d_alias.author.in_(author_vals))

			language_vals = _as_list(filters.get('language'))
			if language_vals:
				query = query.where(d_alias.language.in_(language_vals))

			return query
	
		with self.db_manager.create_session() as session:
			try:
				e = aliased(SapDocsEmbedding, name='e')
				d = aliased(SapDocument, name='d')

				# 1) Vector candidates (global)
				q_vec_base = (
					select(
						e.id.label('embedding_id'),
						d.id.label('document_id'),
						e.raw_data.label('raw_data'),
						e.summary.label('summary'),
						e.data_type.label('data_type'),
						d.document_name.label('document_name'),
						e.embedding.cosine_distance(vector).label('cosine_distance'),
					)
					.join(d, e.document_id == d.id)
				)
				q_vec_base = _apply_filters(q_vec_base, d)
				q_vec = q_vec_base.order_by('cosine_distance').limit(vec_k)
				vec_rows = session.execute(q_vec).all()

				if not vec_rows:
					return []

				# Vector-only mode
				if not use_hybrid:
					final_rows = vec_rows[:top_k]
				else:
					# 2) BM25 candidates (PostgreSQL FTS)
					bm25_rows = []
					try:
						search_text = func.concat_ws(
							' ',
							func.coalesce(d.document_name, ''),
							func.coalesce(d.keywords, ''),
							func.coalesce(e.summary, ''),
							func.coalesce(e.raw_data, ''),
						)
						tsv = func.to_tsvector(fts_lang, search_text)

						if fts_query_mode == 'websearch':
							tsq = func.websearch_to_tsquery(fts_lang, bm25_text)
						else:
							# default: websearch_tsquery
							tsq = func.websearch_tsquery(fts_lang, bm25_text)

						q_bm25 = (
							select(
								e.id.label('embedding_id'),
								d.id.label('document_id'),
								func.ts_rank_cd(tsv, tsq).label('bm25_score'),
							)
							.join(d, e.document_id == d.id)
							.where(tsv.op('@@')(tsq))
						)
						q_bm25 = _apply_filters(q_bm25, d)
						q_bm25 = q_bm25.order_by(desc('bm25_score')).limit(bm25_k)
						bm25_rows = session.execute(q_bm25).all()

					except Exception as e_bm25:
						# If FTS fails (e.g., non-Postgres DB), fall back to vector-only to avoid breaking the service.
						logger.warning(
							f'BM25/FTS query failed; falling back to vector-only. Error: {str(e_bm25)}'
						)
						bm25_rows = []

					# Determine whether falling back to vector-only results
					best_dist = None
					try:
						best_dist = float(getattr(vec_rows[0], 'cosine_distance', None))
					except Exception:
						best_dist = None

					def _should_vector_fallback() -> bool:
						if allow_vector_fallback:
							return True
						if soft_fallback and best_dist is not None and best_dist <= fallback_max_dist:
							return True
						return False

					# 3) Doc-level gate + doc-restricted vector rerank
					if not bm25_rows:
						if _should_vector_fallback():
							final_rows = vec_rows[:top_k]
						else:
							return []
					else:
						bm25_doc_ids = {
							r.document_id for r in bm25_rows
							if getattr(r, 'document_id', None) is not None
						}
						if not bm25_doc_ids:
							if _should_vector_fallback():
								final_rows = vec_rows[:top_k]
							else:
								return []
						else:
							hybrid_rows = [r for r in vec_rows if r.document_id in bm25_doc_ids]
							if hybrid_rows:
								final_rows = hybrid_rows[:top_k]
							else:
								# Second-stage: vector search restricted to BM25-hit documents
								q_vec_in_docs = (
									q_vec_base
									.where(d.id.in_(list(bm25_doc_ids)))
									.order_by('cosine_distance')
									.limit(max(top_k, 10))
								)
								vec_doc_rows = session.execute(q_vec_in_docs).all()
								if vec_doc_rows:
									final_rows = vec_doc_rows[:top_k]
								elif _should_vector_fallback():
									final_rows = vec_rows[:top_k]
								else:
									return []

			except Exception as e_search:
				logger.error(f'Error searching documents: {str(e_search)}')
				return []

		# Deduplicate by raw_data to keep response stable
		unique_results = []
		seen = set()
		for r in final_rows:
			key = r.raw_data
			if key not in seen:
				unique_results.append(r)
				seen.add(key)

		return [
			{
				'score': r.cosine_distance,
				'raw_data': r.raw_data,
				'summary': r.summary,
				'data_type': r.data_type,
				'document_name': r.document_name,
				'embedding_id': getattr(r, 'embedding_id', None),
				'document_id': getattr(r, 'document_id', None),
			}
			for r in unique_results
		]


	def list_documents(self, limit: int = 200, offset: int = 0) -> list[dict]:
		"""List documents (metadata only) for the management UI.

		Performance notes:
		- Avoid ORM entity materialization (`session.query(SapDocument).all()`), which can be slow.
		- Select only the columns the UI needs and apply pagination.
		"""
		try:
			with self.db_manager.create_session() as session:
				stmt = (
					select(
						SapDocument.id,
						SapDocument.document_name,
						SapDocument.data_type,
						SapDocument.date_time,
						SapDocument.version,
						SapDocument.status,
						SapDocument.validity_end,
						SapDocument.validity_year,
						SapDocument.domain,
						SapDocument.language,
						SapDocument.author,
						SapDocument.keywords,
					)
					.order_by(desc(SapDocument.date_time))
					.limit(limit)
					.offset(offset)
				)
				rows = session.execute(stmt).all()

				result: list[dict] = []
				for r in rows:
					keywords = None
					if r.keywords:
						keywords = [k.strip() for k in str(r.keywords).split(",") if k and str(k).strip()]

					result.append(
						{
							"id": r.id,
							"document_name": r.document_name,
							"data_type": r.data_type,
							"date_time": r.date_time.strftime("%Y-%m-%d %H:%M:%S") if r.date_time else None,
							"presigned_url": None,
							"version": r.version,
							"status": r.status,
							"validity_end": r.validity_end.isoformat() if r.validity_end else None,
							"validity_year": r.validity_year,
							"domain": r.domain,
							"language": r.language,
							"author": r.author,
							"keywords": keywords,
						}
					)
				return result
		except Exception as e:
			logger.error(f"Error listing documents: {str(e)}")
			return []


	def update_document_metadata(self, doc_id: int, updates: dict) -> Optional[dict]:
		"""Update metadata fields for a document without re-ingesting embeddings."""
		try:
			with self.db_manager.create_session() as session:
				doc = session.query(SapDocument).filter(SapDocument.id == doc_id).first()
				if not doc:
					return None

				# Apply updates (only keys provided)
				for k, v in updates.items():
					if k == 'keywords':
						if v is None:
							doc.keywords = None
						elif isinstance(v, list):
							doc.keywords = ",".join([str(x).strip() for x in v if x is not None and str(x).strip()]) or None
						else:
							doc.keywords = str(v)
					elif hasattr(doc, k):
						setattr(doc, k, v)

				session.commit()

				return {
					'id': doc.id,
					'document_name': doc.document_name,
					'data_type': doc.data_type,
					'date_time': doc.date_time.strftime('%Y-%m-%d %H:%M:%S'),
					'presigned_url': None,
					'version': getattr(doc, 'version', None),
					'status': getattr(doc, 'status', None),
					'validity_end': doc.validity_end.isoformat() if doc.validity_end else None,
					'validity_year': getattr(doc, 'validity_year', None),
					'domain': getattr(doc, 'domain', None),
					'language': getattr(doc, 'language', None),
					'author': getattr(doc, 'author', None),
					'keywords': doc.keywords.split(",") if getattr(doc, 'keywords', None) else None,
				}
		except Exception as e:
			logger.error(f'Error updating document metadata: {str(e)}')
			return None

	def get_document_id_by_name(self, document_name: str) -> Optional[int]:
		"""
		Get the ID of a document by its name.

		Args:
		    document_name: Name of the document to look up

		Returns:
		    Optional[int]: ID of the document if found, None otherwise
		"""
		with self.db_manager.create_session() as session:
			try:
				# Query the Document table for the given document_name
				document = session.query(SapDocument).filter_by(document_name=document_name).first()
				if document:
					return document.id
				else:
					return None
			except Exception as e:
				logger.error(f'Error getting document ID by name: {str(e)}')
				return None

	def _add_years(self, d: date, years: int) -> date:
		"""Add years to a date, handling leap days safely."""
		try:
			return d.replace(year=d.year + years)
		except ValueError:
			# Handle Feb 29 -> Feb 28 for non-leap target years.
			return d.replace(month=2, day=28, year=d.year + years)

	def expire_documents(self, today: date | None = None) -> list[dict]:
		"""Mark expired documents as deprecated based on validity metadata.

		Expiration rules:
		1) If validity_end is set: expired when validity_end < today.
		2) If validity_end is empty: expired when (upload_date + validity_year years) < today.

		Only documents with status == 'active' are auto-deprecated.

		Returns:
		A list of dictionaries describing documents that were changed from active to deprecated.
		"""
		today = today or date.today()
		updated: list[dict] = []

		with self.db_manager.create_session() as session:
			try:
				docs = session.query(SapDocument).filter(SapDocument.status == 'active').all()
				for doc in docs:
					upload_dt = getattr(doc, 'date_time', None)
					upload_date = upload_dt.date() if upload_dt else None

					# Rule: explicit validity_end
					if doc.validity_end is not None:
						if doc.validity_end < today:
							doc.status = 'deprecated'
							updated.append({
								'id': doc.id,
								'document_name': doc.document_name,
								'previous_status': 'active',
								'expiry_rule': 'validity_end',
								'expiry_date': doc.validity_end.isoformat(),
								'upload_date': upload_date.isoformat() if upload_date else None,
								'validity_year': getattr(doc, 'validity_year', None),
								'domain': getattr(doc, 'domain', None),
								'language': getattr(doc, 'language', None),
								'author': getattr(doc, 'author', None),
							})
						continue

					# Rule: derive validity from upload date + validity_year
					if upload_date is None:
						continue

					try:
						y = int(getattr(doc, 'validity_year', None) or 1)
					except Exception:
						y = 1

					expiry_date = self._add_years(upload_date, y)
					if expiry_date < today:
						doc.status = 'deprecated'
						updated.append({
							'id': doc.id,
							'document_name': doc.document_name,
							'previous_status': 'active',
							'expiry_rule': 'validity_year',
							'expiry_date': expiry_date.isoformat(),
							'upload_date': upload_date.isoformat(),
							'validity_year': y,
							'domain': getattr(doc, 'domain', None),
							'language': getattr(doc, 'language', None),
							'author': getattr(doc, 'author', None),
						})

				if updated:
					session.commit()
				return updated

			except Exception as e:
				logger.error(f'Error expiring documents: {str(e)}')
				session.rollback()
				return []