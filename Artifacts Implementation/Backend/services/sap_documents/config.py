import os

from ...models.sap_documents_model import VECTOR_SIZE, SapDocsEmbedding, SapDocument


class SAPDocumentsConfig:
	"""
	Configuration settings for SAP document processing and storage.

	This class centralizes configuration values for database tables,
	document processing parameters, embedding models, LLM settings,
	batch processing, file paths, and S3 storage locations.
	Environment-specific settings are handled with conditional logic.
	"""

	# Table configuration
	TABLE_DOCUMENTS: str = SapDocument.__tablename__
	TABLE_EMBEDDINGS: str = SapDocsEmbedding.__tablename__

	# Document processing configuration
	MAX_IMAGE_SIZE: tuple = (800, 600)
	MIN_WIDTH: int = 100
	MIN_HEIGHT: int = 100

	# Text splitting configuration
	CHUNK_SIZE: int = 4000
	CHUNK_OVERLAP: int = 500

	# Embedding model configuration
	if os.getenv('ENV') == 'production':
		DEPLOYMENT_NAME: str = 'cov-sdaiagent-france-prod-embedding-3l'
	else:
		DEPLOYMENT_NAME: str = 'cov-sdaiagent-france-test-text-embedding-3-large'
	EMBEDDING_MODEL: str = 'text-embedding-3-large'
	VECTOR_SIZE: int = VECTOR_SIZE

	# LLM configuration
	LLM_MODEL_ID: str = 'global.anthropic.claude-sonnet-4-20250514-v1:0'
	LLM_TEMPERATURE: int = 0
	REGION_NAME: str = 'us-east-1'

	# Batch processing configuration
	BATCH_SIZE: int = 200

	# File paths
	INPUT_FOLDER: str = './data'
	OUTPUT_FOLDER: str = './output'
	IMAGE_FOLDER: str = './images'
	TABLE_FOLDER: str = './tables'

	# S3 configuration
	S3_SAP_DOCS: str = os.getenv('S3_SAP_DOCS', default='vsda-sap-documents-test')
	S3_IMAGE_CACHE: str = os.getenv('S3_IMAGE_CACHE', default='vsda-sap-docs-image-cache')

	#Hybrid retrieval configuration (Vector + BM25 gate)
	# - HYBRID_RETRIEVAL: enable/disable hybrid gating
	# - ALLOW_VECTOR_FALLBACK_WHEN_BM25_EMPTY: if False, no BM25 hit => return no results (safer, less hallucination)
	# - FTS_LANGUAGE: PostgreSQL FTS configuration ('english'|'simple' etc.)
	# - *_MULT / *_MIN: how many candidates to fetch before intersecting
	HYBRID_RETRIEVAL: bool = os.getenv('HYBRID_RETRIEVAL', 'true').lower() in ('1','true','yes','y')
	ALLOW_VECTOR_FALLBACK_WHEN_BM25_EMPTY: bool = os.getenv('ALLOW_VECTOR_FALLBACK_WHEN_BM25_EMPTY', 'false').lower() in ('1','true','yes','y')
	FTS_LANGUAGE: str = os.getenv('FTS_LANGUAGE', default='english')
	HYBRID_VEC_MULT: int = int(os.getenv('HYBRID_VEC_MULT', default='15'))
	HYBRID_BM25_MULT: int = int(os.getenv('HYBRID_BM25_MULT', default='15'))
	HYBRID_VEC_MIN: int = int(os.getenv('HYBRID_VEC_MIN', default='50'))
	HYBRID_BM25_MIN: int = int(os.getenv('HYBRID_BM25_MIN', default='50'))

	# Soft fallback when BM25 yields no hits:
	# - If ALLOW_VECTOR_FALLBACK_WHEN_BM25_EMPTY is false, vector-only answers are still allowed
	#   when the top vector match is very strong (cosine_distance below threshold).
	HYBRID_SOFT_FALLBACK: bool = os.getenv('HYBRID_SOFT_FALLBACK', 'true').lower() in ('1','true','yes','y')
	HYBRID_FALLBACK_MAX_DISTANCE: float = float(os.getenv('HYBRID_FALLBACK_MAX_DISTANCE', default='0.28'))

	# FTS query mode: 'websearch' (default) or 'websearch' (PostgreSQL websearch_to_tsquery)
	FTS_QUERY_MODE: str = os.getenv('FTS_QUERY_MODE', default='websearch')


sap_documents_config = SAPDocumentsConfig()
