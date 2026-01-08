"""
Pydantic models for SAP document processing API.

This module defines the request and response models for SAP document search
and document management operations, using Pydantic for data validation.
"""

from pydantic import BaseModel
from typing import List, Optional
from datetime import date

class SAPDocumentsRequest(BaseModel):
	question: str
	version: Optional[str] = None
	status: Optional[List[str]] = None
	validity_end: Optional[date] = None
	validity_year: int = 1
	domain: Optional[List[str]] = None
	language: Optional[List[str]] = None
	author: Optional[List[str]] = None
	keywords: Optional[List[str]] = None


class SAPDocumentsResponse(BaseModel):
	answer: str
	search_results: list


class UploadResponse(BaseModel):
	message: str
	doc_name: str | None


class UploadRequest(BaseModel):
	s3_key: str
	
	#new added metadata
	version: str
	validity_end: Optional[date] = None
	validity_year: int
	status: str
	domain: Optional[str] = None
	language: Optional[str] = None
	author: Optional[str] = None
	keywords: Optional[list[str]] = None

class DocumentResponse(BaseModel):
	"""
	Response model for document retrieval operations.

	This model defines the structure for responses when retrieving document
	information, including metadata.
	"""

	id: int
	document_name: str
	data_type: str
	date_time: str
	
	#new added metadata
	version: str
	validity_end: Optional[date] = None
	validity_year: int
	status: str
	domain: Optional[str] = None
	language: Optional[str] = None
	author: Optional[str] = None
	keywords: Optional[list[str]] = None

class UpdateMetadataRequest(BaseModel):
	"""Request model for updating document metadata (partial update)."""

	doc_id: int
	version: Optional[str] = None
	status: Optional[str] = None
	validity_end: Optional[date] = None
	validity_year: Optional[int] = None
	domain: Optional[str] = None
	language: Optional[str] = None
	author: Optional[str] = None
	keywords: Optional[List[str]] = None