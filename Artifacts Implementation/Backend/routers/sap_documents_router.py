"""FastAPI router for SAP document management and search.

This module provides endpoints for managing SAP documents, including
searching document contents, uploading new documents, analyzing document content,
listing available documents, and deleting documents from the system.
"""

# ruff: noqa: B008
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from be_module.src.dependencies import DB_MANAGER, env_creds
from be_module.src.schemas.sap_documents_schema import (
	DocumentResponse,
	SAPDocumentsRequest,
	SAPDocumentsResponse,
	UploadRequest,
	UploadResponse,
)
from be_module.src.services.sap_documents.main import DocumentSearch, SAPDocuments

router = APIRouter()

creds = {'azure_endpoint': env_creds['AZURE_OPENAI_ENDPOINT'], 'api_key': env_creds['AZURE_OPENAI_API_KEY']}
sap_documents_search = DocumentSearch(db_manager=DB_MANAGER, creds=creds)
sap_documents = SAPDocuments(document_search=sap_documents_search)


def get_sap_document_service() -> SAPDocuments:
	"""Dependency for the SAP Documents service.

	Returns:
		SAPDocuments: An initialized SAP Documents service instance.
	"""
	return sap_documents


def process_question(request: SAPDocumentsRequest) -> str:
	"""Dependency to extract and process the question from the request.

	Args:
		request (SAPDocumentsRequest): The incoming request object.

	Returns:
		str: The extracted question.
	"""
	return request.question


async def stream_response(stream_response: AsyncGenerator) -> AsyncGenerator:
	"""Dependency for streaming response handling.

	Args:
		stream_response (AsyncGenerator): The generator providing response chunks.

	Yields:
		str: JSON-formatted response chunks with newline separators.
	"""
	async for item in stream_response:
		yield json.dumps(item)
		yield '\n'

from be_module.src.schemas.sap_documents_schema import (
    SAPDocumentsRequest,
    SAPDocumentsResponse,
    UploadResponse,
    DocumentResponse,
	 UpdateMetadataRequest,
)
@router.post('/sap_documents/stream')
async def get_sap_documents_stream(
	request: SAPDocumentsRequest,
	sap_docs_service: SAPDocuments = Depends(get_sap_document_service),
):

		filters: dict = {}
		if getattr(request, 'status', None) is not None:
			filters['status'] = request.status
		if getattr(request, 'domain', None) is not None:
			filters['domain'] = request.domain
		if getattr(request, 'language', None) is not None:
			filters['language'] = request.language

		filters_arg = filters or None
		response = sap_docs_service.execute_stream(
			question=request.question,
			filters=filters_arg,
		)

		return StreamingResponse(stream_response(response), media_type='text/event-stream')

@router.post('/sap_documents/enforce_validity', response_model=dict)
async def enforce_validity(
	sap_docs_service: SAPDocuments = Depends(get_sap_document_service),
):
	"""Run document validity enforcement.

	This endpoint can be used by administrators or scheduled jobs to trigger an
	immediate validity check and auto-deprecate expired documents.
	"""
	updated = sap_docs_service.enforce_document_validity()
	return {'message': 'Validity enforcement executed', 'updated_count': len(updated), 'updated': updated}


@router.get('/sap_documents/check_uploadable', response_model=UploadResponse)
async def check_doc_uploadable(
	file_name: str,
):
	"""Check if a new SAP document can be uplaoded to the system

	Returns:
		UploadResponse: A response containing a success message if the document can be uploaded

	"""
	try:
		_ = await sap_documents.check_document_uploadable(file_name)
		return UploadResponse(message='Document can be uploaded', doc_name=file_name)
	except HTTPException as e:
		if e.status_code == 400:
			return UploadResponse(message=str(e.detail), doc_name=None)
		else:
			return e


@router.post('/sap_documents/upload', response_model=UploadResponse)
async def upload_document(
	request: UploadRequest,
):
	"""Upload a new SAP document to the system.

	Processes and stores an uploaded document file in the SAP documents database.

	Returns:
		UploadResponse: A response containing a success message and the document name.
	"""
	try:
		doc_name = await sap_documents.upload_document(s3_key=request.s3_key,
												 version=request.version,
												 status=request.status,
												 validity_end=request.validity_end,
												 validity_year=request.validity_year,
												 domain=request.domain,
												 language=request.language,
												 author=request.author,
												 keywords=request.keywords,
												 )
		return UploadResponse(message='Document uploaded successfully', doc_name=doc_name)
	except HTTPException as e:
		if e.status_code == 400:
			return UploadResponse(message=e.detail, doc_name=None)
		raise e

async def analyze_uploaded_document(file: UploadFile, sap_docs_service: SAPDocuments = Depends(get_sap_document_service)) -> dict:
	"""Dependency for analyzing document content.

	Args:
		file (UploadFile): The document file to analyze.
		sap_docs_service (SAPDocuments): The SAP Documents service.

	Returns:
		dict: The document analysis results.
	"""
	return await sap_docs_service.analyze_document(file)


@router.post('/sap_documents/analyze')
async def analyze_document(analysis: dict = Depends(analyze_uploaded_document)):
	return {'message': 'Document analyzed successfully', 'analysis': analysis}


def get_document_list(
	limit: int = 200,
	offset: int = 0,
	sap_docs_service: SAPDocuments = Depends(get_sap_document_service),
) -> list[DocumentResponse]:
	"""Dependency for retrieving the list of available documents."""
	return sap_docs_service.list_documents(limit=limit, offset=offset)


@router.get('/sap_documents/list', response_model=list[DocumentResponse])
async def list_documents(documents: list[DocumentResponse] = Depends(get_document_list)):
	return documents


def delete_doc_by_id(doc_id: int, sap_docs_service: SAPDocuments = Depends(get_sap_document_service)) -> bool:
	"""Dependency for document deletion.

	Args:
		doc_id (int): The unique identifier of the document to delete.
		sap_docs_service (SAPDocuments): The SAP Documents service.

	Returns:
		bool: True if deletion was successful, False otherwise.

	Raises:
		HTTPException: If the document with the specified ID is not found.
	"""
	deleted = sap_docs_service.delete_document(doc_id)
	if not deleted:
		raise HTTPException(status_code=404, detail='Document not found')
	return deleted


@router.delete('/sap_documents/delete', response_model=dict)
async def delete_document(_: bool = Depends(delete_doc_by_id)):
	"""Delete a specific SAP document from the system.

	Removes a document from the system based on its unique identifier.

	Returns:
		dict: A success message confirming document deletion.
	"""
	return {'message': 'Document deleted successfully'}



@router.patch('/sap_documents/metadata', response_model=DocumentResponse)
async def update_document_metadata(
	request: UpdateMetadataRequest,
	sap_docs_service: SAPDocuments = Depends(get_sap_document_service),
):
	"""Update SAP document metadata without re-uploading the file."""
	data = request.model_dump(exclude_unset=True)
	doc_id = int(data.pop('doc_id'))
	updated = sap_docs_service.update_document_metadata(doc_id=doc_id, updates=data)
	return updated

@router.get("/sap_documents/trace")
async def sap_docs_trace(question: str):
    return sap_documents.trace(question=question, top_k=5)
