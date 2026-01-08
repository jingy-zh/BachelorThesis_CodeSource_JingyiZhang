"""FastAPI router for feedback management endpoints.

This module provides FastAPI endpoints for handling user feedback operations,
including retrieving, saving, and summarizing feedback, as well as managing
user preference profiles based on feedback analysis.
"""

import asyncio

from requests import request  # noqa: I001

from fastapi import APIRouter

from be_module.src.dependencies import DB_MANAGER, INCIDENT_INFO_EXTRACTOR
from be_module.src.schemas.chat_schema import ResponseQuery
from be_module.src.schemas.feedback_schema import SaveFeedbackRequest
from be_module.src.schemas.cova_schema import CovaMessage
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from loguru import logger

router = APIRouter()

OUTDATED_KNOWLEDGE_NOTIFY_EMAIL = os.getenv('OUTDATED_KNOWLEDGE_NOTIFY_EMAIL', 'replace_by_real_email_address')
NOTIFY_FROM_EMAIL = os.getenv('NOTIFY_FROM_EMAIL', 'no-reply@solver.local')

def _send_email(*, to_email: str, subject: str, body: str) -> bool:
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
		logger.warning('SMTP_HOST is not configured; skip sending notification email.')
		return False

	smtp_port = int(os.getenv('SMTP_PORT', '587'))
	smtp_user = os.getenv('SMTP_USER')
	smtp_password = os.getenv('SMTP_PASSWORD')
	use_tls = os.getenv('SMTP_USE_TLS', 'true').strip().lower() in {'1', 'true', 'yes', 'y'}

	msg = EmailMessage()
	msg['From'] = NOTIFY_FROM_EMAIL
	msg['To'] = to_email
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
		logger.error(f'Failed to send email notification: {e}')
		return False


def _extract_last_turn(conversation) -> tuple[str | None, str | None]:
	"""Best-effort extraction of the last user and assistant messages."""
	last_user = None
	last_assistant = None
	try:
		for msg in conversation[::-1]:
			role = None
			content = None
			if isinstance(msg, dict):
				role = msg.get('role')
				content = msg.get('content')
			else:
				role = getattr(msg, 'role', None)
				content = getattr(msg, 'content', None)

			if role == 'assistant' and last_assistant is None:
				last_assistant = content
			if role == 'user' and last_user is None:
				last_user = content

			if last_user is not None and last_assistant is not None:
				break
	except Exception:
		return None, None
	return last_user, last_assistant


def get_db_manager():
	"""
	Dependency that provides the database manager instance.

	Returns:
		The database manager instance for feedback operations.
	"""
	return DB_MANAGER


def get_incident_info_extractor():
	"""
	Dependency that provides the incident information extractor.

	Returns:
		The incident information extractor for generating retrospective summaries.
	"""
	return INCIDENT_INFO_EXTRACTOR


@router.get('/get_feedback', response_model=ResponseQuery)
async def get_feedback(session_id: str):
	feedback_data = DB_MANAGER.get_feedback(session_id)
	return ResponseQuery(response=feedback_data)


@router.get('/get_feedback_summary', response_model=ResponseQuery)
async def get_feedback_summary(date: str):
	feedback_summary = DB_MANAGER.query_feedback_summary(date)
	return ResponseQuery(response=feedback_summary)


@router.post('/save_feedback')
async def save_feedback(request: SaveFeedbackRequest):
	"""Save user feedback and asynchronously generate a retrospective summary.

	Saves the user's feedback rating and comments to the database, then
	asynchronously processes the conversation to extract a retrospective summary.

	Args:
		request (SaveFeedbackRequest): The request containing feedback data including
			session_id, rating_score, comments, and the conversation history.

	Returns:
		ResponseQuery: A success message indicating the feedback was saved.
	"""
	DB_MANAGER.save_feedback(request.session_id, request.rating_score, request.comments)
	# Notify the responsible role when the user flags the answer as outdated knowledge.
	if (request.issue_type or '').strip().lower() == 'outdated_knowledge':
		last_user, last_assistant = _extract_last_turn(request.conversation)
		timestamp = datetime.now(timezone.utc).isoformat()
		subject = 'Solver feedback: Answer contains outdated knowledge'
		body = (
			f'Timestamp (UTC): {timestamp}\n'
			f'Session ID: {request.session_id}\n'
			f'Rating score: {request.rating_score}\n'
			f'User comments: {request.comments or ""}\n\n'
			f'Last user message:\n{last_user or ""}\n\n'
			f'Last assistant message:\n{last_assistant or ""}\n'
		)
		_send_email(to_email=OUTDATED_KNOWLEDGE_NOTIFY_EMAIL, subject=subject, body=body)

	asyncio.create_task(
		get_retro_summary(request.session_id, request.conversation, request.rating_score, request.comments, DB_MANAGER, INCIDENT_INFO_EXTRACTOR)
	)

	return ResponseQuery(response='Successfully saved feedback')


async def get_retro_summary(
	session_id: str, conversation: list[CovaMessage], rating_score: int, feedback_comments: str, db_manager=None, incident_extractor=None
):
	retro_summary_dict = await incident_extractor.get_retro_summary(conversation, rating_score, feedback_comments)
	db_manager.update_slots(session_id, retro_summary_dict)
	return 'success'
