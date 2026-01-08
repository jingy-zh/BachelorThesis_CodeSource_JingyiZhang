"""
Feedback data models for the conversation system.

This module defines Pydantic models for handling user feedback collection,
querying, and processing within the conversation system.
"""

from pydantic import BaseModel

from .cova_schema import CovaMessage


class SaveFeedbackRequest(BaseModel):
	"""
	Request model for saving user feedback.

	This model encapsulates the data needed to store user feedback about a
	conversation session, including the conversation history, rating score,
	and optional comments.
	"""

	session_id: str
	conversation: list[CovaMessage]
	rating_score: int
	comments: str | None
	issue_type: str | None = None