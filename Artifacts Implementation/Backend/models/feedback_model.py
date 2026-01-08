import datetime as dt
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from ..models.base import Base


class Feedback(Base):
	"""
	Database model for user feedback.

	This class defines the structure for storing user feedback data,
	including session information, numerical ratings, textual comments,
	and submission timestamps for quality assessment and improvement.
	"""
	__tablename__ = 'feedback'
	id = Column(Integer, primary_key=True)
	session_id = Column(String)
	rating_score = Column(Integer)
	comments = Column(String)
	timestamp = Column(DateTime, default=datetime.now(dt.UTC))
