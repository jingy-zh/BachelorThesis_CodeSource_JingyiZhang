"""
Database models for SAP document storage and retrieval.

This module defines the database schema for storing SAP documents and their
vector embeddings, facilitating semantic search capabilities across SAP
documentation.
"""
from pgvector.sqlalchemy import Vector
from sqlalchemy import TIMESTAMP, Column, ForeignKey, Integer, Text, func, String, DateTime, Date
from sqlalchemy.orm import mapped_column, relationship

from ..models.base import Base
import datetime

VECTOR_SIZE = 3072


class SapDocument(Base):
	"""
	Database model for SAP document metadata.

	This class defines the schema for storing high-level information about
	SAP documents, serving as the parent entity for associated document
	embeddings.
	"""
	__tablename__ = 'sap_documents'
	__table_args__ = {'schema': 'public'}

	id = Column(Integer, primary_key=True, autoincrement=True)
	document_name = Column(Text, nullable=False)
	data_type = Column(Text, nullable=False)
	date_time = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())

	#new metadata
	version= Column(Text, nullable=False, server_default='v1.0')
	status=Column(Text, nullable=False, server_default='active')
	validity_end = Column(Date, nullable=True)
	validity_year = Column(Integer, nullable=False, default=1)
	domain = Column(Text, nullable=True)
	keywords =Column (Text,nullable=True)
	author =Column (Text, nullable=True)
	language =Column (Text, nullable=True)
	
	
	# Establish the relationship
	embeddings = relationship('SapDocsEmbedding', order_by='SapDocsEmbedding.id', back_populates='document')


class SapDocsEmbedding(Base):
	"""
	Database model for SAP document embeddings.

	This class defines the schema for storing vector embeddings of SAP document
	content, enabling semantic search and retrieval of document sections based
	on their meaning.
	"""
	__tablename__ = 'sap_document_embeddings'
	__table_args__ = {'schema': 'public'}

	id = Column(Integer, primary_key=True, autoincrement=True)
	embedding = mapped_column(Vector(VECTOR_SIZE))
	raw_data = Column(Text)
	summary = Column(Text)
	data_type = Column(Text)
	# Corrected foreign key reference with schema awareness
	document_id = Column(Integer, ForeignKey(f'public.{SapDocument.__tablename__}.id', ondelete='CASCADE'))

	# Establish the relationship
	document = relationship('SapDocument', back_populates='embeddings')
