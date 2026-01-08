"""
Streamlit-based frontend interface.

This module provides a web interface for interacting with the AI assistant,
including chat interfaces, document management, feedback window, feedback collection, and real-time communication with backend services.
"""

import json
import os
import sys
import requests
from datetime import datetime, timedelta
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from dotenv import load_dotenv
from streamlit_date_picker import PickerType, date_range_picker
from streamlit_feedback import streamlit_feedback
from utils.streamlit_utils import call_api_data
import requests
import streamlit as st
import re

sys.path.append('.')
sys.path.append('..')
sys.path.append('../src')

from fe_module.utils.streamlit_utils import (
	authorise_user,
	call_api,
	display_feedback,
	display_feedback_sumamry,
	display_user_background,
	get_sessions,
	handle_stream_response,
	process_session_token,
	process_uploaded_image,
	update_key,
	visualize_sessions_info,
)
from shared_access.utils import CredentialsManager, ReleaseNoteS3, S3Handler, ServiceNowClient

@st.cache_data(ttl=30, show_spinner=False)
def fetch_sap_docs_page(be_endpoint: str, headers_items: tuple[tuple[str, str], ...], limit: int, offset: int):
    """
    Fetch one page of SAP docs. Requires backend support for limit/offset.
    Falls back gracefully if backend ignores params (still helps reduce reruns via cache).
    """
    url = f"{be_endpoint.rstrip('/')}/sap_documents/list"
    headers = dict(headers_items)
    params = {"limit": int(limit), "offset": int(offset)}
    r = requests.get(url, headers=headers, params=params, timeout=(3.05, 15))
    r.raise_for_status()
    data = r.json()
    return data or []

load_dotenv()

RATING_MAPS = {'faces': {'😞': 1, '🙁': 2, '😐': 3, '🙂': 4, '😀': 5}}

_FEEDBACK_ISSUE_OPTIONS = {
	'Select an issue type (optional)': None,
	'Hallucinated answer': 'hallucinated_answer',
	'Incomplete answer': 'answer_incomplete',
	'Inaccurate answer': 'answer_inaccurate',
	'Answer contains outdated knowledge': 'outdated_knowledge',
}
# ----------------------------
# SAP Documents image rendering
# ----------------------------
_S3_KEY_PATTERN = re.compile(r"\[\(VSDA_S3_KEY:([^\)]+)\)\]")

def _safe_get_presigned_url(handler, key: str, *, expiration: int = 3600) -> str | None:
	"""Return a presigned URL for a given S3 key, never raising exceptions."""
	if not handler or not key:
		return None
	try:
		return handler.get_presigned_url(key, expiration=expiration)
	except TypeError:
		try:
			return handler.get_presigned_url(key)
		except Exception:
			return None
	except Exception:
		return None

def render_text_with_sap_images(text: str) -> str:
	"""Replace VSDA_S3_KEY placeholders with markdown image links (never raises)."""
	if not text or 'VSDA_S3_KEY:' not in text:
		return text

	def _repl(match: re.Match) -> str:
		key = (match.group(1) or '').strip()
		url = _safe_get_presigned_url(IMG_HANDLER, key, expiration=86400)
		if not url:
			return match.group(0)
		alt = key.split('/')[-1]
		return f"![{alt}]({url})"

	try:
		return _S3_KEY_PATTERN.sub(_repl, text)
	except Exception:
		return text


def load_config():
	"""
	Load configuration settings from environment variables and YAML files.

	Initializes global variables including backend endpoints, model configurations,
	credentials, and service clients for S3 and ServiceNow.
	"""
	global BE_ENDPOINT, FE_CFG, SNOW_CLIENT, S3_HANDLER, RELEASE_NOTES, WEBSOCKET_URL, IMG_HANDLER, S3_SAP_DOCS_HANDLER

	with open('./fe_module/configs/fe_config.yaml') as f:
		FE_CFG = yaml.load(f, Loader=yaml.SafeLoader)

	BE_ENDPOINT = os.getenv('BE_ENDPOINT', 'http://127.0.0.1:8502')
	creds = CredentialsManager().fetch_env_credentials()
	WEBSOCKET_URL = creds['WEBSOCKET_ENDPOINT']
	SNOW_CLIENT = ServiceNowClient((creds['SNOW_USER'], creds['QA_SNOW_PWD'], creds['PROD_SNOW_PWD']))
	S3_HANDLER = S3Handler(creds['S3_BUCKET_NAME'])
	S3_SAP_DOCS_HANDLER = S3Handler(creds.get('S3_SAP_DOCS', 'vsda-sap-documents-test'))
	RELEASE_NOTES = ReleaseNoteS3(creds['RELEASE_BUCKET_NAME'])
	IMG_HANDLER = S3Handler(FE_CFG['sap_images_cache_bucket'])


def init_session():
	st.session_state.session_id = str(uuid4())
	st.session_state.messages = []
	st.session_state.uploader_key = 0
	st.session_state.prompt_config = None
	st.session_state.process_running = False
	st.session_state.feedback_issue_type = None
	st.session_state.smart_suggestions = []
	st.session_state.choosen_suggestion = None


def display_slots(user_slots: dict):
	for slot in user_slots:
		st.write(f'{slot} : {user_slots[slot]}')


def display_msg(messages: list[dict]):
	"""
	Display chat messages in the Streamlit interface.

	Handles different message types (system, user, assistant) and renders
	references including images with proper formatting.

	Args:
		messages: List of message dictionaries with role, content, and optional references
	"""
	for message in messages:
		if message['role'] in ['system_message', 'liveagentdisconnect']:
			st.markdown(f'*{message["content"]}*')
		else:
			with st.chat_message(message['role']):
				content = message.get('content', '')
				if isinstance(content, str) and 'VSDA_S3_KEY:' in content:
					st.markdown(render_text_with_sap_images(content))
				else:
					st.write(content)


				if message['role'] == 'user' and (user_refs := message.get('references')):
					for ref in user_refs:
						key = ref.get('ref')
						if not key:
							continue
						img_presigned_url = _safe_get_presigned_url(S3_HANDLER, key, expiration=86400)
						if img_presigned_url:
							st.image(img_presigned_url, use_column_width=True)

				if message['role'] == 'assistant' and (assist_refs := message.get('references')):
					with st.expander('Learn more about this response'):
						st.write(assist_refs)


def render_sidebar():
	"""
	Render the sidebar with action buttons and controls.

	Creates buttons for clearing chat, providing feedback, and accessing
	additional features like solver info.
	"""
	with st.sidebar:
		if st.button('Clear Chat & Give Feedback', type='primary', key='clear_chat'):
			show_feedback_prompt()
		if st.session_state.is_admin is True:  # noqa: SIM102
			if st.button(
				'Solver Info',
				disabled=st.session_state.process_running,
				key='solver_info',
			):
				solver_info()


# Smart suggestions UI removed per request


def set_choosen_suggestion(suggestion: str):
	st.session_state.choosen_suggestion = suggestion


def normal_chat_interface():
	return chat_interface()


def ma_chat_interface():
	return chat_interface('sendMessageMA')


def chat_interface(service: str = 'sendMessage'):
	"""
	Render the main chat interface for interacting with Solver.

	Handles message display, user input processing, file uploads, smart suggestions,
	and streaming responses from the backend service.

	Args:
		service: The backend service endpoint to use for sending messages (default: 'sendMessage')
	"""
	if service == 'sendMessage':
		display_user_background()

	title = 'Solver' if service == 'sendMessage' else 'Solver Multi-Agents Version'
	st.title(title)
	if service == 'service':
		st.write(FE_CFG['intro_message'].format(user_name=st.session_state.user_background['name']))
	display_msg(st.session_state.messages)
	# Accept user input
	with st.sidebar:
		uploaded_pics = st.file_uploader(
			'Choose one or more images',
			accept_multiple_files=True,
			type=['png', 'jpg', 'jpeg'],
			key=f'uploader_{st.session_state.uploader_key}',
		)
	# Smart suggestions UI removed
	prompt = st.chat_input('What is up?', disabled=st.session_state.process_running) or st.session_state.choosen_suggestion
	with st._bottom:
		st.info('Solver can make mistakes. Consider verifying important information.', icon=':material/warning:')

	if prompt:
		st.session_state.process_running = True
		references = []
		if st.session_state.choosen_suggestion is not None:
			prompt = st.session_state.choosen_suggestion
			st.session_state.choosen_suggestion = None
			references.append({'title': 'message_source', 'content': 'suggestion'})

		st.session_state.smart_suggestions = []

		new_messages = []
		if uploaded_pics:
			with st.spinner('Uploading images...'):
				for i, uploaded_pic in enumerate(uploaded_pics):
					file_extention = uploaded_pic.name.split('.')[-1].lower()
					image_key = f'{st.session_state.session_id}_{st.session_state.uploader_key}_{i}.{file_extention}'
					img_message = process_uploaded_image(uploaded_pic, image_key)
					references.append(img_message)
				update_key()

		# Add user message to chat history
		new_messages.append({'role': 'user', 'content': prompt, 'references': references})
		st.session_state.messages.extend(new_messages)
		with st.spinner('Saving messages...'):
			call_api(
				data={
					'session_id': st.session_state.session_id,
					'cwid': st.session_state.user_background['cwid'],
					'new_messages': new_messages,
				},
				endpoint=BE_ENDPOINT,
				service='save_new_messages',
				headers=st.session_state.be_headers,
			)
		st.rerun()

	if st.session_state.messages and st.session_state.messages[-1]['role'] == 'user':
		# Display assistant response in chat message container
		with st.chat_message('assistant'):  # Do the spinner
			render_sidebar()
			with st.spinner('Solver is thinking...'):
				response_container = st.empty()

				data = {
					'messages': st.session_state.messages,
				}

				assistant_response = call_api(data=data, endpoint=BE_ENDPOINT, service=service, headers=st.session_state.be_headers, stream=True)
				bot_response = ''
				references = []
				expander = st.expander('Learn more about this response')

				for chunk in handle_stream_response(assistant_response):
					if isinstance(chunk, str):
						bot_response += chunk
						response_container.write(bot_response)
					elif isinstance(chunk, dict):
						if (message_type := chunk.get('type')) == 'references':
							references.extend(chunk['content'])
							expander.write(references)
						elif message_type == 'suggestions':
							st.session_state.smart_suggestions.extend(chunk['content'])

			with st.spinner('Saving assistant response...'):
				assistant_messages = [{'role': 'assistant', 'content': bot_response, 'references': references}]
				call_api(
					data={
						'session_id': st.session_state.session_id,
						'cwid': st.session_state.user_background['cwid'],
						'new_messages': assistant_messages,
					},
					endpoint=BE_ENDPOINT,
					service='save_new_messages',
					headers=st.session_state.be_headers,
				)

			st.session_state.messages.extend(assistant_messages)
			st.session_state.process_running = False
			st.rerun()
	else:
		render_sidebar()


def sap_docs_test_page():
	"""
	Render the SAP Documents testing interface.

	Provides a chat interface specifically for testing document Q&A functionality
	with SAP documents, including document reference display.
	"""
	st.title('SAP Documents Q&A testing')

	def clear_screen():
		st.session_state.sap_messages = []

	with st.sidebar:
		st.file_uploader('Choose one or more images', accept_multiple_files=True, type=['png', 'jpg'])
		st.button('New Chat', on_click=clear_screen, type='primary')

		st.markdown ('---')
		st.header('Search Filters')

		status_filter = st.multiselect(
			'Status', 
			options=['active', 'deprecated', 'archived'],
			default = ['active'],
			key='sap_filter_status',
		)
		domain_filter = st.multiselect(
			'Domain', 
			options=['BW','GTS','FI','MDM','SD','PPE','QM','PM','IBP','PRO','OM','MTS','FI-AA','FI-GL','Other'],
			key='sap_filter_domain',
		)
		language_filter = st.multiselect(
			'Language', 
			options=['EN', 'DE','FR','ES','IT','CN','JP','Other'],
			key='sap_filter_language',
		)

	if 'sap_messages' not in st.session_state:
		st.session_state.sap_messages = []

	display_msg(st.session_state.sap_messages)

	prompt = st.chat_input()
	if prompt:
		st.session_state.sap_messages.append({'role': 'user', 'content': prompt})
		with st.chat_message('user'):
			st.write(prompt)

			# Prepare filters
			data = {'question': prompt}
			if status_filter:
				data['status'] = status_filter
			if domain_filter:
				data['domain'] = domain_filter
			if language_filter:
				data['language'] = language_filter

		api_response = call_api(
			data=data, endpoint=BE_ENDPOINT, service='sap_documents/stream', headers=st.session_state.be_headers, stream=True
		)

		with st.chat_message('assistant'):
			response_container = st.empty()
			bot_response = ''
			references = []

			for chunk in handle_stream_response(api_response):
				if isinstance(chunk, str):
					bot_response += chunk
					response_container.markdown(render_text_with_sap_images(bot_response))
				elif isinstance(chunk, list):
					references.extend(chunk)
					st.markdown('#### Source Details')
					for i, doc in enumerate(references):
						display_document_details(i, doc)

		st.session_state.sap_messages.append({'role': 'assistant', 'content': bot_response})


def display_document_details(index, doc):
	"""
	Display detailed information about a document reference.

	Creates an expandable section with document metadata, summary, and content preview.

	Args:
		index: Index of the document in the references list
		doc: Document metadata dictionary
	"""
	with st.expander(f"Document {index + 1}: {doc['document_name']}"):
		st.markdown(f"**Data Type:** {doc['data_type']}")
		st.markdown(f"**Score:** {doc['score']}")
		st.markdown('**Summary:**')
		st.markdown(doc['summary'])
		st.markdown('**Raw Data:**')
		if doc['data_type'].lower() == 'image':
			presigned_url = IMG_HANDLER.get_presigned_url(doc['raw_data'])
			st.write(f'![Image]({presigned_url})')
		else:
			st.markdown(doc['raw_data'])


def chat_history():
	"""
	Render the chat history interface.

	Allows users to browse and review their previous chat sessions,
	including ticket information, feedback, and message history.
	"""
	st.header('Your Chat History')
	if st.session_state.get('is_admin', False):
		all_sessions = call_api(data={}, endpoint=BE_ENDPOINT, service='query_all_session_ids', method='get', headers=st.session_state.be_headers)[
			'response'
		]
	else:
		all_sessions = call_api(
			data={
				'by': 'cwid',
				'filter_value': st.session_state.user_background.get('cwid', None),
			},
			method='get',
			endpoint=BE_ENDPOINT,
			service='query_session_ids_with_filter',
			headers=st.session_state.be_headers,
		)['response']

	labels = list(range(len(all_sessions)))
	session_choice_label = st.selectbox('Select a session', options=labels, index=0)
	session_choice = all_sessions[labels.index(session_choice_label)]

	st.subheader('Ticket Information')
	slots_data = call_api(
		data={'session_id': session_choice}, endpoint=BE_ENDPOINT, service='get_slots', method='get', headers=st.session_state.be_headers
	)['response']
	if st.session_state.get('is_admin'):
		display_slots(slots_data)
	feedback_data = call_api(
		data={'session_id': session_choice}, endpoint=BE_ENDPOINT, service='get_feedback', method='get', headers=st.session_state.be_headers
	)['response']
	display_feedback(feedback_data)

	st.subheader('Solver')
	chat_session = call_api(
		data={'session_id': session_choice}, endpoint=BE_ENDPOINT, service='chat_history', method='get', headers=st.session_state.be_headers
	)['response']
	display_msg(chat_session)


@st.dialog('Feedback', width='large')
def show_feedback_prompt():
	"""
	Display a dialog for collecting user feedback.

	Creates a form with rating options and text input for detailed feedback.

	Returns:
		The collected feedback data
	"""
	st.write(FE_CFG['feedback_prompt'])

	issue_label = st.selectbox(
		'Issue type',
		options=list(_FEEDBACK_ISSUE_OPTIONS.keys()),
		index=0,
		key=f'feedback_issue_type_{st.session_state.session_id}',
	)
	st.session_state.feedback_issue_type = _FEEDBACK_ISSUE_OPTIONS.get(issue_label)

	feedback = streamlit_feedback(
		feedback_type='faces',
		on_submit=_submit_feedback,
		key=f'feedback_{st.session_state.session_id}',
		optional_text_label='Please provide some more information',
		max_text_length=2000,
		align='flex-start',
	)
	return feedback


def _submit_feedback(feedback_data: dict):
	"""
	Submit user feedback to the backend.

	Processes and sends the feedback data to the appropriate backend services
	and resets the session state.

	Args:
		feedback_data: Dictionary containing rating score and optional text feedback
	"""
	data = {
		'session_id': st.session_state.session_id,
		'conversation': st.session_state.messages,
		'rating_score': RATING_MAPS['faces'][feedback_data['score']],
		'comments': feedback_data.get('text'),
		# Optional issue taxonomy selected by the user in the feedback dialog.
		'issue_type': st.session_state.get('feedback_issue_type'),
	}
	call_api(data=data, endpoint=BE_ENDPOINT, service='save_feedback', headers=st.session_state.be_headers)

	init_session()
	st.rerun()


@st.dialog('Solver Info', width='large')
def solver_info():
	st.header('How does Solver work?')
	st.markdown(FE_CFG['solver_arch_info'])

def manage_sap_documents():
	"""
	Render the interface for managing SAP documents.

	Provides controls for uploading, analyzing, and deleting SAP documents,
	along with a table view of existing documents.
	"""
	st.title('Manage SAP Documents')

	with st.sidebar:
		# Analyze an Upload Document
		st.header('Analyze an Upload Document')
		uploaded_file = st.file_uploader('Choose a file', type=['docx'], key='sap_docs_uploader')
		
		# Add new metadata
		if uploaded_file is not None:
			st.markdown('---')
			st.subheader('Document Metadata')
			version = st.text_input('Version', value='1.0')
			status= st.selectbox('Status', ['active', 'deprecated', 'archived'], index=0)
			validity_year = st.number_input(
				'Validity period (years)',
				min_value=1,
				max_value=20,
				value=1,
				step=1)
			use_validity_end = st.checkbox('Set validity end date', value=False)
			validity_end = None
			if use_validity_end:
				validity_end = st.date_input('Validity end date')
			domain = st.selectbox(
				'Domain',
				options=['BW','GTS','FI','MDM','SD','PPE','QM','PM','IBP','PRO','OM','MTS','FI-AA','FI-GL','Other'],
				index=0,
			)
			language= st.selectbox(
				'Language',
				options=['EN', 'DE','FR','ES','IT','CN','JP','Other'],
				index=0,
			)
			author= st.text_input('Author', value= '')
			keywords_str = st.text_input('Keywords (separated by commas)', value='')

			col1, col2 = st.columns(2)

			#Analyze Document button
			with col1:
				if 'analyze_clicked' not in st.session_state:
					st.session_state.analyze_clicked = False

				def handle_analyze_click():
					st.session_state.analyze_clicked = True
					files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
					
					response = call_api(
						data={},
						endpoint=BE_ENDPOINT,
						service='sap_documents/analyze',
						method='POST',
						files=files,
						headers=st.session_state.be_headers,
					)
					document_analysis(response.get('analysis'))
					st.session_state.analyze_clicked = False
				
				st.button(
					 'Analyze Document',
                    on_click=handle_analyze_click,
                    disabled=st.session_state.analyze_clicked,
				)
			
			#Upload Document button
			with col2:
				if 'upload_clicked' not in st.session_state:
					st.session_state.upload_clicked = False

				def handle_upload_click():
					st.session_state.upload_clicked = True
					files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}

					check_uploadable = call_api(
						data={'file_name': uploaded_file.name},
						endpoint=BE_ENDPOINT,
						service='sap_documents/check_uploadable',
						method='GET',
						files=files,
						headers=st.session_state.be_headers,
					)
					if check_uploadable.get('message') == 'Document can be uploaded':
						s3_key = S3_SAP_DOCS_HANDLER.upload_document(
							file=uploaded_file,
							file_name=uploaded_file.name,
						)
						payload = {
							's3_key': s3_key,
							'version': version,
							'status': status,
							'validity_year': int(validity_year),
							'domain': None if domain == 'Other' else domain,
							'language': language,
							'author': author,
							'keywords': [k.strip() for k in keywords_str.split(',') if k.strip ()],
						}

						if validity_end is not None:
							payload['validity_end'] = validity_end.isoformat()

						response = call_api(
							data=payload,
							endpoint=BE_ENDPOINT,
							service='sap_documents/upload',
							method='POST',
							headers=st.session_state.be_headers,
							timeout=300,
						)
						if response.get('message') == 'Document uploaded successfully':
							st.success('Document uploaded successfully!')
						else:
							st.error('Failed to upload document, Error: ' + str(response.get('message')))
					else:
						st.error('Failed to upload document, Error: ' + str(check_uploadable.get('message')))
					
					st.session_state.upload_clicked = False

				st.button('Upload Document', on_click=handle_upload_click, disabled=st.session_state.upload_clicked)

		# Delete document
		st.header('Delete Document')
		doc_id_to_delete = st.number_input('Enter document ID to delete:', min_value=1, step=1, key='sap_docs_delete_id')
		if st.button('Delete Document'):
			delete_document(doc_id_to_delete)
			st.rerun()

	# List all documents (cached + paginated)
	st.subheader("Documents")

	# Controls
	c1, c2, c3 = st.columns([1, 1, 3])
	with c1:
		limit = st.selectbox("Rows per page", [50, 100, 200, 500], index=2, key="sap_docs_limit")
	with c2:
		if st.button("Refresh list", key="sap_docs_refresh"):
			fetch_sap_docs_page.clear()
			st.rerun()
	offset = st.session_state.get("sap_docs_offset", 0)

	# Fetch (avoid call_api_data/tenacity for list)
	headers_items = tuple(sorted((st.session_state.be_headers or {}).items()))
	try:
		documents_json = fetch_sap_docs_page(BE_ENDPOINT, headers_items, limit, offset)
	except Exception as e:
		st.error(f"Failed to load documents list: {e}")
		st.stop()

	# Pager buttons (works best if backend supports limit/offset)
	p1, p2, p3 = st.columns([1, 1, 6])
	with p1:
		if st.button("Prev", disabled=(offset == 0), key="sap_docs_prev"):
			st.session_state.sap_docs_offset = max(0, offset - int(limit))
			st.rerun()
	with p2:
		if st.button("Next", disabled=(len(documents_json) < int(limit)), key="sap_docs_next"):
			st.session_state.sap_docs_offset = offset + int(limit)
			st.rerun()
	

	# Normalize rows to a stable schema to avoid frontend mismatches after navigation
	def normalize_documents(rows: list[dict]) -> list[dict]:
		stable_cols = [
			'id',
			'document_name',
			'data_type',
			'version',
			'status',
			'domain',
			'language',
			'validity_end',
			'validity_year',
			'date_time',
			'author',
			'keywords',
		]
		normalized = []
		for r in rows:
			row = {k: r.get(k) for k in stable_cols}
			normalized.append(row)
		return normalized

	# Display documents in a table
	if not documents_json:
		st.info('No SAP documents found.')
		return
	
	df = pd.DataFrame(normalize_documents(documents_json))

	# Force stable columns order (defensive against backend changes)
	stable_cols = [
			'id',
			'document_name',
			'data_type',
			'version',
			'status',
			'domain',
			'language',
			'validity_end',
			'validity_year',
			'date_time',
			'author',
			'keywords',
	]
	for c in stable_cols:
			if c not in df.columns:
				df[c] = None
	df = df[stable_cols]

	# Format date_time column
	if 'date_time' in df.columns:
		df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

		#Format validity_end column
	if 'validity_end' in df.columns:
		# Keep as date object for stable rendering and for date_input defaults
		df['validity_end'] = pd.to_datetime(df['validity_end'], errors='coerce').dt.date

	if 'keywords' in df.columns:
		def _kw_to_str(x):
			if isinstance(x, list):
				return ', '.join([str(i) for i in x if i is not None and str(i).strip()])
			return '' if x is None else str(x)
		df['keywords'] = df['keywords'].apply(_kw_to_str)
		

	# Display document table
	col_labels = {
		'id': 'Id',
		'document_name': 'Document Name',
		'data_type': 'Document Type',
		'version': 'Version',
		'status': 'Status',
		'domain': 'Domain',
		'language': 'Language',
		'validity_end': 'Validity End Date',
		'validity_year': 'Validity Period (years)',
		'date_time': 'Upload Date/Time',
		'author': 'Author',
		'keywords': 'Keywords',
	}

	df_to_show = df.rename(columns=col_labels)

	st.subheader('Documents')
		
	use_scrollable = st.checkbox(
		'Use scrollable table (recommended)',
		value=True,
		key='sap_docs_use_scrollable_table',
		help='If you ever see a frontend React table error again, disable this to fall back to a static table.',
	)

	if use_scrollable:
		st.dataframe(
				df_to_show,
				width="stretch",
				height=650,
				key='sap_docs_table_df',
	)
	else:
		st.table(df_to_show.head(200))

	st.markdown('---')
	st.subheader('Edit Document Metadata')
		
		# This feature requires backend endpoint: PATCH /sap_documents/metadata
	doc_ids = df['id'].dropna().astype(int).tolist()
	if not doc_ids:
		st.info('No documents available.')
		return

	selected_id = st.selectbox('Select a document ID', options=doc_ids, key='sap_docs_edit_select_id')
	row = df[df['id'] == selected_id].iloc[0]

	status_options = ['active', 'deprecated', 'archived']
	domain_options = ['BW','GTS','FI','MDM','SD','PPE','QM','PM','IBP','PRO','OM','MTS','FI-AA','FI-GL','Other']
	language_options = ['EN', 'DE','FR','ES','IT','CN','JP','Other']

	def _norm_choice(v, options, default):
		try:
			# handle NaN
			if v is None:
				return default
			vs = str(v).strip()
			if not vs or vs.lower() == 'nan':
				return default
			return vs if vs in options else default
		except Exception:
			return default

	current_status = _norm_choice(row.get('status'), status_options, 'active')
	current_domain = _norm_choice(row.get('domain'), domain_options, 'Other')
	current_language = _norm_choice(row.get('language'), language_options, 'Other')

	with st.form('sap_docs_edit_form', clear_on_submit=False):
		edit_version = st.text_input('Version', value=str(row.get('version') or ''), key=f'sap_docs_edit_version_{selected_id}')
				
		edit_status = st.selectbox(
			'Status', 
			options=status_options,
			index=status_options.index(current_status),
			key=f'sap_docs_edit_status_{selected_id}',
			)
		
		edit_domain = st.selectbox(
			'Domain',
			options=domain_options,
			index=domain_options.index(current_domain),
			key=f'sap_docs_edit_domain_{selected_id}',
		)
		
		edit_language = st.selectbox(
			'Language',
			options=language_options,
			index=language_options.index(current_language),
			key=f'sap_docs_edit_language_{selected_id}',
		)

		edit_author = st.text_input(
			'Author',
			value=str(row.get('author') or ''),
			key=f'sap_docs_edit_author_{selected_id}',
		)

		edit_validity_year = st.number_input(
			'Validity Period (years)',
			min_value=1,
			max_value=20,
			value=int(row.get('validity_year') or 1),
			step=1,
			key=f'sap_docs_edit_validity_year_{selected_id}',
		)

		# validity_end can be None/NaT/date; avoid bool(pd.NaT) which raises
		ve = row.get('validity_end')
		use_end_default = False
		try:
			use_end_default = (ve is not None) and pd.notna(ve)
		except Exception:
			use_end_default = False

		use_end = st.checkbox(
			'Set validity end date', 
			value=use_end_default, 
			key=f'sap_docs_edit_use_end_{selected_id}')
		
		edit_validity_end = None
		if use_end:
			try:
				default_end = datetime.strptime(str(row.get('validity_end') or ''), '%Y-%m-%d').date()
			except Exception:
				default_end = datetime.now().date()
			edit_validity_end = st.date_input(
				'Validity End Date', 
				value=default_end, 
				key=f'sap_docs_edit_validity_end_{selected_id}'
				)

		edit_keywords = st.text_input(
			'Keywords (comma-separated)',
			value=str(row.get('keywords') or ''),
			key=f'sap_docs_edit_keywords_{selected_id}',
		)

		submit = st.form_submit_button('Save metadata')

	if submit:
		payload = {
					'doc_id': int(selected_id),
					'version': edit_version or None,
					'status': edit_status or None,
					'domain': edit_domain or None,
					'language': edit_language or None,
					'validity_year': int(edit_validity_year) if edit_validity_year is not None else None,
					'validity_end': edit_validity_end.isoformat() if edit_validity_end else None,
					'author': edit_author or None,
					'keywords': [k.strip() for k in str(edit_keywords).split(',') if k.strip()],
		}

		# Use requests directly to avoid call_api retry wrapper issues with PATCH (UnboundLocalError / RetryError)
		url = f"{BE_ENDPOINT.rstrip('/')}/sap_documents/metadata"

		try:
				
			r = requests.patch(url, json=payload, headers=st.session_state.be_headers, timeout=30)
			if r.ok:
				st.success('Metadata updated successfully.')
				st.rerun()
			else:
				st.error(f'Failed to update metadata: {r.status_code} {r.text}')
		except Exception as e:
			st.error(f'Failed to update metadata: {e}')

			
def delete_document(doc_id):
	"""
	Delete a document from the system.

	Sends a request to the backend to delete the specified document
	and displays the result.

	Args:
		doc_id: ID of the document to delete
	"""
	response = call_api(
		data={'doc_id': doc_id}, endpoint=BE_ENDPOINT, service='sap_documents/delete', method='DELETE', headers=st.session_state.be_headers
	)
	if response.get('message') == 'Document deleted successfully':
		st.success(f'Document {doc_id} deleted successfully')
	else:
		st.error(f'Failed to delete document {doc_id}')


@st.dialog('Document Analysis', width='large')
def document_analysis(document):
	st.markdown(document)


def manage_documents():
	"""
	Render the document management interface.

	Provides a selector for choosing which document type to manage
	and renders the appropriate management interface.
	"""
	accessible_doc_pool = ['SAP Docs']
	if st.session_state.is_admin:
		accessible_doc_pool.append("Solver's KB Articles")
	doc_pool_map = {'SAP Docs': manage_sap_documents}
	with st.sidebar:
		doc_pool = st.sidebar.selectbox('Choose a knowledge pool', accessible_doc_pool, key='knowledge_pool')
	doc_pool_map[doc_pool]()


load_config()
if 'session_id' not in st.session_state:
	init_session()

PAGE_MAPPING = {
	'Chat Interface': normal_chat_interface,
	'Multi-Agent Chat Interface': ma_chat_interface,
	'Analytics Dashboard': analytics_dashboard,
	'SAP Documents Q&A Testing': sap_docs_test_page,
	'Manage Documents': manage_documents,
}

st.set_page_config(page_title=FE_CFG['page_title'], layout='centered')

