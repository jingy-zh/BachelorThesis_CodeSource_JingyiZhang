# Quantitative Evaluation

**To run the quantitative evalution, please follow these steps:**

***1. Install Sentence Transformers library:***

   Option 1 (may fail due to TLS CA certificate error): `pip install pandas sentence-transformers rouge-score scikit-learn scipy numpy openpyxl`

   Option 2: (Used in the thesis): Downloading the files to the local path: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

***2. Run the script in terminal:***

`python evaluate.py`

> [!TIP]
> The Evaluation_Queries_and_Answers.xlsx test set has been uploaded to the "Quantitative Evaluation" folder in GitHub and the WWU Thesis Uploader. Should the GitHub fail to open, please use the copy in the WWU Thesis Uploader instead.***
   
# Artifacts Implementation

> [!IMPORTANT]
>  Due to the high dependency of enterprise AWS and Azure configurations, the application can not be run in the standard environment. This folder only contains the scripts that are relevant to this thesis. This codebase is provided for academic review as part of the Bachelor Thesis.

### Key Features

* **Hybrid Retrieval:** Utilizes a combination of Vector Search (via `pgvector`) and BM25 Search to maximize retrieval accuracy and reduce hallucinations.
  
* **Query Expansion** Extract 2-6 core technical keywords or short phrases from query for search
   
* **Knowledge Lifecycle Governance:**
    * **Validity Enforcement:** metadata fields for validity years and end dates.
      
    * **Auto-Deprecation:** Automated background checks document validity, notify responsible roles via email when they expire and set documents' status to "deprecated".
      
    * **Admin Dashboard:** A interface for document management, metadata uploading, and metadata editing.

* **Interactive UI:** A interface for chatting with the AI, filtering metadata to narrow RAG retrieval scope.
  
* **Feedback Loop:** Users can rate feedback and flag "outdated knowledge," which triggers automated email alerts to responsible roles.

### Layers and Functional Description of Scripts

## Backend

* **Database Layer (`*_model.py`):** Database structure for data storage.SQLAlchemy ORM models represents database tables.
   * **feedback_model.py:** Database structure to store user feedback data.
   * **sap_documents_model.py:** Database structure to storeg SAP documents and their vector embeddings. New Metadata fields implemented from the Demonstration are added to the database.

* **API Layer (`*_router.py`):** FastAPI routers handling HTTP requests for chat streaming, feedback, and document CRUD operations.
    * **feedback_router.py:** FastAPI endpoints for feedback management. **The function of 'sending email to the responsible role when the user flags the answer as outdated knowledge' is defined in the functions `def _send_email()` and `async def save_feedback ()`.** 
    * **sap_documents_router.py:** FastAPI endpoints for managing SAP documents.
       
* **Validation Layer (`*_schema.py`):** Pydantic models for request/response validation.
   * **feedback_schema.py:** Request model for saving user feedback, using Pydantic for data validation
   * **sap_documents_schema.py:** Request and response models for retrieving and managing SAP document, using Pydantic for data validation. 
      
* **Service Layer** The core functionality logic.
   * **sap_documents** Core functionality in sap document management, retrieval, and validity monitoring.
      * **config.py:** Configuration settings for SAP document processing and storage, including LLM model and temperature, chunk size, embedding model, and vector size etc.
      * **document_processor.py:** Processing document before retrieval, including extracting text, images, and tables. Text is segements into chunks, images and tables are stored in AWS S3 bucket.
      * **document_search.py:** **_Managing document embeddings in the vector database._** Creating document, **hybrid retrieving in the database (`def search ()`)**, removing document from the database, **updating metadata fields re-ingesting embeddings (`def update_document_metadata ()`)**, **system automated monitoring document validity, and setting status of expired document to 'deprectaed' in database (`def expire_documents ()`)**. Hard gate and softe gate in hybrid retrieval also are defined in the `def search ()` function.
      * **main.py:** **_SAP Document processing module through documents._** **Expanding query (`class InternalQueryExpander`)**; uploading, listing, and deleting document; **updating metadata (`def update_document_metadata ()`)**; **monitoring document validity and sending notification email to responsible roles (`def enforce_document_validity ()`)**, **constraining the LLM through prompting to avoid hallucination answers (`def execute ()`, `def execute_stream ()`)**.
    
## Frontend

* **va_app.py:** A Streamlit frontend interface for chatting (`def sap_docs_test_page()`), submitting feedback (`def _submit_feedback()`), and document management (`def manage_sap_documents()`). **Metadata filter function on chat interface is defined in `def sap_docs_test_page()`**.
