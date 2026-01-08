# Quantitative Evaluation

**To run the quantitative evalution, please follow these steps:**

***1. Install Sentence Transformers library:***

   Option 1 (may fail due to TLS CA certificate error): `pip install pandas sentence-transformers rouge-score scikit-learn scipy numpy openpyxl`

   Option 2: (Used in the thesis): Downloading the files to the local path: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

***2. Run the script in terminal:***

`python evaluate.py`

***Small reminder: the Excel file of the test set is uploaded in the folder already.***
   
# Artifacts Implementation

## This folder consists of the core scripts that are related to the thesis

### Key Features

* **Hybrid Search:** Utilizes a combination of Vector Search (via `pgvector`) and BM25 Search to maximize retrieval accuracy and reduce hallucinations.
  
* **Query Expansion** Extract 2-6 core technical keywords or short phrases from query for search
   
* **Document Lifecycle Governance:**
    * **Validity Enforcement:** metadata fields for validity years and end dates.
      
    * **Auto-Deprecation:** Automated background checks document validity, notify responsible roles via email when they expire and set documents' status to "deprecated".
      
    * **Metadata Management:** Rich metadata tagging (status, domain, language, version). Edit metadata on the document management interface. Upload matedata when publishing a document. Filter metadata on the chat interface.

* **Interactive UI:** A Streamlit application for chatting with the AI, uploading documents, managing metadata.
  
* **Feedback Loop:** Users can rate feedback and flag "outdated knowledge," which triggers automated email alerts to responsible roles.

## Architecture Modules

The repository is structured into the following core components:

* **Frontend (`va_app.py`):** A Streamlit interface for chat, document Management.

* **API Layer (`*_router.py`):** FastAPI routers handling HTTP requests for chat streaming, feedback, and document CRUD operations.

* **Service Layer (`main.py`):** The core logic orchestrating the RAG, hybrid search, query expansion, metagata management, document validity monitor, etc.

* **Data Processing (`document_processor.py`):** Processing documents, extracting and chunking content.
  
* **Storage Layer (`document_search.py` & Models):** Manages interactions with PostgreSQL (using `pgvector`) and S3 for object storage.
