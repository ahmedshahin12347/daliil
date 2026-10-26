# Daliil

A simple multilingual document Q&A API built with **FastAPI**,
**ChromaDB**, and **Sentence Transformers**.

Daliil lets you upload PDF, TXT, and Markdown files, ask questions in
Arabic or English, and retrieve relevant passages with source and page
information.

## Features

-   Upload and index PDF, TXT, and Markdown documents.
-   Search document content using multilingual semantic embeddings.
-   Generate answers grounded in retrieved passages, with source
    citations.
-   Normalize common Arabic PDF text-extraction issues.
-   Look up numbered Arabic legal articles using exact matching.
-   Run in retrieval-only mode without an LLM.
-   Manage indexed documents through API endpoints.

## How It Works

1.  **Upload:** Extract text from the supported document formats.
2.  **Chunk:** Split the text into overlapping word-based chunks.
3.  **Index:** Create embeddings and store chunks in ChromaDB.
4.  **Retrieve:** Find passages relevant to the user's question.
5.  **Answer:** If an LLM is configured, generate an answer using the
    retrieved passages.

Without an LLM configuration, Daliil returns the retrieved passages
without generating an answer.

## Tech Stack

-   Python
-   FastAPI
-   ChromaDB
-   Sentence Transformers (`intfloat/multilingual-e5-small`)
-   pypdf
-   OpenAI-compatible LLM API

## Getting Started

### 1. Clone the repository

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository URL.

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd daliil
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using `.env.example` as a reference.

  Variable          Default                            Description
  ----------------- ---------------------------------- ---------------------------------------
  `EMBED_MODEL`     `intfloat/multilingual-e5-small`   Embedding model
  `CHROMA_DIR`      `./chroma_db`                      Local vector database directory
  `CHUNK_WORDS`     `200`                              Number of words per chunk
  `CHUNK_OVERLAP`   `40`                               Overlap between chunks
  `TOP_K`           `4`                                Number of passages to retrieve
  `LLM_BASE_URL`    Not set                            Base URL of an OpenAI-compatible API
  `LLM_API_KEY`     `none`                             API key for the LLM provider
  `LLM_MODEL`       Not set                            Model name used for answer generation

LLM settings are optional. To enable generated answers, configure
`LLM_BASE_URL` and `LLM_MODEL`.

**Security:** Do not commit your real `.env` file or expose API keys.

### 5. Run the API

If `main.py` is in the repository root, run:

``` bash
uvicorn main:app --reload
```

Open the interactive API documentation:

**http://127.0.0.1:8000/docs**

## API Endpoints

  -----------------------------------------------------------------------
  Method                  Endpoint                Description
  ----------------------- ----------------------- -----------------------
  `GET`                   `/health`               Check service status,
                                                  indexed chunk count,
                                                  and mode

  `GET`                   `/documents`            List indexed document
                                                  filenames

  `POST`                  `/ingest`               Upload and index a PDF,
                                                  TXT, or Markdown file

  `POST`                  `/ask`                  Ask a question and
                                                  retrieve relevant
                                                  passages

  `DELETE`                `/documents/{source}`   Delete a document from
                                                  the index

  `DELETE`                `/documents`            Clear the document
                                                  index
  -----------------------------------------------------------------------

### Example: Ask a Question

Send a `POST` request to `/ask` with a JSON body:

``` json
{
  "question": "What does Article 5 say?",
  "top_k": 4
}
```

The response includes the operating mode, retrieved source passages,
page numbers, and similarity scores. When an LLM is configured, it also
includes a generated answer.

To index a document, use `/ingest` and upload the file as form data.

## Limitations

-   Scanned PDFs are not supported because OCR is not implemented.
-   Chunking is based on word counts and may split content in the middle
    of a sentence.
-   There is no reranker or similarity threshold yet.
-   Answer generation depends on the configured LLM endpoint.
-   Automated retrieval and answer-quality evaluation has not been added
    yet.

## Future Improvements

-   Add automated retrieval evaluation.
-   Add a reranking model.
-   Support OCR for scanned PDFs.
-   Improve chunking and retrieval quality.
-   Add automated tests and containerized deployment.

