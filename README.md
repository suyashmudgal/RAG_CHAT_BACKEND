# DocChat AI — Production RAG Document Intelligence Platform

DocChat AI is an enterprise-grade, retrieval-augmented generation (RAG) backend engineered with **FastAPI**, **LangChain**, **ChromaDB**, **PostgreSQL (Supabase)**, **Supabase Storage**, and **Groq Cloud LLMs**. The platform enables multi-user document ingestion, layout-aware multimodal parsing (text, tables, charts, and scanned OCR), adaptive retrieval with query intent classification, grounded reasoning with anti-hallucination guardrails, real-time Server-Sent Events (SSE) streaming, persistent conversation management, and publication-quality conversation PDF export.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Client["Frontend Client (React/Vite SPA)"]
        UI["Web Interface (Port 5173)"]
    end

    subgraph AuthLayer["Authentication & Security"]
        JWT["HTTP-Only JWT Cookie (72h TTL)"]
        AuthDB["SQLite auth.db (Fast Auth State)"]
        UserSync["User Sync Service (120s In-Memory Cache)"]
    end

    subgraph APILayer["FastAPI Application (Port 8000/8001)"]
        R_Auth["/auth (Signup, Signin, Google OAuth, Profile)"]
        R_Upload["/upload (Non-blocking Ingestion, ?sync=true)"]
        R_Docs["/documents (List, Status, Download, Delete)"]
        R_Conv["/conversations (CRUD, Bulk Delete, Pin/Unpin, Export PDF)"]
        R_Chat["/chat & /chat/stream (SSE RAG Streaming)"]
    end

    subgraph IngestionPipeline["Document Ingestion & Extraction"]
        Storage["Supabase Private Storage (users/{user_id}/...)"]
        Progress["ProgressTracker State Machine (11 Stages)"]
        PyMuPDF["PyMuPDF (Layout-Aware Text & Tables)"]
        VisualExtract["VisualExtractor (RapidOCR + Chart/Diagram Structuring)"]
        Chunker["DocumentChunker (800/200 + Element-Aware Protection)"]
    end

    subgraph StorageLayer["Data & Vector Persistence"]
        PG[(Supabase PostgreSQL: users, documents, conversations, messages)]
        Chroma[(ChromaDB Collection: 'documents' + all-MiniLM-L6-v2 384d)]
    end

    subgraph RAGCore["Query Understanding & Retrieval Engine"]
        QU["QueryUnderstanding (Intent, Follow-up Context, Expansions)"]
        BalRet["Multi-Doc Balanced Retrieval (Similarity + Overlap)"]
        Grounding["Groundedness Guardrails & Claim Sanitizer"]
        GroqLLM["Groq LLM: openai/gpt-oss-120b (Resilient Fallback Tier)"]
    end

    UI -->|Credentials / OAuth| R_Auth
    R_Auth --> JWT
    R_Auth --> AuthDB
    AuthDB --> UserSync
    UserSync --> PG

    UI -->|Multipart Upload| R_Upload
    R_Upload --> Storage
    R_Upload --> Progress
    R_Upload --> PyMuPDF
    R_Upload --> VisualExtract
    PyMuPDF --> Chunker
    VisualExtract --> Chunker
    Chunker -->|Batch Size 32| Chroma
    Progress -->|Status Sync| PG

    UI -->|"User Question (SSE Stream)"| R_Chat
    R_Chat --> QU
    QU --> BalRet
    BalRet --> Chroma
    BalRet --> Grounding
    Grounding --> GroqLLM
    GroqLLM -->|SSE Tokens| UI
    GroqLLM -->|Atomic Persist| PG
    R_Chat -->|Citation Precision Filter| UI
```

---

## Core Features

- **Multi-Tenant User Isolation**: Strict ownership validation across database records, physical storage paths, vector collections, and chat memory. Cross-user access consistently returns `404 Not Found`.
- **Hybrid High-Performance Auth**: SQLite (`auth.db`) credential resolution coupled with asynchronous synchronization to Supabase PostgreSQL, fronted by an in-memory user cache (120-second TTL) protected by thread locks.
- **Multimodal Document Processing**:
  - Layout-aware PDF text block parsing preserving multi-column reading orders.
  - Native table extraction converting 2D grid structures into clean Markdown tables with row-level semantic representations.
  - Local visual extraction powered by PyMuPDF and RapidOCR (with Tesseract fallback) that extracts pie charts, bar charts, line graphs, and vector drawings, clustering sub-charts and deriving categorical totals.
  - Full-page OCR for scanned documents (triggered when native page text < 50 characters) with confidence reporting.
  - Support for DOCX (`docx2txt`) and plain text TXT with multi-encoding fallback (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`).
- **Element-Aware Chunking**: Preserves tables, charts, graphs, diagrams, and OCR blocks intact up to 3,000 characters without destructive splitting, maintaining row/column and visual data coherence.
- **Adaptive Query Understanding & Conceptual Expansion**:
  - Classifies queries into 9 distinct intents (`single_doc_direct`, `multi_doc_synthesis`, `comparison`, `concept_reasoning`, `application_implementation`, `summary`, `follow_up`, `general_knowledge`, `document_qa`).
  - Follow-up pronoun resolution and contextual entity restoration across recent conversational turns.
  - Dynamic conceptual search rewrites for mathematics, optimization algorithms, neural architectures, ATS resume matching, and numbered exam problems.
- **Balanced Multi-Document Retrieval**: Prevents dominant documents from starving smaller candidates during comparison or cross-document synthesis queries.
- **Strict Groundedness Guardrails**:
  - System prompts enforce strict claim faithfulness and uncertainty preservation.
  - Post-processing safety filter (`sanitize_grounded_claims`) intercepts and reverts unsupported, exaggerated certainty claims (e.g. converting "guaranteed market" back to "potential market").
  - 5-category answer classification: `DIRECTLY_GROUNDED`, `DERIVED_FROM_DOCUMENT_CONCEPT`, `APPLICATION_IMPLEMENTATION`, `GENERAL_KNOWLEDGE`, and `INSUFFICIENT_INFORMATION`.
- **Precision Citation Selection**:
  - Negative answer suppression (returns empty citation arrays when an answer admits lack of evidence).
  - Cosine similarity conversion from Chroma Euclidean distances.
  - Combined scoring: semantic similarity (50%), answer term overlap (30%), explicit page reference bonus (+35%), and numeric fact overlap (+15%).
  - Code blocks stripped prior to overlap computation to prevent hallucinated citations on generated code.
  - Page-level deduplication grouping multiple supporting passages into a single citation per document page.
- **Persistent Conversation Management**:
  - Auto-titling based on initial query length with custom title preservation (`is_custom_title: True`).
  - Pinning and unpinning with composite-indexed ordering (`is_pinned DESC, updated_at DESC`).
  - Atomic bulk conversation deletion with single-transaction rollback.
  - Cascade message deletion via PostgreSQL foreign key constraints.
- **Publication-Quality PDF Export**:
  - ReportLab-based server-side generation operating completely in-memory via `io.BytesIO`.
  - Full UTF-8 support using bundled DejaVu fonts (currency ₹/€/$, arrows, Greek letters, math operators).
  - LaTeX mathematics rendered to crisp embedded images via `matplotlib.mathtext` (`$...$` and `$$...$$`).
  - Strict code block masking ensuring programming code containing LaTeX-like symbols is never converted into math.
  - Two-pass `NumberedCanvas` producing dynamic "Page X of Y" footers.

---

## Repository & Project Structure

```
RAG_CHAT_BACKEND/
├── auth.db                               # Local SQLite database for fast auth state
├── backend/
│   ├── alembic/                          # Alembic database migrations
│   │   ├── env.py                        # Migration runtime environment
│   │   ├── script.py.mako                # Migration template
│   │   └── versions/                     # Revision history
│   │       ├── 8330f427d2ee_create_initial_tables.py
│   │       ├── 9ffbbc8ff6d7_add_indexes_for_conversations_and_.py
│   │       ├── d4e5f6a7b8c9_phase6a_progress_and_indexes.py
│   │       └── e5f6a7b8c9d0_phase7a_pinned_and_custom_title.py
│   ├── alembic.ini                       # Alembic CLI configuration
│   ├── auth/                             # Authentication subsystem
│   │   ├── database.py                   # Async SQLite connection and table init
│   │   ├── dependencies.py               # JWT cookie extraction & in-memory user cache
│   │   ├── models.py                     # Pydantic auth request/response schemas
│   │   ├── router.py                     # Auth routes (signup, signin, Google OAuth, me)
│   │   └── utils.py                      # bcrypt hashing and PyJWT token generation
│   ├── chroma_db/                        # ChromaDB persistent vector database directory
│   ├── config/
│   │   └── settings.py                   # Pydantic BaseSettings loading from .env
│   ├── database/
│   │   └── session.py                    # SQLAlchemy engine, pooled sessions, ping utility
│   ├── models/
│   │   ├── database.py                   # SQLAlchemy ORM models (User, Document, Conversation, Message)
│   │   └── schemas.py                    # Pydantic schemas for API endpoints
│   ├── routes/
│   │   ├── chat.py                       # POST /chat and POST /chat/stream
│   │   ├── conversations.py              # GET/POST/PATCH/DELETE /conversations & PDF export
│   │   ├── documents.py                  # GET /documents, status, download, delete
│   │   └── upload.py                     # POST /upload (sync & background async)
│   ├── services/
│   │   ├── chat_service.py               # RAG coordinator, Groq LLM client, SSE streamer
│   │   ├── citation_service.py           # Precision scoring, overlap analysis, deduplication
│   │   ├── deps.py                       # LRU cached singleton service dependencies
│   │   ├── document_metadata.py          # Legacy metadata helpers
│   │   ├── document_processor.py         # Pipeline orchestrator (extract -> chunk -> embed -> index)
│   │   ├── document_representation.py    # Normalized document/page/element data models
│   │   ├── embedding_service.py          # HuggingFace sentence-transformers singleton
│   │   ├── pdf_export_service.py         # ReportLab conversation PDF generator with math support
│   │   ├── progress_service.py           # Thread-safe in-memory + DB progress state machine
│   │   ├── query_understanding.py        # Intent classifier, follow-up resolver, search expansion
│   │   ├── storage_service.py            # Supabase Storage client & filename sanitizer
│   │   ├── text_chunker.py               # Element-aware recursive text splitter
│   │   ├── text_extractor.py             # Multi-format extractor (PyMuPDF, docx2txt, text)
│   │   ├── user_sync.py                  # SQLite-to-PostgreSQL user synchronization
│   │   ├── vector_store.py               # LangChain Chroma wrapper with multi-doc retrieval
│   │   └── visual_extractor.py           # Local OCR, chart parsing, and diagram extractor
│   ├── main.py                           # FastAPI application entrypoint & lifespan
│   ├── migrate_local_to_supabase.py      # Storage migration script for existing files
│   ├── reindex_visual_documents.py       # Re-indexing utility for visual document content
│   ├── requirements.txt                  # Python dependencies
│   ├── .env.example                      # Documented environment variable template
│   └── test_*.py                         # Complete suite of regression and integration tests
└── README.md                             # Engineering documentation
```

---

## Tech Stack Verification

| Component | Technology | Verified Implementation Details |
| :--- | :--- | :--- |
| **API Framework** | FastAPI (Python 3.10+) | Asynchronous ASGI routes, lifespan context manager, dependency injection |
| **Server Engine** | Uvicorn (`uvicorn[standard]`) | ASGI server running asynchronous event loops |
| **Language Model Client** | LangChain Groq (`ChatGroq`) | Primary model: `openai/gpt-oss-120b`, `temperature: 0.3`, `max_tokens: 2048` |
| **Model Fallbacks** | Multi-tier Fallback List | `[settings.groq_model, "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]` |
| **Local Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Run locally on CPU via `langchain-huggingface`, 384 dimensions, normalized |
| **Vector Database** | ChromaDB (`langchain-chroma`) | Persistent local directory `./chroma_db`, collection `"documents"` |
| **Relational Database** | PostgreSQL (Supabase) | Managed via SQLAlchemy 2.0 with connection pooling (`pool_recycle=300`) |
| **Schema Migrations** | Alembic | Version-controlled schema migrations under `backend/alembic/versions` |
| **Document Storage** | Supabase Storage | Private bucket `"documents"`, REST API via `httpx`, user-scoped paths |
| **PDF Extraction** | PyMuPDF (`fitz`) | Text block extraction, native table detection, pixmap rendering at 200 DPI |
| **Local OCR** | `rapidocr-onnxruntime` | Fast CPU-based ONNX OCR engine with fallback to `pytesseract` |
| **Word Processing** | `docx2txt` | Native `.docx` text extraction |
| **PDF Export Engine** | ReportLab (`reportlab>=4.0.0`) | In-memory flowable generation, custom canvas, DejaVu Unicode fonts |
| **Math Rendering** | Matplotlib (`mathtext`) | Inline and display LaTeX rendered into embedded image flowables |
| **Authentication** | SQLite (`aiosqlite`) + PyJWT + bcrypt | HTTP-only cookies, 72h JWT lifespan, bcrypt password hashing |

---

## Frontend Architecture & System Separation

The user-facing frontend is **not located in this repository**. It is maintained as a separate React/Vite single-page application (SPA).

Verified interaction specifications between this backend and the frontend:
- **Default Frontend Origin**: `http://localhost:5173` (configured via `FRONTEND_URL` and `CORS_ORIGINS`).
- **Authentication Handshake**: JWT token transported via HTTP-only cookie (`token`, `SameSite=Lax`, `Path=/`, `Max-Age=259200`).
- **OAuth Callback Redirection**:
  - Success redirects to `{FRONTEND_URL}/app` with authenticated cookie set.
  - Failure redirects to `{FRONTEND_URL}/login?error={safe_message}`.
- **Streaming Contract**: SSE stream at `POST /chat/stream` sending discrete JSON messages (`type: "token"`, `type: "sources"`, `type: "done"`, `type: "error"`).
- **Export Endpoint**: Direct binary stream at `GET /conversations/{id}/export/pdf` with `Content-Disposition: attachment; filename="<safe_title>.pdf"`.

---

## Detailed Ingestion Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor Client as Frontend Client
    participant Upload as POST /upload
    participant Storage as Supabase Storage
    participant DB as PostgreSQL
    participant Tracker as ProgressTracker
    participant Extractor as Text & Visual Extractor
    participant Chunker as DocumentChunker
    participant Embedder as all-MiniLM-L6-v2
    participant Chroma as ChromaDB

    Client->>Upload: Upload files (multipart/form-data)
    Upload->>Upload: Validate extension (.pdf, .docx, .txt) & size (<= 50MB)
    Upload->>Tracker: Register job (QUEUED, 5%)
    Upload->>Tracker: Update stage (UPLOADING, 15%)
    Upload->>Storage: Store at users/{user_id}/documents/{doc_id}/{filename}
    Upload->>DB: Insert Document record (status='UPLOADED', progress=25%)
    Upload->>Tracker: Update stage (UPLOADED, 25%)
    Upload-->>Client: Return 200 OK with job IDs (Background processing begins)

    par Background Worker
        Upload->>Tracker: Update stage (EXTRACTING, 35%)
        Upload->>Extractor: Extract text, tables, charts, OCR
        Upload->>Tracker: Update stage (PARSING, 45%)
        Upload->>Tracker: Update stage (CHUNKING, 55%)
        Upload->>Chunker: Element-aware chunking (tables/charts intact)
        Upload->>Tracker: Update stage (EMBEDDING, 55-85%)
        loop Batch Size = 32
            Upload->>Embedder: Generate dense embeddings
            Upload->>Chroma: Insert chunks with user_id & document_id metadata
            Upload->>Tracker: Increment processed_chunks
        end
        Upload->>Tracker: Update stage (INDEXING, 90%)
        Upload->>Tracker: Update stage (FINALIZING, 95%)
        Upload->>DB: Update Document (status='processed', stage='COMPLETED', progress=100%)
        Upload->>Tracker: Mark completed (COMPLETED, 100%)
    end
```

### Supported File Formats & Validation
- **Supported Extensions**: `.pdf`, `.docx`, `.txt`.
- **Validation Rules**:
  - File extension check against `ALLOWED_EXTENSIONS`.
  - Empty file detection (`len(content) == 0`).
  - Size enforcement: configured by `MAX_FILE_SIZE_MB` (default `50 MB`).
- **Filename Sanitization**:
  - Path traversal sequences (`../`, `..\`) stripped via regex.
  - Basename extraction ensuring no folder nesting.
  - Characters restricted to `[a-zA-Z0-9_.-]`.
  - Leading periods removed to prevent hidden file attacks.

### Extraction Implementation
1. **PDF Documents**:
   - PyMuPDF (`fitz`) opens document in memory.
   - Native tables extracted via `page.find_tables()`, converting cells to Markdown with a `Semantic Row Values` appendix.
   - Text blocks extracted in natural reading order, filtering out blocks that intersect table bounding boxes by >50% to prevent duplication.
   - Falls back to `PyPDFLoader` if PyMuPDF encounters structural corruption.
   - `visual_extractor.extract_visuals_from_pdf` renders pages at 200 DPI, runs OCR, detects vector drawings, and structures charts.
2. **Word Documents (`.docx`)**:
   - Loaded via `Docx2txtLoader`.
   - Structural text extracted; page metadata default set to 0.
3. **Plain Text (`.txt`)**:
   - `TextLoader` attempted across encodings: `utf-8`, `utf-8-sig`, `latin-1`, `cp1252`.

### Error Handling & Rollback Guarantees
If extraction, chunking, or indexing fails during ingestion:
1. **ChromaDB**: All chunks associated with `document_id` are purged (`vector_store.delete_document(document_id)`).
2. **Supabase Storage**: The uploaded file is deleted (`storage_service.delete_file(storage_path)`).
3. **PostgreSQL**:
   - In synchronous mode (`?sync=true`): the document record is deleted to prevent orphaned records.
   - In asynchronous mode (default): the document status is marked `FAILED`, recording a sanitized, single-line error message (`doc.error_message`) without stack traces or secret leakage.
4. **Temporary Files**: `tempfile.NamedTemporaryFile` instances are unlinked in `finally` blocks.

---

## Chunking Strategy

Implemented in `services/text_chunker.py` using `DocumentChunker`:
- **Underlying Splitter**: `langchain_text_splitters.RecursiveCharacterTextSplitter`.
- **Chunk Size**: 800 characters (`CHUNK_SIZE`).
- **Chunk Overlap**: 200 characters (`CHUNK_OVERLAP`).
- **Separators**: `["\n\n", "\n", ". ", " ", ""]`.
- **Length Function**: Python built-in `len`.

### Element-Aware Preservation
To prevent mathematical severance of structured information:
- Any chunk tagged as `table`, `visual`, `CHART`, `GRAPH`, `DIAGRAM`, `IMAGE`, or `OCR` that has a length **$\le$ 3,000 characters** is **preserved intact without splitting**.
- Narrative text and oversized elements are recursively split while preserving paragraph and sentence boundaries.
- **Metadata Stamped on Every Chunk**:
  - `document_id`: UUID string of the parent document.
  - `user_id`: Stringified PostgreSQL user ID for query filtering.
  - `page_number`: 1-indexed page number (0 for non-paginated files).
  - `element_type`: Classified element type (`TEXT`, `TABLE`, `CHART`, `DIAGRAM`, `OCR`).
  - `chunk_index`: Integer position in document chunk list.
  - `chunk_id`: Unique identifier (`{document_id}_chunk_{chunk_index}`).

---

## ChromaDB Architecture & Vector Store

Implemented in `services/vector_store.py`:
- **Persistence Path**: Configurable via `CHROMA_PERSIST_DIR` (default: `./chroma_db`).
- **Collection Name**: `"documents"`.
- **Embedding Model**: `all-MiniLM-L6-v2` loaded through `services/embedding_service.py`.
- **Metadata Storage**: Chroma collection metadata + auxiliary `document_metadata.json` persisted to disk.
- **Batch Processing**: Ingestion occurs in batches of 32 chunks (`batch_size=32`), triggering progress updates after each batch.
- **Multi-Document Retrieval**:
  - `similarity_search_multi_doc_for_user`: Iterates across targeted documents, retrieving `top_k_per_doc` (default 2–3) chunks per document, supplemented by global top hits.
  - Chunks are deduplicated by composite key `{doc_id}_{page}_{hash(content[:100])}` and ordered by distance.
- **User Isolation**:
  - Ingestion stamps `metadata["user_id"] = str(user_id)`.
  - Retrieval queries apply filter `{"user_id": str(user_id)}` or `$and` filters: `{"$and": [{"user_id": str(user_id)}, {"document_id": document_id}]}`.
  - Unauthenticated or cross-user vector retrieval is prevented at the database driver layer.

---

## Retrieval & Query Understanding Pipeline

```mermaid
flowchart LR
    Q[User Question] --> QU[Query Understanding]
    QU -->|Intent Classification| Intent{Intent?}
    QU -->|Entity Resolution| Context[Contextualized Query]
    QU -->|Domain Expansions| Exp[Conceptual Rewrites]

    Intent -->|Comparison / Multi-Doc| BalRet[Multi-Doc Balanced Retrieval]
    Intent -->|Single Doc / Direct| SingleRet[User Vector Search]
    Intent -->|Concept / Implementation| ConceptRet[Conceptual Expansion Search]

    BalRet --> Candidates[Raw Candidate Chunks]
    SingleRet --> Candidates
    ConceptRet --> Candidates

    Candidates --> SamePage[Same-Page Text Context Enrichment]
    SamePage --> Filter[Top 8 Chunks Window]
    Filter --> Prompt[Context Assembly & Grounded Prompt]
    Prompt --> LLM[Groq LLM: openai/gpt-oss-120b]
```

### 1. Intent Classification
`QueryUnderstanding.analyze_query` evaluates regex patterns, conversation history, and uploaded document filenames to classify the request into one of nine intents:
1. `CONCEPT_REASONING`: Mathematical questions, parameter updates, derivative reasoning.
2. `APPLICATION_IMPLEMENTATION`: Requests for programming code, PyTorch/Python scripts.
3. `MULTI_DOC_SYNTHESIS`: Questions spanning multiple documents or common skills.
4. `COMPARISON`: Candidate evaluation, ATS comparisons, "which one is better".
5. `SUMMARY`: Requests for overviews or candidate profiles.
6. `SINGLE_DOC_DIRECT`: Queries explicitly mentioning a specific file or topic.
7. `FOLLOW_UP`: Elliptical questions relying on prior turns ("What about certifications?").
8. `GENERAL_KNOWLEDGE`: Questions outside uploaded documents (e.g. "What is supervised learning?").
9. `DOCUMENT_QA`: Standard single-document question answering.

### 2. Follow-Up Resolution
When `history` contains prior messages and the query is short or elliptical:
- The system extracts the last 6 turns from history.
- Resolves pronouns ("it", "that", "this") and concept references (e.g. mapping "Give code for it" to "Give me practical Python / PyTorch implementation code for Transformer architecture and self-attention").
- Maps candidate entity contexts (e.g. resumes, candidates, AI/ML profiles).

### 3. Conceptual Rewrites & Query Expansion
Expands user queries with domain-specific search vectors:
- **Visual & Charts**: Appends keywords `chart visual graph data distribution percentage breakdown values`.
- **Arithmetic & Mathematics**: Adds `definition and concept of addition arithmetic combining numbers operations sum`.
- **Optimization & ML**: Expands with `gradient descent parameter update learning rate loss reduction minimization`.
- **Architectures**: Adds `Transformer architecture self-attention multi-head attention encoder decoder layers`.
- **ATS & Resumes**: Expands with technical skill catalogs, frameworks, databases, and experience indicators.

### 4. Same-Page Context Enrichment
If retrieved candidates include a visual chart or table from page $P$ of document $D$:
- An internal search automatically queries for plain text on page $P$ of document $D$.
- This guarantees that introductory directions, contextual problem statements, and base population totals (e.g. "Total students = 4,500") are supplied to the LLM alongside visual data.

---

## Knowledge-Grounded Reasoning & Anti-Hallucination

The system enforces strict distinction between verified facts, logical inferences, practical implementations, and general knowledge.

### The 5 Architectural Answer Classifications
1. `DIRECTLY_GROUNDED`: The answer is explicitly and verbatim supported by the retrieved document context. Citations point to the exact page.
2. `DERIVED_FROM_DOCUMENT_CONCEPT`: The answer is logically or mathematically derived from concepts in the document (e.g., calculating $3 + 4 = 7$ from an addition rule, or deducing parameter updates when gradient $> 0$). The conceptual foundation is cited, and the derivation is explicitly labeled.
3. `APPLICATION_IMPLEMENTATION`: Practical code (e.g. Python or PyTorch) generated to implement an architecture or method described in the document. The response explicitly clarifies that code was constructed from specifications, not extracted verbatim from the file.
4. `GENERAL_KNOWLEDGE`: The user asked a broad concept or domain question not contained in the documents. The system provides an answer using general knowledge without fabricating document citations.
5. `INSUFFICIENT_INFORMATION`: A specific document fact, metric, or entity was requested that is completely absent and cannot be derived. The system issues an explicit, concise notice of absence.

### Post-Processing Claim Sanitizer (`sanitize_grounded_claims`)
To counter model tendencies to exaggerate certainty during summarization:
- If context states "potential markets" or "may become suppliers", phrases such as "guaranteed market", "ready-made market", "guaranteed customers", or "guaranteed returns" are intercepted and rewritten.
- If context states "could reduce demand", claims that an initiative "will eliminate illegal excavation" or "eliminates illegal excavation" are rewritten to "could reduce the demand for illegally excavated objects".

---

## Precision Citation Engine

Implemented in `services/citation_service.py`:

```mermaid
flowchart TD
    Raw[Retrieved Chunks + LLM Answer] --> NegCheck{Negative Answer?}
    NegCheck -->|Yes: Information not found| Empty[Return Empty Citations Array]
    NegCheck -->|No| DistConv[Convert Euclidean Distance to Cosine Similarity]
    DistConv --> SimThresh{Similarity >= 0.20?}
    SimThresh -->|No| Drop1[Discard Chunk]
    SimThresh -->|Yes| Align[Calculate Combined Alignment Score]

    Align --> MarginFilter{Within 0.25 of Top Score?}
    MarginFilter -->|No| Drop2[Discard Chunk]
    MarginFilter -->|Yes| Substantive{Substantive Support?}
    Substantive -->|No| Drop3[Discard Chunk]
    Substantive -->|Yes| Dedup[Page-Level Deduplication]
    Dedup --> Cap[Cap at Max Citations: Default 4, Comparisons 6]
```

### Citation Scoring Formula
$$\text{Alignment Score} = (\text{Cosine Sim} \times 0.50) + (\text{Term Overlap Ratio} \times 0.30) + \text{Page Bonus} + \text{Number Bonus}$$

- **Cosine Similarity ($50\%$)**: Derived from Chroma distance: $\text{sim} = 1.0 - \frac{d^2}{2.0}$.
- **Term Overlap Ratio ($30\%$)**: Ratio of answer vocabulary found in chunk content (excluding code blocks and stop words).
- **Page Explicitly Cited Bonus ($+0.35$)**: Awarded if the LLM explicitly referenced the chunk's page number in its text (e.g., "Page 6").
- **Number Match Bonus ($+0.15$)**: Awarded when specific factual numbers (e.g., "810", "64", "500,000") in the answer match the chunk.
- **Code Block Isolation**: Fenced code blocks (` ```...``` `) are stripped before token extraction so programming variable names do not produce spurious citation matches.

---

## Visual, Chart, & PDF Understanding

Implemented in `services/visual_extractor.py`:

1. **Header Logo & Watermark Suppression**:
   - `detect_repeated_images` scans the PDF for image dimensions appearing on $\ge 3$ pages with height $< 150$ px or frequency $> 40\%$ of pages.
   - Decorative headers, corporate watermarks, and bullet icons are discarded.
2. **Quantitative Chart & Graph Parsing**:
   - PyMuPDF extracts embedded raster images or renders vector drawings.
   - RapidOCR extracts textual and numeric tokens with bounding boxes and centroid coordinates (`x_center`, `y_center`).
   - Vertical clustering separates stacked charts (e.g. Chart 1 vs Chart 2).
   - Horizontal and spatial proximity algorithms pair category labels (e.g. "MECH", "CIVIL", "ECE") with percentage or numerical tokens (e.g. "16%", "30%").
   - Dynamic regex extraction parses base population totals from narrative page text (e.g. "Total students = 4,500", "Girls = 2,000").
   - Deduces cross-category breakdowns (e.g. calculating total boys per discipline: $\text{Boys} = \text{Total Students} - \text{Girls}$).
3. **Scanned PDF Handling**:
   - If native page text length $< 50$ characters, the page is classified as scanned.
   - Renders page pixmap at 200 DPI and executes OCR over the entire canvas.
   - Stamped with average confidence scores. If confidence $< 0.45$, appends a notice indicating that the source text is partially degraded.

---

## Server-Sent Events (SSE) Streaming

The streaming endpoint `POST /chat/stream` implements real-time token delivery using standard HTTP Server-Sent Events (`text/event-stream`):

### Protocol Sequence
1. **Client Request**: `POST /chat/stream` with JSON payload `{"question": "...", "conversation_id": "..."}`.
2. **Header Negotiation**:
   - `Content-Type: text/event-stream`
   - `Cache-Control: no-cache`
   - `Connection: keep-alive`
   - `X-Accel-Buffering: no` (disables Nginx/proxy buffering)
3. **Token Stream**: Emitted token by token as returned from `ChatGroq.astream`:
   ```
   data: {"type": "token", "content": "The"}
   data: {"type": "token", "content": " Transformer"}
   ```
4. **Sources & Classification Event**: Emitted upon completion of LLM generation:
   ```
   data: {"type": "sources", "sources": [{"filename": "doc.pdf", "page_number": 1, "relevance_score": 0.88}], "answer_type": "DIRECTLY_GROUNDED"}
   ```
5. **Termination Event**: Emitted to signal graceful closure:
   ```
   data: {"type": "done", "conversation_id": "d3b07384-...", "session_id": "d3b07384-...", "answer_type": "DIRECTLY_GROUNDED"}
   ```
6. **Error Event**: Emitted on exception:
   ```
   data: {"type": "error", "content": "Rate limit exceeded"}
   ```

### Atomic Persistence Guarantee
Tokens are accumulated in memory as they stream to the client. The complete message is persisted to the PostgreSQL `messages` table **exactly once** in a single database transaction upon stream completion.

---

## Conversation Memory & Management

Maintained via PostgreSQL tables `conversations` and `messages`:

- **Context Loading**: `_load_history_for_llm` loads the 20 most recent messages for the conversation, sorted chronologically. Long assistant responses are condensed to 1,000 characters to prevent prompt exhaustion.
- **Auto-Titling**: On the initial message of a new conversation, the title is automatically generated from the first 57 characters of the question.
- **Custom Title Protection**: When a user renames a conversation via `PATCH /conversations/{id}`, the flag `is_custom_title` is set to `True`. Subsequent messages will never overwrite the custom title.
- **Pinned Ordering**:
  - `is_pinned: True` conversations are returned first.
  - Secondary sort order is `updated_at DESC`.
  - Backed by composite database index `ix_conversations_user_pin_updated`.
- **Bulk Deletion (`DELETE /conversations/bulk`)**:
  - Accepts a JSON array of conversation IDs.
  - Verifies that all IDs belong to the authenticated user.
  - Atomic execution: If any ID is invalid or belongs to another user, **no conversations are deleted** and a 404 is returned.
  - Database-level cascade deletes all related messages automatically.

---

## Server-Side PDF Conversation Export

Implemented in `services/pdf_export_service.py` via ReportLab:

- **Endpoint**: `GET /conversations/{conversation_id}/export/pdf`.
- **Memory Operation**: Operates entirely in RAM using `io.BytesIO`. No files are written to server storage during export.
- **Unicode Font Handling**: Automatically locates and registers DejaVu TrueType fonts (`DejaVuSans`, `DejaVuSansMono`) from matplotlib font assets. Provides full support for international currencies (₹, €, $), Greek letters (α, β, γ), arrows (→, ←), and mathematical symbols ($\le$, $\ge$, $\pm$, $\times$).
- **LaTeX Math Rendering**:
  - Detects inline math (`$...$`, `\(...\)`) and display math (`$$...$$`, `\[...\]`).
  - Renders formulas into high-resolution PNGs in memory via `matplotlib.mathtext.math_to_image`.
  - Embeds generated formulas as Platypus `Image` flowables with automatic vertical alignment.
- **Strict Code Block Protection**:
  - Pre-processing step masks fenced code blocks (` ```...``` `) and inline code (`` `...` ``) before math regex parsing.
  - LaTeX syntax inside code snippets remains verbatim monospace text and is never converted to math images.
- **Two-Pass `NumberedCanvas`**:
  - Intercepts canvas render events to calculate the total page count.
  - Draws running footers: `"DocChat AI — Exported Conversation"` on the left and `"Page X of Y"` on the right.

---

## Authentication & User Data Isolation

### User Identity Flow
```mermaid
sequenceDiagram
    autonumber
    actor User as Client Browser
    participant API as FastAPI Router
    participant Dep as get_current_user Dependency
    participant Cache as In-Memory User Cache (120s TTL)
    participant SQLite as SQLite auth.db
    participant PG as PostgreSQL users table

    User->>API: HTTP Request (Cookie: token=JWT)
    API->>Dep: Invoke get_current_user
    Dep->>Dep: Decode JWT & validate expiry
    Dep->>Cache: Check user_id in _user_cache
    alt Cache Hit (Valid TTL)
        Cache-->>Dep: Return cached user dict with pg_id
    else Cache Miss
        Dep->>SQLite: SELECT * FROM users WHERE id = user_id
        SQLite-->>Dep: User record
        Dep->>PG: ensure_pg_user (Find or Create by email)
        PG-->>Dep: PostgreSQL User (id=pg_id)
        Dep->>Cache: Store (monotonic_time + 120s, user_dict)
    end
    Dep-->>API: Inject authenticated user context (pg_id)
```

### Multi-User Isolation Guarantees
- **PostgreSQL**: Queries filter on `user_id == pg_user_id`. Attempting to access another user's document, conversation, or message returns `404 Not Found`.
- **ChromaDB**: Ingestion writes `metadata["user_id"] = str(pg_user_id)`. Vector queries enforce `filter={"user_id": str(pg_user_id)}`.
- **Supabase Storage**: Physical storage paths follow deterministic segmentation: `users/{pg_user_id}/documents/{document_id}/{filename}`.
- **Document Status**: `progress_tracker.get_status` verifies that the requesting user's `pg_user_id` matches the document owner.

---

## Database Architecture

The relational schema is managed in PostgreSQL (Supabase) and version-controlled via Alembic.

```mermaid
erDiagram
    users ||--o{ documents : owns
    users ||--o{ conversations : owns
    conversations ||--o{ messages : contains

    users {
        int id PK
        varchar name
        varchar email UK
        varchar password_hash
        timestamp created_at
        timestamp updated_at
    }

    documents {
        varchar(36) id PK
        int user_id FK
        varchar filename
        varchar storage_path
        bigint file_size
        varchar status
        int chunk_count
        int progress
        varchar processing_stage
        int processed_chunks
        text error_message
        timestamp created_at
        timestamp updated_at
    }

    conversations {
        varchar(36) id PK
        int user_id FK
        varchar title
        boolean is_pinned
        boolean is_custom_title
        timestamp created_at
        timestamp updated_at
    }

    messages {
        varchar(36) id PK
        varchar(36) conversation_id FK
        varchar role
        text content
        timestamp created_at
    }
```

### Table Specifications & Composite Indexes
1. **`users`**:
   - `id`: Integer primary key, autoincrementing.
   - `email`: `String(255)`, unique, indexed (`ix_users_email`).
2. **`documents`**:
   - `id`: `String(36)` UUID primary key.
   - `user_id`: Foreign key referencing `users.id` with `ON DELETE CASCADE`.
   - Index: `ix_documents_user_id_created_at` on `(user_id, created_at DESC)`.
3. **`conversations`**:
   - `id`: `String(36)` UUID primary key.
   - `user_id`: Foreign key referencing `users.id` with `ON DELETE CASCADE`.
   - `is_pinned`: Boolean default `False`.
   - `is_custom_title`: Boolean default `False`.
   - Index: `ix_conversations_user_pin_updated` on `(user_id, is_pinned DESC, updated_at DESC)`.
   - Index: `ix_conversations_user_id_updated_at` on `(user_id, updated_at DESC)`.
4. **`messages`**:
   - `id`: `String(36)` UUID primary key.
   - `conversation_id`: Foreign key referencing `conversations.id` with `ON DELETE CASCADE`.
   - Index: `ix_messages_conversation_id_created_at` on `(conversation_id, created_at ASC)`.

---

## Supabase Storage Configuration

Implemented in `services/storage_service.py`:
- **Target Bucket**: Private bucket configured by `SUPABASE_STORAGE_BUCKET` (default: `"documents"`).
- **Service Role Key**: Configured via `SUPABASE_SERVICE_ROLE_KEY`. Never exposed to client browsers or frontend code.
- **Bucket Creation**: `ensure_bucket` checks existence via REST API; if missing, automatically creates a **private bucket** (`"public": False`).
- **Deterministic Storage Paths**: Built via `build_storage_path`:
  ```
  users/{user_id}/documents/{document_id}/{sanitized_filename}
  ```
- **Signed URLs**: Generated with configurable expiration (default 300 seconds) for secure document preview/download.
- **Mock Store**: Includes an internal in-memory fallback store (`_mock_store`) allowing unit and integration test suites to execute offline without active Supabase cloud credentials.

---

## Document Processing Progress Tracking

Managed by `services/progress_service.py` via `ProgressTracker`:

| Stage | Default Progress | Description |
| :--- | :---: | :--- |
| `QUEUED` | 5% | Document accepted and queued for processing |
| `UPLOADING` | 15% | Physical file uploading to Supabase Storage |
| `UPLOADED` | 25% | File uploaded; parser and extractor initializing |
| `EXTRACTING` | 35% | PyMuPDF and visual extractors parsing pages |
| `PARSING` | 45% | Tables, diagrams, and OCR structures parsed |
| `CHUNKING` | 55% | Chunker generating element-aware passages |
| `EMBEDDING` | 55% – 85% | Generating vector embeddings in batches of 32 |
| `INDEXING` | 90% | Vectors and metadata written to ChromaDB |
| `FINALIZING` | 95% | Writing chunk counts and document records to DB |
| `COMPLETED` | 100% | Ingestion finished; ready for retrieval |
| `FAILED` | 0% | Ingestion failed; rollback executed |

- **Sub-Millisecond Polling**: The status endpoint queries in-memory active jobs (`_jobs`) and completed cache (`_completed_cache`) protected by thread locks, avoiding redundant database roundtrips.

---

## API Reference

All protected endpoints require an authenticated JWT cookie (`token`).

### Authentication (`/auth`)
| Method | Path | Auth Required | Description |
| :--- | :--- | :---: | :--- |
| `POST` | `/auth/signup` | No | Create an account with name, email, and password |
| `POST` | `/auth/signin` | No | Authenticate with email and password; sets HTTP-only cookie |
| `POST` | `/auth/signout` | Yes | Invalidate session by clearing HTTP-only cookie |
| `GET` | `/auth/me` | Yes | Retrieve profile details of authenticated user |
| `PATCH` | `/auth/me` | Yes | Update profile name (email is strictly read-only) |
| `POST` | `/auth/change-password` | Yes | Change user password (rejected for Google OAuth users) |
| `GET` | `/auth/config` | No | Return Google OAuth public status and client ID |
| `POST` | `/auth/google` | No | Authenticate via Google Identity Services ID token |
| `GET` | `/auth/google` | No | Redirect to Google OAuth 2.0 authorization screen |
| `GET` | `/auth/google/callback` | No | Google OAuth redirect callback; issues cookie and redirects to UI |

### Documents (`/`)
| Method | Path | Auth Required | Description |
| :--- | :--- | :---: | :--- |
| `POST` | `/upload` | Yes | Upload one or more documents (supports `?sync=true`) |
| `GET` | `/documents` | Yes | List metadata for documents owned by authenticated user |
| `GET` | `/documents/{id}/status` | Yes | Get real-time stage, progress percentage, and chunk count |
| `GET` | `/documents/{id}/download` | Yes | Download raw file from Supabase Storage |
| `GET` | `/documents/{id}/signed-url` | Yes | Generate short-lived signed download URL (300s TTL) |
| `DELETE` | `/documents/{id}` | Yes | Delete document, ChromaDB vectors, and Storage file |

### Conversations (`/conversations`)
| Method | Path | Auth Required | Description |
| :--- | :--- | :---: | :--- |
| `GET` | `/conversations` | Yes | List conversations (pinned first, then updated_at DESC) |
| `POST` | `/conversations` | Yes | Explicitly create a new conversation thread |
| `GET` | `/conversations/{id}` | Yes | Retrieve conversation details and message history |
| `PATCH` | `/conversations/{id}` | Yes | Rename conversation and/or update pinned status |
| `POST` | `/conversations/{id}/pin` | Yes | Pin conversation to keep at top of list |
| `POST` | `/conversations/{id}/unpin` | Yes | Unpin conversation |
| `GET` | `/conversations/{id}/export/pdf` | Yes | Export conversation as formatted PDF with math & citations |
| `DELETE` | `/conversations/{id}` | Yes | Delete conversation and cascade delete its messages |
| `DELETE` | `/conversations/bulk` | Yes | Atomically delete multiple conversations in one transaction |

### Chat (`/chat`)
| Method | Path | Auth Required | Description |
| :--- | :--- | :---: | :--- |
| `POST` | `/chat` | Yes | Non-streaming RAG question answering |
| `POST` | `/chat/stream` | Yes | Server-Sent Events (SSE) streaming RAG response |

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and configure the following variables:

| Variable | Description | Required | Example / Default |
| :--- | :--- | :---: | :--- |
| `GROQ_API_KEY` | API key from Groq Console for LLM generation | **Yes** | `gsk_...` |
| `GROQ_MODEL` | Primary LLM model identifier on Groq | No | `openai/gpt-oss-120b` |
| `DATABASE_URL` | PostgreSQL connection string (Supabase) | **Yes** | `postgresql://user:pass@db.ref.supabase.co:5432/postgres` |
| `SUPABASE_URL` | Supabase project URL | Optional* | `https://your-ref.supabase.co` (*inferred if omitted) |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase Service Role Key (kept secret) | **Yes** | `eyJhbGciOi...` |
| `SUPABASE_STORAGE_BUCKET` | Storage bucket name for documents | No | `documents` |
| `EMBEDDING_MODEL` | Local HuggingFace embedding model name | No | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | Local directory for ChromaDB vector storage | No | `./chroma_db` |
| `UPLOAD_DIR` | Local scratch directory for temporary files | No | `./uploads` |
| `MAX_FILE_SIZE_MB` | Maximum permitted file upload size in MB | No | `50` |
| `CHUNK_SIZE` | Target chunk size in characters | No | `800` |
| `CHUNK_OVERLAP` | Overlap between sequential chunks | No | `200` |
| `RETRIEVAL_TOP_K` | Number of candidate chunks retrieved | No | `6` |
| `SIMILARITY_THRESHOLD` | Minimum similarity cutoff for citations | No | `0.20` |
| `CITATION_MARGIN` | Relative margin below top score for citations | No | `0.25` |
| `MAX_CITATIONS` | Maximum citations returned per response | No | `4` |
| `JWT_SECRET_KEY` | Secret key for signing authentication JWTs | **Yes** | `use-a-strong-random-secret` |
| `JWT_EXPIRY_HOURS` | Token lifespan in hours | No | `72` |
| `AUTH_DB_PATH` | File path for local SQLite auth database | No | `./auth.db` |
| `GOOGLE_CLIENT_ID` | Google OAuth 2.0 Web Client ID | Optional | `your-google-client-id.apps.googleusercontent.com` |
| `GOOGLE_CLIENT_SECRET` | Google OAuth 2.0 Web Client Secret | Optional | `GOCSPX-...` |
| `GOOGLE_REDIRECT_URI` | Google OAuth callback URL | Optional | `http://localhost:8001/auth/google/callback` |
| `FRONTEND_URL` | URL of the React/Vite user interface | No | `http://localhost:5173` |
| `CORS_ORIGINS` | Comma-separated list of permitted origins | No | `http://localhost:5173,http://localhost:3000` |

---

## Local Development Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Active PostgreSQL database (e.g. Supabase project)
- Free Groq Cloud API key ([console.groq.com](https://console.groq.com/keys))

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/suyashmudgal/RAG_CHAT_BACKEND.git
   cd RAG_CHAT_BACKEND/backend
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your GROQ_API_KEY, DATABASE_URL, and JWT_SECRET_KEY
   ```

5. **Execute Database Migrations**:
   ```bash
   alembic upgrade head
   ```

6. **Start the Development Server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. **Verify Health Endpoint**:
   Open `http://localhost:8000/` in your browser. Expected response:
   ```json
   {
     "status": "healthy",
     "service": "DocChat AI Backend",
     "version": "1.0.0"
   }
   ```
   Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## Database Migrations (Alembic)

The repository uses Alembic for tracking PostgreSQL schema changes:

- **Upgrade to Latest Version**:
  ```bash
  alembic upgrade head
  ```
- **Downgrade One Revision**:
  ```bash
  alembic downgrade -1
  ```
- **Generate New Migration**:
  ```bash
  alembic revision --autogenerate -m "description_of_changes"
  ```

### Migration History
1. `8330f427d2ee_create_initial_tables.py`: Creates `users`, `documents`, `conversations`, and `messages` tables with primary foreign keys and indices.
2. `9ffbbc8ff6d7_add_indexes_for_conversations_and_.py`: Adds single-column indexes on `conversations.updated_at` and `messages.created_at`.
3. `d4e5f6a7b8c9_phase6a_progress_and_indexes.py`: Adds `progress`, `processing_stage`, `processed_chunks`, and `error_message` columns to `documents`. Adds composite indexes `ix_conversations_user_id_updated_at`, `ix_messages_conversation_id_created_at`, and `ix_documents_user_id_created_at`.
4. `e5f6a7b8c9d0_phase7a_pinned_and_custom_title.py`: Adds `is_pinned` and `is_custom_title` flags to `conversations` with composite index `ix_conversations_user_pin_updated`.

---

## Performance Optimizations

1. **In-Memory User Context Cache**:
   - *Problem*: High-frequency authenticated requests (e.g. streaming, progress polling) triggered continuous database lookups to synchronize SQLite with PostgreSQL.
   - *Solution*: Added thread-locked `_user_cache` with a 120-second TTL in `auth/dependencies.py`.
   - *Impact*: Reduces authenticated route overhead by eliminating repeated remote database roundtrips.
2. **Database Composite Indexes & Column Projection**:
   - *Problem*: Loading conversations and documents for high-volume users caused full table scans and unbounded JSON payloads.
   - *Solution*: Added multi-column composite indexes (`ix_conversations_user_pin_updated`, `ix_documents_user_id_created_at`) and replaced `session.query(Model).all()` with explicit column projections (`id`, `title`, `updated_at`, etc.).
   - *Impact*: Sub-millisecond indexed database queries and minimal serialization overhead.
3. **Eager Startup Initialization**:
   - *Problem*: Incurred a 5–10 second cold start penalty on the initial upload while downloading and initializing the embedding model.
   - *Solution*: Initialized `all-MiniLM-L6-v2` and `VectorStore` eagerly within the `lifespan` handler in `main.py`.
   - *Impact*: The system is warm and fully operational before the first user request arrives.
4. **Batch Embedding & Vector Insertion**:
   - *Problem*: Inserting chunks individually created high I/O overhead in ChromaDB.
   - *Solution*: Batched embedding and vector insertion in chunks of 32 (`batch_size=32`).
   - *Impact*: Linear reduction in vector indexing time with granular progress reporting.
5. **Threadpool Offloading**:
   - *Problem*: CPU-bound report generation (ReportLab, Matplotlib LaTeX rendering) and blocking SQLAlchemy operations threatened to block the asyncio event loop.
   - *Solution*: Offloaded blocking tasks to dedicated worker threads via `asyncio.to_thread`.
   - *Impact*: FastAPI event loop remains responsive during heavy export operations.

---

## Engineering Challenges & Verified Fixes

1. **Windows PyTorch / PyArrow CRT DLL Collision**:
   - *Problem*: On Windows operating systems, importing PyTorch after PyArrow caused an immediate access violation crash (`0xC0000005`) due to conflicting C-runtime DLLs.
   - *Fix*: Added a top-level guard in `main.py`, `services/embedding_service.py`, and `services/vector_store.py` importing `pyarrow` before PyTorch loads.
2. **Exaggerated Certainty in LLM Summaries**:
   - *Problem*: Groq LLM tended to upgrade source qualifiers during summarization (e.g. converting "potential markets" to "guaranteed markets").
   - *Fix*: Implemented `sanitize_grounded_claims` in `services/chat_service.py`, using regex filters to detect and revert ungrounded certainty claims.
3. **LaTeX Math Rendering Corrupting Code Blocks in PDF Export**:
   - *Problem*: Inline LaTeX regex patterns (`$...$`) inadvertently matched dollar signs and variable assignments inside fenced programming code blocks.
   - *Fix*: Implemented two-stage code block masking in `services/pdf_export_service.py`. Code blocks are extracted and replaced with unique UUID tokens before math parsing, then restored as literal monospace flowables.
4. **Multi-Document Comparison Starvation**:
   - *Problem*: Standard vector search on resume comparisons returned all chunks from the single largest document, starving the other candidates.
   - *Fix*: Created `similarity_search_multi_doc_for_user` in `services/vector_store.py`, enforcing balanced round-robin retrieval across candidate document IDs.
5. **Atomic Bulk Conversation Deletion Protection**:
   - *Problem*: Partial deletions occurred if an unauthorized or non-existent conversation ID was mixed into a bulk delete request.
   - *Fix*: Implemented strict pre-validation in `routes/conversations.py`. If any ID in the request array is missing or belongs to another user, a 404 is raised immediately with zero database modifications.

---

## Testing

The repository contains an automated test suite verifying correctness, performance, and security across all subsystems.

### Test Execution Commands
```bash
# Run complete test suite via pytest
pytest

# Execute specific integration test suites
python test_phase6_performance.py
python test_phase7_performance_chat_management.py
python test_bulk_delete.py
python test_pdf_export_comprehensive.py
python test_rag_groundedness.py
python test_visual_pdf_understanding.py
python test_supabase_storage.py
python test_user_isolation.py
python test_google_auth.py
```

### Verified Test Suites
- `test_phase6_performance.py`: Ingestion state machine stages, non-blocking upload, rollback on failure, concurrent user uploads, status endpoint.
- `test_phase7_performance_chat_management.py`: Conversation rename, pin/unpin, pinned ordering, PDF export isolation, title preservation.
- `test_bulk_delete.py`: Multi-conversation deletion, deduplication, cascade verification, unauthorized ID rollback.
- `test_pdf_export_comprehensive.py`: Plain text, Markdown tables, inline/display math, LaTeX-in-code protection, UTF-8 symbols, multi-page footers.
- `test_rag_groundedness.py`: Source claim verification, inference labeling, prevention of unsupported claims, prompt guidance.
- `test_visual_pdf_understanding.py`: Chart understanding, categorical calculations, scanned page OCR, table extraction.
- `test_supabase_storage.py`: Path construction, sanitization, private bucket enforcement, download fallbacks, signed URLs.
- `test_user_isolation.py`: Cross-user document, chat, vector, and status isolation checks.

---

## Deployment

> **Deployment configuration was not verified in this repository.**
> No `Dockerfile`, `render.yaml`, `Procfile`, `railway.json`, or cloud deployment templates exist in the codebase.

### Production Startup Command
When deploying to a container or virtual machine, run the application using a production ASGI server configuration:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Security Architecture

- **Authentication & Tokens**: Signed JWTs with 72-hour expiration delivered via HTTP-only cookies (`SameSite=Lax`, `Path=/`).
- **Authorization & Isolation**: Ownership is enforced at the database level using `user_id`. Queries never accept `user_id` from client payloads.
- **Private Storage**: Supabase Storage bucket configured as private (`"public": False`). Downloads require authentication or time-limited signed URLs (300s TTL).
- **Path Traversal Protection**: `sanitize_filename` removes relative path patterns (`../`), backslashes, and special characters.
- **Credential Protection**: Database passwords and Supabase Service Role keys are loaded via environment variables and never logged or exposed via endpoints.
- **Exception Sanitization**: Global exception handlers and progress state machines sanitize error messages to prevent database connection strings or stack traces from reaching clients.

---

## Current Limitations

1. **Local Vector Database Storage**: ChromaDB runs locally on the host filesystem (`./chroma_db`). In horizontally scaled multi-instance deployments, a centralized vector database would be required.
2. **Single Primary LLM Provider**: The primary LLM integration relies on Groq Cloud APIs. If Groq experiences service degradation, generation depends on the configured fallback models.
3. **Visual Processing Constraints**: Handwritten text, complex 3D diagrams, and low-contrast scanned documents may exhibit lower OCR accuracy compared to high-resolution native PDF text.
4. **Separate Frontend Repository**: The frontend application is maintained independently, requiring manual alignment of CORS origins and cookie configurations during deployment.

---

## Potential Future Improvements

*(Based on project architecture and engineering opportunities)*

- Containerization via a multi-stage `Dockerfile` and `docker-compose.yml` defining PostgreSQL, ChromaDB, and backend services.
- Integration of an external managed vector store (e.g. pgvector in Supabase or Qdrant Cloud) for horizontal backend scaling.
- Webhook-based or WebSocket-based real-time progress notifications to eliminate client polling of the `/status` endpoint.
- Asynchronous Celery / Redis task queue for document ingestion in high-concurrency environments.

---

## License

This project is licensed under the MIT License.
