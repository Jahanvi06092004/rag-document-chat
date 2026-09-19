# RAG-Based Document Q&A System

**A Retrieval-Augmented Generation (RAG) system for answering questions from custom documents using semantic search, vector embeddings, ChromaDB, and Groq-powered LLM generation.**

Python • LangChain • HuggingFace • ChromaDB • Groq

---

## Overview

**RAG-Based Document Q&A System** is a document-based question answering system built using the **Retrieval-Augmented Generation (RAG)** architecture.

The system converts a collection of documents into a searchable knowledge base. During ingestion, documents are loaded, divided into smaller overlapping chunks, converted into semantic embeddings using the **HuggingFace `all-MiniLM-L6-v2` model**, and stored in **ChromaDB**.

When a user asks a question, the system performs semantic similarity search to retrieve the most relevant document chunks. The retrieved context is then combined with the user's query and passed to an LLM through the **Groq API** to generate a grounded response.

The main goal of the system is to retrieve relevant information from the provided documents before generating an answer, rather than relying only on the LLM's pretrained knowledge.

---

# Key Features

## Document Ingestion

- Loads documents from a dedicated `docs/` directory
- Supports `.txt` documents in the current ingestion pipeline
- Automatically loads multiple text files
- Preserves document metadata
- Validates the document directory and available files

## Text Processing

- Splits large documents into smaller chunks
- Uses `RecursiveCharacterTextSplitter`
- Configurable chunk size and overlap
- Default chunk size: **1000 characters**
- Default chunk overlap: **100 characters**

## Semantic Embeddings

- Uses HuggingFace Sentence Transformers
- Embedding model: `all-MiniLM-L6-v2`
- Converts text chunks into vector representations
- Uses the same embedding model for query retrieval

## Vector Database

- Uses **ChromaDB** for persistent vector storage
- Stores document embeddings and corresponding text chunks
- Supports semantic similarity search
- Uses cosine similarity for vector comparison

## Retrieval

- Converts the user query into an embedding
- Performs semantic similarity search
- Retrieves the **top 5 relevant document chunks**
- Uses retrieved chunks as context for the LLM

## LLM Generation

- Uses the **Groq API** for LLM inference
- Combines the user query with retrieved document context
- Generates responses based on the retrieved information
- Instructs the LLM not to answer beyond the available document context

---

# System Architecture

```text
                         ┌─────────────────────────┐
                         │     Source Documents    │
                         │        docs/*.txt       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Document Loader     │
                         │ DirectoryLoader/TextLoader│
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      Text Chunking      │
                         │  Chunk Size: 1000       │
                         │  Overlap: 100            │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Embedding Generation  │
                         │    all-MiniLM-L6-v2     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │        ChromaDB         │
                         │    Vector Database      │
                         └────────────┬────────────┘
                                      │
══════════════════════════════════════════════════════════════
                         Retrieval Pipeline
══════════════════════════════════════════════════════════════

                                      │
                                User Query
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Query Embedding      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Semantic Retrieval    │
                         │       Top 5 Chunks      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Query + Context      │
                         │    Prompt Construction  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │        Groq API         │
                         │      LLM Inference      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      Final Answer       │
                         └─────────────────────────┘
```

---

# Project Structure

```text
RAG-PYTHON
│
├── db/
│   └── chroma_db/
│       ├── <chroma-collection-id>/
│       └── chroma.sqlite3
│
├── docs/
│   ├── attention-is-all-you-need.pdf
│   ├── Google.txt
│   ├── Microsoft.txt
│   ├── Nvidia.txt
│   ├── SpaceX.txt
│   └── Tesla.txt
│
├── venv/
│
├── .env
├── ingestion_pipeline.py
└── retriever_pipeline.py
```

### File Description

| File / Folder | Purpose |
|---|---|
| `docs/` | Contains the source documents used as the knowledge base |
| `db/chroma_db/` | Persistent ChromaDB vector database |
| `ingestion_pipeline.py` | Loads, chunks, embeds, and stores documents |
| `retriever_pipeline.py` | Retrieves relevant chunks and generates answers |
| `.env` | Stores API credentials and environment variables |
| `venv/` | Python virtual environment |

---

# Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| RAG Framework | LangChain |
| Document Loading | DirectoryLoader, TextLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embedding Model | `all-MiniLM-L6-v2` |
| Embedding Framework | HuggingFace Sentence Transformers |
| Vector Database | ChromaDB |
| Similarity Metric | Cosine Similarity |
| LLM Inference | Groq API |
| Environment Management | python-dotenv |

---

# How RAG Works in This Project

The project consists of two main pipelines:

```text
Ingestion Pipeline
        +
Retrieval Pipeline
```

---

## 1. Ingestion Pipeline

The ingestion pipeline prepares the documents for retrieval.

### Step 1 — Load Documents

The system scans the `docs/` directory and loads `.txt` files.

```text
Documents
    ↓
DirectoryLoader
    ↓
Loaded Documents
```

---

## Step 2 — Split Documents

Large documents are divided into smaller chunks using:

```text
RecursiveCharacterTextSplitter
```

Default configuration:

```text
Chunk Size    : 1000
Chunk Overlap : 100
```

Chunking allows the retrieval system to work with smaller, more relevant sections of a document.

---

## Step 3 — Generate Embeddings

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The process is:

```text
Text Chunk
    ↓
Embedding Model
    ↓
Vector Representation
```

The vector represents the semantic meaning of the text.

---

## Step 4 — Store in ChromaDB

The generated embeddings and document chunks are stored in ChromaDB.

```text
Document Chunk
       +
Embedding
       ↓
   ChromaDB
```

The database is persisted locally inside:

```text
db/chroma_db/
```

---

# Retrieval Pipeline

Once the vector database has been created, the retrieval pipeline can answer user questions.

## Step 1 — User Query

Example:

```text
How much did Microsoft pay to acquire GitHub?
```

---

## Step 2 — Query Embedding

The question is represented using the same embedding model:

```text
User Query
    ↓
all-MiniLM-L6-v2
    ↓
Query Vector
```

---

## Step 3 — Semantic Search

The query vector is compared against the vectors stored in ChromaDB.

The system retrieves the:

```text
Top 5 Most Relevant Chunks
```

based on semantic similarity.

---

## Step 4 — Context Construction

The retrieved document chunks are combined with the user's question.

```text
User Query
     +
Retrieved Documents
     ↓
Contextual Prompt
```

The prompt instructs the LLM to answer using the provided documents.

---

## Step 5 — Groq LLM

The contextual prompt is sent to an LLM through the **Groq API**.

```text
Query + Retrieved Context
            ↓
        Groq API
            ↓
        LLM Response
```

The generated response is then returned to the user.

---

# Example

Suppose the user asks:

```text
How much did Microsoft pay to acquire GitHub?
```

The system performs:

```text
User Question
      ↓
Question Embedding
      ↓
ChromaDB Similarity Search
      ↓
Top 5 Relevant Chunks
      ↓
Question + Retrieved Context
      ↓
Groq LLM
      ↓
Final Answer
```

Instead of searching the entire document collection manually, the system retrieves the most relevant information first and then uses the LLM to formulate the answer.

---

# Example Questions

The current document collection can be used for questions such as:

```text
What was NVIDIA's first graphics accelerator called?

Which company did NVIDIA acquire to enter the mobile processor market?

What was Microsoft's first hardware product release?

How much did Microsoft pay to acquire GitHub?

In what year did Tesla begin production of the Roadster?

Who succeeded Ze'ev Drori as CEO in October 2008?

What was the name of the autonomous spaceport drone ship
that achieved the first successful sea landing?

What was the original name of Microsoft before it became Microsoft?
```

---

# Installation

## 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG-PYTHON
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace `your_groq_api_key` with your actual Groq API key.

**Do not upload the `.env` file to GitHub.**

Add it to `.gitignore`:

```text
.env
venv/
__pycache__/
db/chroma_db/
```

---

# Usage

## Step 1 — Add Documents

Place your supported `.txt` documents inside:

```text
docs/
```

Example:

```text
docs/
├── Google.txt
├── Microsoft.txt
├── Nvidia.txt
├── SpaceX.txt
└── Tesla.txt
```

---

## Step 2 — Create the Vector Database

Run:

```bash
python ingestion_pipeline.py
```

The pipeline performs:

```text
Load Documents
      ↓
Split Documents
      ↓
Generate Embeddings
      ↓
Store Embeddings in ChromaDB
```

The vector database will be created inside:

```text
db/chroma_db/
```

---

## Step 3 — Run the Retrieval Pipeline

Run:

```bash
python retriever_pipeline.py
```

The system will:

```text
Load ChromaDB
      ↓
Receive User Query
      ↓
Retrieve Top 5 Relevant Chunks
      ↓
Construct Context
      ↓
Send Context to Groq
      ↓
Generate Final Answer
```

---

# Why RAG?

A traditional LLM generates answers primarily from its pretrained knowledge.

A RAG system adds an external knowledge source:

```text
Traditional LLM

User Query
    ↓
   LLM
    ↓
Answer
```

Whereas this project follows:

```text
RAG

User Query
    ↓
Semantic Retrieval
    ↓
Relevant Documents
    ↓
Query + Context
    ↓
LLM
    ↓
Grounded Answer
```

This makes the system better suited for answering questions about a specific document collection.

---

# Design Philosophy

> **Retrieve first. Generate later.**

The system follows a retrieval-first architecture where relevant document context is retrieved before the LLM generates a response.

The prompt also instructs the model to use only the provided documents and to indicate when the available information is insufficient.

---

# Future Improvements

- [ ] PDF document ingestion
- [ ] DOCX document support
- [ ] Multiple document formats
- [ ] Metadata-based filtering
- [ ] Similarity score thresholding
- [ ] Hybrid keyword + vector search
- [ ] Reranking retrieved documents
- [ ] Source citations in generated answers
- [ ] Conversation memory
- [ ] Streaming responses
- [ ] Web-based chat interface
- [ ] REST API
- [ ] Docker deployment
- [ ] RAG evaluation metrics
- [ ] Cloud deployment

---

# Learning Outcomes

Through this project, the following concepts were implemented:

- Retrieval-Augmented Generation
- Document preprocessing
- Text chunking
- Semantic embeddings
- Vector databases
- Similarity search
- Context retrieval
- Prompt construction
- LLM-based generation
- Grounded question answering

---

# Author

**Jahanvi Mishra**

B.Tech — Mechanical Engineering 
National Institute of Technology Agartala



---

# License

This project is developed for educational and portfolio purposes.
