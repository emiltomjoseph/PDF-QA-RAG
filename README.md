# DocuMind AI

DocuMind AI is a Conversational Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF document and ask natural language questions about its contents.

The application extracts text from the uploaded PDF, converts it into embeddings, stores them in a Chroma vector database, retrieves relevant information for each query, and uses Google's Gemini model to generate accurate, context-aware responses.

---

## Features

- Upload any PDF document
- Automatic text extraction
- Intelligent document chunking
- Vector embeddings using Sentence Transformers
- ChromaDB vector database
- Retrieval-Augmented Generation (RAG)
- Context-aware conversation
- Interactive Gradio web interface

---

## Tech Stack

### Frontend

- Gradio

### Backend

- Python

### AI Model

- Google Gemini 3.5 Flash Lite

### Embedding Model

- sentence-transformers/all-MiniLM-L6-v2

### Vector Database

- ChromaDB

### Libraries

- LangChain
- PyMuPDF
- Python Dotenv

---

## How It Works

```
User uploads PDF
        │
        ▼
Extract Text (PyMuPDF)
        │
        ▼
Split into Chunks
        │
        ▼
Generate Embeddings
        │
        ▼
Store in ChromaDB
        │
        ▼
Retrieve Relevant Chunks
        │
        ▼
Conversation History
        │
        ▼
Gemini 3.5 Flash Lite
        │
        ▼
Context-Aware Response
```

---

## Usage

1. Launch the application.
2. Upload a PDF document.
3. Wait for the document to be indexed.
4. Ask questions in natural language.
5. Continue asking follow-up questions—the chatbot maintains conversation context for a more natural experience.

---

## License

This project is intended for educational and learning purposes.