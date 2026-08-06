"""
PDF Question Answering Application (RAG)
Epochs '26 — Assignment 11

Upload a PDF, ask questions about it, and get context-aware answers.
Conversation memory lets you ask natural follow-up questions.

Stack:
    - LangChain            (orchestration)
    - PyPDFLoader           (PDF loading)
    - Google Gemini API     (embeddings + LLM, free tier)
    - ChromaDB              (vector store)
    - Gradio                (UI)
"""

import os
import shutil
import tempfile

import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
VECTOR_DB_DIR = os.path.join(tempfile.gettempdir(), "pdf_rag_chroma_db")

# ---------------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------------

def _extract_text(response) -> str:
    """Normalize an LLM response's .content into a plain string.

    Some providers (e.g. Gemini via langchain-google-genai) can return
    `.content` as a list of content-part dicts instead of a plain string.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


class PDFChatBot:
    def __init__(self):
        self.vectordb = None
        self.retriever = None
        self.llm = None
        self.chat_history = []  # list of (question, answer) tuples
        self.pdf_name = None

    def _fresh_vector_dir(self):
        # Wipe any previous session's vector store so PDFs don't mix.
        if os.path.exists(VECTOR_DB_DIR):
            shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    def load_pdf(self, file_path: str) -> str:
        if not GOOGLE_API_KEY:
            return (
                "⚠️ No GOOGLE_API_KEY found. Add it to a .env file "
                "(see .env.example) before chatting."
            )

        try:
            self._fresh_vector_dir()

            # 1. Load
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # 2. Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            chunks = splitter.split_documents(documents)

            # 3. Embed + store in ChromaDB (Gemini embeddings — no local model,
            #    keeps memory usage low on free-tier hosting)
            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY,
                model=EMBEDDING_MODEL,
            )
            self.vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=VECTOR_DB_DIR,
            )
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 4})

            # 4. LLM (Google Gemini — free tier)
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=GOOGLE_API_KEY,
                model=GEMINI_MODEL,
                temperature=0.2,
            )

            # 5. Fresh conversation memory for this document
            self.chat_history = []

            self.pdf_name = os.path.basename(file_path)
            return f"✅ Loaded **{self.pdf_name}** — {len(chunks)} chunks indexed. Ask away!"

        except Exception as e:
            return f"❌ Error processing PDF: {e}"

    def _condense_question(self, question: str) -> str:
        """Rewrite a follow-up question into a standalone one using chat history."""
        if not self.chat_history:
            return question

        history_text = "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in self.chat_history[-5:]
        )
        prompt = (
            "Given the conversation history and a follow-up question, rephrase "
            "the follow-up question into a standalone question that includes "
            "any necessary context from the history. If it is already "
            "standalone, return it unchanged. Reply with ONLY the rephrased "
            "question, no preamble.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Follow-up question: {question}\n\n"
            "Standalone question:"
        )
        response = self.llm.invoke(prompt)
        return _extract_text(response).strip() or question

    def ask(self, question: str):
        if self.retriever is None or self.llm is None:
            return "Please upload and process a PDF first."
        if not question.strip():
            return "Please enter a question."

        try:
            # 1. Condense with chat history so follow-ups resolve correctly
            standalone_question = self._condense_question(question)

            # 2. Retrieve relevant chunks
            docs = self.retriever.invoke(standalone_question)
            context = "\n\n".join(d.page_content for d in docs)

            # 3. Generate a grounded answer
            answer_prompt = (
                "Answer the question using only the context below, which is "
                "extracted from a PDF document. If the answer isn't contained "
                "in the context, say you don't know based on the document — "
                "don't make anything up.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
            response = self.llm.invoke(answer_prompt)
            answer = _extract_text(response)

            # 4. Append source pages
            if docs:
                pages = sorted({str(d.metadata.get("page", "?")) for d in docs})
                answer += f"\n\n*Source page(s): {', '.join(pages)}*"

            # 5. Update memory
            self.chat_history.append((question, answer))

            return answer
        except Exception as e:
            return f"❌ Error generating answer: {e}"

    def reset(self):
        self.vectordb = None
        self.retriever = None
        self.llm = None
        self.chat_history = []
        self.pdf_name = None
        if os.path.exists(VECTOR_DB_DIR):
            shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)


bot = PDFChatBot()

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def handle_upload(file):
    if file is None:
        return "Please upload a PDF.", []
    status = bot.load_pdf(file)
    return status, []


def handle_chat(message, history):
    if not message:
        return history, ""
    answer = bot.ask(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, ""


def handle_clear():
    bot.reset()
    return None, "Upload a PDF to get started.", []


with gr.Blocks(title="PDF Q&A with RAG") as demo:
    gr.Markdown(
        """
        # 📄 PDF Question Answering (RAG)
        Upload a PDF, then ask questions about its content. Follow-up
        questions work naturally thanks to conversation memory.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            status_box = gr.Markdown("Upload a PDF to get started.")
            clear_btn = gr.Button("Clear / Start Over")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=450)
            question_box = gr.Textbox(
                label="Ask a question",
                placeholder="e.g. What is this document about?",
            )
            send_btn = gr.Button("Send", variant="primary")

    pdf_input.change(fn=handle_upload, inputs=pdf_input, outputs=[status_box, chatbot])
    send_btn.click(fn=handle_chat, inputs=[question_box, chatbot], outputs=[chatbot, question_box])
    question_box.submit(fn=handle_chat, inputs=[question_box, chatbot], outputs=[chatbot, question_box])
    clear_btn.click(fn=handle_clear, outputs=[pdf_input, status_box, chatbot])

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
