import re

import fitz
from dotenv import load_dotenv
from google import genai

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()


class PDFChatbot:

    def __init__(self):

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = genai.Client()

        self.db = None

        self.chat_history = []

    # --------------------------------------------------
    # Load PDF
    # --------------------------------------------------

    def load_pdf(self, pdf_path):

        pdf = fitz.open(pdf_path)

        text = ""

        for page in pdf:
            text += page.get_text()

        pdf.close()

        text = re.sub(r"\n{2,}", "\n", text)

        return [Document(page_content=text)]

    # --------------------------------------------------
    # Create Vector Database
    # --------------------------------------------------

    def create_vector_db(self, pdf_path):

        documents = self.load_pdf(pdf_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        self.db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory="chroma_db"
        )

        self.chat_history = []

        return len(chunks)

    # --------------------------------------------------
    # Load Existing Database
    # --------------------------------------------------

    def load_vector_db(self):

        self.db = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embedding_model
        )

    # --------------------------------------------------
    # Ask Question
    # --------------------------------------------------

    def ask(self, question):

        if self.db is None:
            self.load_vector_db()

        docs = self.db.max_marginal_relevance_search(
            question,
            k=4,
            fetch_k=10
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        history = ""

        for user, assistant in self.chat_history[-5:]:

            history += f"""
User: {user}

Assistant: {assistant}

"""

        prompt = f"""
You are DocuMind AI.

You are answering questions about a PDF.

Use BOTH the conversation history and the PDF context.

If the current question depends on previous questions,
use the conversation history to understand it.

Examples:

User:
Who is this document about?

Assistant:
This document is about Emil Tom Joseph.

User:
Where does he study?

Assistant:
He studies at Amal Jyothi College of Engineering.

Never invent information.

If the answer is not present in the context, reply exactly:

I couldn't find that information in the document.

------------------------------------

Conversation History

{history}

------------------------------------

PDF Context

{context}

------------------------------------

Current Question

{question}

Answer:
"""

        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        answer = response.text.strip()

        self.chat_history.append(
            (question, answer)
        )

        return answer


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    bot = PDFChatbot()

    chunks = bot.create_vector_db("Emil Tom Joseph - Resume.pdf")

    print(f"\nPDF Indexed Successfully ({chunks} chunks)\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        answer = bot.ask(question)

        print("\nAssistant:\n")

        print(answer)

        print("\n" + "=" * 70 + "\n")