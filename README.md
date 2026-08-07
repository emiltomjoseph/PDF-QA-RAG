# 📄 PDF Question Answering App (RAG) — Epochs '26, Day 11

**Participant Name:** Emil Tom Joseph
**MUID:** emiltomjoseph@mulearn

## Project Overview
A Retrieval-Augmented Generation (RAG) application that lets a user upload a
PDF and ask natural-language questions about its content. The app loads the
PDF, splits it into chunks, embeds those chunks, stores them in a vector
database, and retrieves the most relevant chunks to ground each LLM answer.
Conversation history is preserved so follow-up questions ("what about the
second one?") resolve correctly without the user having to repeat context.

## Technologies Used
| Purpose            | Tool                                             |
|---------------------|--------------------------------------------------|
| Orchestration       | LangChain                                         |
| PDF loading         | PyPDFLoader                                       |
| Embeddings          | Google Gemini API (`gemini-embedding-001`, free tier)|
| Vector store        | ChromaDB                                          |
| LLM                 | Google Gemini API (free tier, `gemini-2.5-flash`) |
| UI                  | Gradio                                            |

## Memory Implementation
Conversation memory is implemented manually as a running list of
`(question, answer)` turns kept on the chatbot instance, combined with a
two-step "condense-then-retrieve" flow on every question:
1. If prior turns exist, the LLM rewrites the new question into a
   standalone question using the last few turns of history (so pronouns
   and references like "that" or "the second one" resolve).
2. The standalone question is used to retrieve relevant chunks from
   ChromaDB.
3. The LLM answers using only the retrieved context, and the turn is
   appended to history for the next follow-up.

This avoids relying on LangChain's older prebuilt chain/memory classes
(`ConversationalRetrievalChain`, `ConversationBufferMemory`), which were
removed in recent LangChain releases — the implementation here only depends
on stable `retriever.invoke()` / `llm.invoke()` calls, so it isn't tied to
a specific LangChain version.

This means a question like "what does it say about chapter 2?" followed by
"summarize that in one line" works naturally — the second question is
understood in light of the first.

## How to Run Locally
```bash
git clone <this-repo-url>
cd pdf-rag-qa
pip install -r requirements.txt
cp .env.example .env        # then add your free Gemini API key
python app.py
```
The app opens in your browser via Gradio (default: http://127.0.0.1:7860).

Get a free Google Gemini API key at https://aistudio.google.com/app/apikey.

## Deployment
Deployed on **Hugging Face Spaces** (Gradio SDK).
Deployment link: *(add your public Space URL here)*

To deploy on Hugging Face Spaces:
1. Create a new Space → SDK: Gradio.
2. Upload `app.py`, `requirements.txt`.
3. In Space Settings → Secrets, add `GOOGLE_API_KEY`.
4. Space builds automatically and serves `app.py`.

## Challenges Faced
- Free-tier hosting (Render's 512MB free instance) ran out of memory loading
  a local Sentence Transformers model, since it pulls in PyTorch. Switched
  to Gemini's embedding API (`gemini-embedding-001`) instead of a local model,
  removing the PyTorch dependency entirely and keeping memory usage low
  enough for free hosting.
- LangChain's 1.0 release removed several legacy prebuilt chains/memory
  classes (`ConversationalRetrievalChain`, `ConversationBufferMemory`).
  Rather than pin to an old version, the app implements the
  condense-question + retrieve + answer flow manually with plain
  `retriever.invoke()` / `llm.invoke()` calls, which keeps it working across
  LangChain versions.
- Keeping retrieval grounded so the model doesn't answer from general
  knowledge when the PDF doesn't cover the question — mitigated by showing
  retrieved source page numbers alongside each answer for transparency.
- Making follow-up questions resolve correctly required an explicit
  question-condensing step rather than just appending raw history to the
  prompt.
- Keeping the app free end-to-end: local Sentence Transformers embeddings
  (no API cost) + Gemini's free tier for inference.

## Future Improvements
- Support multiple PDFs / a persistent document library instead of one
  session-scoped PDF.
- Add citation highlighting that jumps to the exact page/paragraph in a PDF
  preview pane.
- Swap in a re-ranking step (e.g. cross-encoder) before answer generation
  for better precision on longer documents.
- Add streaming token-by-token responses in the Gradio UI.
