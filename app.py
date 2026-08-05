import gradio as gr
from pathlib import Path

from rag import PDFChatbot


bot = PDFChatbot()


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

def upload_pdf(pdf):

    if pdf is None:
        return "No PDF uploaded."

    filename = Path(pdf.name).name

    chunks = bot.create_vector_db(pdf.name)

    return f"""
Document : {filename}

Status : Ready

Chunks : {chunks}
"""


# --------------------------------------------------
# Chat
# --------------------------------------------------

def chat(message, history):

    if not message.strip():
        return "", history

    answer = bot.ask(message)

    history.append(
        {"role": "user", "content": message}
    )

    history.append(
        {"role": "assistant", "content": answer}
    )

    return "", history


# --------------------------------------------------
# Clear
# --------------------------------------------------

def clear():

    bot.chat_history = []

    return []


# --------------------------------------------------
# UI
# --------------------------------------------------

with gr.Blocks(
    title="DocuMind AI",
    theme=gr.themes.Soft(),
    fill_height=True
) as demo:

    gr.Markdown(
        """
# 📄 DocuMind AI

### Chat with your PDF using Gemini + RAG
"""
    )

    with gr.Row():

        # Sidebar

        with gr.Column(scale=1):

            pdf = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )

            status = gr.Textbox(
                label="Status",
                interactive=False
            )

            clear_btn = gr.Button(
                "Clear Chat"
            )

        # Chat

        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                type="messages",
                height=550,
                label="Conversation"
            )

            msg = gr.Textbox(
                placeholder="Ask anything about the uploaded PDF...",
                show_label=False
            )

    # Events

    pdf.change(
        upload_pdf,
        pdf,
        status
    )

    msg.submit(
        chat,
        [msg, chatbot],
        [msg, chatbot]
    )

    clear_btn.click(
        clear,
        outputs=chatbot
    )


demo.launch()