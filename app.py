import streamlit as st
st.set_page_config(page_title="Document Intelligence Assistant", layout="wide")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# ---- HEADER ----
st.markdown("""
# 📄 Document Intelligence Assistant
Upload a document and interact with it using AI.
""")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("Controls")
    mode = st.radio(
        "Choose Action",
        ["Ask Question", "Summarize", "Get Insights"]
    )
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ---- CHAT MEMORY ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- DISPLAY CHAT ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ---- MAIN LOGIC ----
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    docs = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    ).split_documents(
        PyPDFLoader("temp.pdf").load()
    )

    if len(docs) == 0:
        st.error("Unable to read document.")
        st.stop()

    db = FAISS.from_documents(
        docs,
        HuggingFaceEmbeddings()
    )

    st.success("Document processed successfully")

    query = st.chat_input("Type your request...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        context = " ".join(
            [d.page_content for d in db.similarity_search(query)]
        )

        generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")

        if mode == "Ask Question":
            prompt = f"{context}\nQuestion: {query}"
        elif mode == "Summarize":
            prompt = f"Summarize:\n{context}"
        else:
            prompt = f"Insights:\n{context}"

        answer = generator(prompt, max_length=200)[0]["generated_text"]

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info("Upload a PDF from the sidebar to begin")
