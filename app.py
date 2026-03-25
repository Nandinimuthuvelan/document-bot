import streamlit as st
st.set_page_config(page_title="AI Document Bot", layout="wide")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.title("AI Document Bot")

mode = st.selectbox(
    "What do you want to do?",
    ["Ask Question", "Summarize", "Get Insights"]
)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If no file
if uploaded_file is None:
    st.warning("Please upload a PDF first")

else:
    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Check empty docs
    if len(docs) == 0:
        st.error("No content found in the PDF.")
        st.stop()

    # Create vector DB
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    # Chat input (IMPORTANT: inside else block)
    query = st.chat_input("Ask something about the document...")

    if query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Search context
        results = db.similarity_search(query)
        context = " ".join([doc.page_content for doc in results])

        # Light model (works on cloud)
        generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")

        # Prompt logic
        if mode == "Ask Question":
            prompt = f"Answer based on this:\n{context}\n\nQuestion: {query}"

        elif mode == "Summarize":
            prompt = f"Summarize this:\n{context}"

        elif mode == "Get Insights":
            prompt = f"Give key insights from this:\n{context}"

        result = generator(prompt, max_length=200)
        answer = result[0]["generated_text"]

        # Show bot response
        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


