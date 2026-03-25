import streamlit as st
st.set_page_config(page_title="AI Document Bot", layout="wide")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.title("My First AI Document Bot")

mode = st.selectbox(
    "What do you want to do?",
    ["Ask Question", "Summarize", "Get Insights"]
)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    st.write("Document ready. Ask a question below 👇")

    query = st.chat_input("Ask something about the document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    results = db.similarity_search(query)
    context = " ".join([doc.page_content for doc in results])

    generator = pipeline("text-generation", model="google/flan-t5-small")

    if mode == "Ask Question":
        prompt = f"Answer clearly based on the document:\n{context}\n\nQuestion: {query}"

    elif mode == "Summarize":
        prompt = f"Give a short and clear summary of this document:\n{context}"

    elif mode == "Get Insights":
        prompt = f"Explain the key insights and implications from this document in simple words:\n{context}"

    result = generator(prompt, max_length=200)
    answer = result[0]["generated_text"]

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })