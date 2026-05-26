import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Streamlit Page
st.set_page_config(page_title="PDF RAG Chatbot")

st.title("PDF RAG Chatbot")

# GROQ API KEY
groq_api_key = st.text_input(
    "Enter GROQ API KEY",
    type="password"
)

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file and groq_api_key:

    # Save PDF Temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF Uploaded Successfully")

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Database
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # Retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    # LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question only from the provided context.

If answer is not available in context,
say "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    # User Question
    question = st.text_input(
        "Ask Question"
    )

    # Generate Response
    if st.button("Generate Response"):

        # Retrieve Documents
        retrieved_docs = retriever.invoke(question)

        # Combine Context
        context = "\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        # Final Prompt
        final_prompt = prompt.format(
            context=context,
            question=question
        )

        # LLM Response
        response = llm.invoke(final_prompt)

        st.subheader("Answer")

        st.write(response.content)