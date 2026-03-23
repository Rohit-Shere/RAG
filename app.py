import os
import streamlit as st
from dotenv import load_dotenv

# 2026 Standard Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile

load_dotenv()

def get_pdf_text_and_chunks(pdf_docs):
    """Processes uploaded PDFs into Document chunks."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    for pdf in pdf_docs:
        # Save temp file for PyMuPDFLoader to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.getvalue())
            tmp_path = tmp.name
        
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        all_chunks.extend(splitter.split_documents(docs))
        os.remove(tmp_path)
    return all_chunks

def get_vectorstore(chunks):
    """Initializes Chroma (Auto-persists in 2026)."""
    # Updated to 2026 stable embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name="pdf_chat_collection",
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    # vectorstore.persist() is no longer required in 2026
    return vectorstore

def get_conversation_chain(vectorstore):
    """Sets up the chat chain with the 2026 production-grade Gemini model."""
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash-lite', # Optimized 2026 choice
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

def handle_userinput(user_question):
    # Updated to 'invoke' for modern LangChain standards
    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)

def main():
    st.set_page_config(page_title="AI PDF Chat 2026", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation:
            handle_userinput(user_question)
        else:
            st.warning("Please upload and process your PDFs first.")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Analyzing..."):
                chunks = get_pdf_text_and_chunks(pdf_docs)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Ready to chat!")

if __name__ == '__main__':
    main()
