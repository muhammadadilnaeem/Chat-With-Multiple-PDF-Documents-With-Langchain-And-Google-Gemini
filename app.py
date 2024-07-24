# Importing Required Libraries
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Setting Up Environment Variables
import os
from dotenv import load_dotenv
load_dotenv()

# Configure GOOGLE_API_KEY
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define a function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Define a function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Define a function to get vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Define a function to get conversation chain from vector store and prompt template
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Define a function to get user input and process it
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

# Define the main function to run the app and display the UI
def main():
    st.set_page_config(page_title="📚 Chat Multiple PDF")
    st.markdown("<h1 style='color: #4CAF50;'>📚 Chat with Multiple PDF using Gemini</h1>", unsafe_allow_html=True)
    
    user_question = st.text_input("💬 Ask a Question from the PDF Files")
    if user_question:
        response = user_input(user_question)
        st.markdown(f"<h2 style='color: #2196F3;'>🔍 Response:</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #000000; background-color: #E8F5E9; padding: 10px; border-radius: 5px;'>{response}</div>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<h2 style='color: #FF5722;'>📂 Menu:</h2>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("📄 Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("🚀 Submit & Process"):
            with st.spinner("⏳ Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Done")

if __name__ == "__main__":
    main()
