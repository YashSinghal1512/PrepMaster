import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Set your Google API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAQo3wHs74mJA10Dp5kvDzDEPkgvtZtYq4'

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store using FAISS and Google Palm embeddings
def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to create a conversational chain using Google Palm LLM and the vector store
def get_conversational_chain(vector_store):
    llm = GooglePalm(model_name="models/text-bison-001", google_api_key=os.getenv('GOOGLE_API_KEY'))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display conversation
def user_input(user_question, conversation_chain):
    response = conversation_chain({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("Bot: ", message.content)

# Streamlit app configuration
st.set_page_config("Chat with Multiple PDFs")
st.header("Chat with your notes 💬")
st.markdown(
    """
    Seamlessly tackle interview questions from your PDFs, ensuring a comprehensive and personalized preparation experience.
    """
)

# Text input for user questions
user_question = st.text_input("Ask Questions from your notes")

# Initialize conversation and chat history in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chatHistory" not in st.session_state:
    st.session_state.chatHistory = None

# Handle user question input
if user_question:
    user_input(user_question, st.session_state.conversation)

# Sidebar settings for file upload
with st.sidebar:
    st.title("Settings")
    st.subheader("Upload your Documents")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
    
    # Process button to handle PDF processing
    if st.button("Process"):
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversational_chain(vector_store)
            st.success("Done")
