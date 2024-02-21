import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from htmlTemplates import css
from langchain_community.vectorstores.chroma import Chroma 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.chat_models import ChatOpenAI
from htmlTemplates import css,bot_template,user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for pages in pdf_reader.pages:
            text += pages.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.create_documents([raw_text])
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=os.environ.get('OPENAI_API_KEY'))
    vectordb = Chroma.from_documents(documents=text_chunks,embedding=embeddings)

def get_conversation_chain(user_question):
    llm = ChatOpenAI(model="gpt-4-0613",temperature=0.1,api_key= os.environ.get('OPENAI_API_KEY'))
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=os.environ.get('OPENAI_API_KEY'))
    vectordb = Chroma(embedding_function=embeddings)
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                    retriever=vectordb.as_retriever(),
                                                    memory=memory
                                                               )
    response = conversation_chain.invoke({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
def main():
    load_dotenv()   
    st.set_page_config('Chat with Multiple PDFs',page_icon=':books:')
    st.write(css,unsafe_allow_html=True)
    st.header('Chat with Multiple PDFs')
    user_question = st.text_input('Ask your question about the PDF')
    
    with st.sidebar:
        st.subheader('Upload files')
        pdf_docs = st.file_uploader('Upload your PDFs here and click on "Process"',
                                    accept_multiple_files=True,type=['pdf'])
        if st.button('Process'):
            with st.spinner('Processing'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                
    if st.button('Generate') and user_question:
        with st.spinner('Processing'):
            get_conversation_chain(user_question)
              
    
if __name__ == '__main__':
    main()
