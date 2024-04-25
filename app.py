from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

st.title("RAG System on Leave No Context Behind Paper")

loader = PyPDFLoader("2404.07143.pdf")
data = loader.load_and_split()



from langchain_text_splitters import NLTKTextSplitter

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(data)



embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyC2Bztff9XtDCDrCJfMJ8py9JaT8VkwSlY", 
                                               model="models/embedding-001")


# Embed each chunk and load it into the vector store
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")

# Persist the database on drive
db.persist()

db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

retriever = db_connection.as_retriever(search_kwargs={"k": 5})



chat_model = ChatGoogleGenerativeAI(google_api_key="AIzaSyC2Bztff9XtDCDrCJfMJ8py9JaT8VkwSlY", 
                                   model="gemini-1.5-pro-latest")

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
]) 
output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

question=st.text_input("Enter your query ")
if st.button("Search"):

    response = rag_chain.invoke(question)
    st.write(response)







