import os
import bs4
import config
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# This script demonstrates how to create a vector store with document embeddings,
# retrieve information from it, and use a large language model for answering questions.

# Define a function to create and persist a vector store from web content
def create_vector_store():
    # Initialize a web loader for scraping the specified webpage
    loader = WebBaseLoader(
        web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent'],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=(
                'post-header', 'post-title', 'post-content'
            ))
        )
    )
    # Load the webpage content into a document object
    docs = loader.load()
    
    # Use a text splitter to break the content into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=4)
    splits = splitter.split_documents(docs)
    
    # Define a directory to persist the vector store data
    persist_dir = 'ollama_chroma_data_dir'
    
    # Create a vector store using Chroma with embeddings from the specified Ollama model
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="qwen2:7b"),
        persist_directory=persist_dir
    )

# Define a function to load the persisted vector store and retrieve information
def load_vector_store():
    # Specify the directory where the vector store data is stored
    persist_dir = 'ollama_chroma_data_dir'
    
    # Load the vector store from the directory
    vector_store = Chroma(
        embedding_function=OllamaEmbeddings(model="qwen2:7b"),
        persist_directory=persist_dir
    )
    
    # Convert the vector store into a retriever for querying
    retriever = vector_store.as_retriever()
    
    # Define the system-level prompt to guide the assistant's response style
    system_prompt = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer
    the question. If you don't find the answer, say that you
    don't know. Use three sentences maximum and keep the answer concise. \n

    {context}
    """
    
    # Create a prompt template for interacting with the language model
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', '{input}')
    ])
    
    # Initialize the Ollama chat model with the specified version
    model = ChatOllama(
        model='qwen2:7b'
    )
    
    # Create a document-combination chain to generate answers from retrieved documents
    chain1 = create_stuff_documents_chain(model, prompt)
    
    # Create a retrieval chain that combines retrieval and answer generation
    chain2 = create_retrieval_chain(retriever, chain1)
    
    # Query the chain with a specific question
    resp = chain2.invoke({
        'input': 'What is Task Decomposition?'
    })
    # Print the response to the console
    print(resp)

# Uncomment the following line to create the vector store
# create_vector_store()

# Load the vector store and retrieve information
load_vector_store()
