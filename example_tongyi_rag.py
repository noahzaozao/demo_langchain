import os
import bs4
import config
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# Function to create a vector store using Chroma and embeddings.
def create_vector_store():
    # Load web content from the specified URL.
    loader = WebBaseLoader(
        web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent'],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=(
                'post-header', 'post-title', 'post-content'  # Define the HTML classes to parse.
            ))
        )
    )
    docs = loader.load()  # Load documents from the web.
    
    # Split the loaded documents into smaller chunks for processing.
    splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=4)
    splits = splitter.split_documents(docs)
    
    # Define the directory to persist the vector store data.
    persist_dir = 'chroma_data_dir'
    
    # Create a Chroma vector store using the document splits and embeddings.
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=DashScopeEmbeddings(),  # Use DashScopeEmbeddings for vectorization.
        persist_directory=persist_dir  # Specify the directory to save data.
    )


# Function to load the vector store and execute a retrieval-based chain.
def load_vector_store():
    persist_dir = 'chroma_data_dir'  # Specify the directory containing the saved vector store.
    
    # Load the Chroma vector store using precomputed embeddings.
    vector_store = Chroma(
        embedding_function=DashScopeEmbeddings(),
        persist_directory=persist_dir
    )
    
    # Configure the retriever for retrieving relevant context.
    retriever = vector_store.as_retriever()
    
    # Define a system prompt for the assistant.
    system_prompt = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer
    the question. If you don't find the answer, say that you
    don't know. Use three sentences maximum and keep the answer concise. \n

    {context}
    """
    
    # Create a prompt template for the chain.
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),  # Add the system message for guidance.
        ('human', '{input}')  # Placeholder for user input.
    ])
    
    # Initialize the language model with specified parameters.
    model = Tongyi(temperature=1)
    
    # Create a chain to process retrieved documents.
    chain1 = create_stuff_documents_chain(model, prompt)
    
    # Create a retrieval-based chain by combining the retriever and document processing chain.
    chain2 = create_retrieval_chain(retriever, chain1)
    
    # Invoke the chain with a specific input question.
    resp = chain2.invoke({
        'input': 'What is Task Decomposition?'  # Define the user query.
    })
    
    # Print the response from the chain.
    print(resp)


# Uncomment the following line to create the vector store:
# create_vector_store()

# Load the vector store and retrieve answers.
load_vector_store()
